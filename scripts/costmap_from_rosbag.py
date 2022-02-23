#!/usr/bin/env python3

import rospy
import math
import tf
import pickle
import os
import rosbag
import numpy as np
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32, QuaternionStamped, Quaternion, TwistWithCovariance, PoseStamped, TransformStamped
from tf.transformations import quaternion_from_euler
from vox_msgs.msg import VoxGrid
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import PointCloud2
from teb_local_planner.msg import FeedbackMsg, TrajectoryMsg, TrajectoryPointMsg
import tf2_geometry_msgs

import matplotlib.pyplot as plt


def feedback_callback(data):
    global trajectory

    if not data.trajectories:
        trajectory = []
        return
    trajectory = data.trajectories[data.selected_trajectory_idx].trajectory

def load_saved_costmaps(file_path):

    with open(file_path, 'rb') as f:
        collider_data = pickle.load(f)

    return collider_data

def get_collisions_msg(collision_preds, t0, origin0, dl_2D, time_resolution):

    # Define header
    msg = VoxGrid()
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = 'odom'

    # Define message
    msg.depth = collision_preds.shape[0]
    msg.width = collision_preds.shape[1]
    msg.height = collision_preds.shape[2]
    msg.dl = dl_2D
    msg.dt = time_resolution
    msg.origin.x = origin0[0]
    msg.origin.y = origin0[1]
    msg.origin.z = -1.0

    #msg.theta = q0[0]
    msg.theta = 0
    msg.data = collision_preds.ravel().tolist()

    return msg

def get_collisions_visu_msg(collision_preds, t0, origin0, dl_2D, visu_T=15):
    '''
    0 = invisible
    1 -> 98 = blue to red
    99 = cyan
    100 = yellow
    101 -> 127 = green
    128 -> 254 = red to yellow
    255 = vert/gris
    '''

    # Define header
    msg = OccupancyGrid()
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = 'odom'

    # Define message meta data
    msg.info.map_load_time = rospy.Time.from_sec(t0)
    msg.info.resolution = dl_2D
    msg.info.width = collision_preds.shape[1]
    msg.info.height = collision_preds.shape[2]
    msg.info.origin.position.x = origin0[0]
    msg.info.origin.position.y = origin0[1]
    msg.info.origin.position.z = -0.01
    #msg.info.origin.orientation.x = q0[0]
    #msg.info.origin.orientation.y = q0[1]
    #msg.info.origin.orientation.z = q0[2]
    #msg.info.origin.orientation.w = q0[3]

    # Define message data
    data_array = collision_preds[visu_T, :, :].astype(np.float32)
    mask = collision_preds[visu_T, :, :] > 253
    mask2 = np.logical_not(mask)
    data_array[mask2] = data_array[mask2] * 98 / 253
    data_array[mask2] = np.maximum(1, np.minimum(98, data_array[mask2] * 1.0))
    data_array[mask] = 98  # 101
    data_array = data_array.astype(np.int8)
    msg.data = data_array.ravel()

    return msg

def get_pred_points(collision_preds, t0, origin0, dl_2D, dt):

    # Get mask of the points we want to show

    mask = collision_preds > 0.8 * 255

    nt, nx, ny = collision_preds.shape
    

    t = np.arange(0, nt, 1)
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)

    tv, yv, xv = np.meshgrid(t, x, y, indexing='ij')
    

    tv = tv[mask].astype(np.float32)
    xv = xv[mask].astype(np.float32)
    yv = yv[mask].astype(np.float32)


    xv = origin0[0] + (xv + 0.5) * dl_2D
    yv = origin0[1] + (yv + 0.5) * dl_2D
    tv *= dt * 0.5

    labels = collision_preds[mask].astype(np.float32) / 255

    return np.stack((xv, yv, tv), 1), labels

def get_pointcloud_msg(new_points, labels):

    # data structure of binary blob output for PointCloud2 data type
    output_dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring'],
                             'formats': ['<f4', '<f4', '<f4', '<f4', '<u2'],
                             'offsets': [0, 4, 8, 16, 20],
                             'itemsize': 32})

    # fill structured numpy array with points and classes (in the intensity field). Fill ring with zeros to maintain Pointcloud2 structure
    c_points = np.c_[new_points, labels, np.zeros(len(labels))]
    c_points = np.core.records.fromarrays(c_points.transpose(), output_dtype)

    # convert to Pointcloud2 message and publish
    msg = pc2.array_to_pointcloud2(c_points, rospy.get_rostime(), 'odom')

    return msg

def plot_velocity_profile(fig, ax_v, ax_omega, t, v, omega):
    ax_v.cla()
    ax_v.grid()
    ax_v.set_ylabel('Trans. velocity [m/s]')
    ax_v.plot(t, v, '-bx')
    ax_omega.cla()
    ax_omega.grid()
    ax_omega.set_ylabel('Rot. velocity [rad/s]')
    ax_omega.set_xlabel('Time [s]')
    ax_omega.plot(t, omega, '-bx')
    fig.canvas.draw()


#
#
#
#
##########################################################################################################################################################
#
#
#
#


def read_collider_preds(topic_name, bagfile):
    '''returns list of Voxgrid message'''

    all_header_stamp = []
    all_dims = []
    all_origin = []
    all_dl = []
    all_dt = []
    all_preds = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        all_header_stamp.append(msg.header.stamp.to_sec())
        all_dims.append([msg.depth, msg.width, msg.height])
        all_origin.append([msg.origin.x, msg.origin.y, msg.origin.z])
        all_dl.append(msg.dl)
        all_dt.append(msg.dt)
        array_data = np.frombuffer(msg.data, dtype=np.uint8)
        all_preds.append(array_data.tolist())

    collider_data = {}

    collider_data['header_stamp'] = np.array(all_header_stamp, dtype=np.float64)
    collider_data['dims'] = np.array(all_dims, dtype=np.int32)
    collider_data['origin'] = np.array(all_origin, dtype=np.float64)
    collider_data['dl'] = np.array(all_dl, dtype=np.float32)
    collider_data['dt'] = np.array(all_dt, dtype=np.float32)
    collider_data['preds'] = np.array(all_preds, dtype=np.uint8)

    return collider_data

def read_local_plans(topic_name, bagfile):
    '''returns list of local plan message'''

    local_plans = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        path_dict = {}
        path_dict['header_stamp'] = msg.header.stamp.to_sec()
        path_dict['header_frame_id'] = msg.header.frame_id

        pose_list = []
        for msg_pose in msg.poses:
            pose = (msg_pose.pose.position.x,
                    msg_pose.pose.position.y,
                    msg_pose.pose.position.z,
                    msg_pose.pose.orientation.x,
                    msg_pose.pose.orientation.y,
                    msg_pose.pose.orientation.z,
                    msg_pose.pose.orientation.w)
            pose_list.append(pose)

        path_dict['pose_list'] = pose_list

        local_plans.append(path_dict)

    return local_plans

def read_tf_transform(parent_frame, child_frame, bagfile, static=False):
    ''' returns a list of time stamped transforms between parent frame and child frame '''
    arr = []
    if (static):
        topic_name = "/tf_static"
    else:
        topic_name = "/tf"

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):
        for transform in msg.transforms:
            if (transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame):
                arr.append(transform)

    return arr

def transforms_to_trajectory(transforms):
    ''' converts a list of time stamped transforms to a list of pose stamped messages, where the pose is that of the child frame relative to it's parent'''
    traj = []
    for transform in transforms:
        geo_msg = PoseStamped()
        geo_msg.header = transform.header
        geo_msg.header.frame_id = transform.child_frame_id
        geo_msg.pose.position = transform.transform.translation
        geo_msg.pose.orientation = transform.transform.rotation
        traj.append(geo_msg)

    return traj

def interpolate_pose(time, pose1, pose2):
    ''' given a target time, and two PoseStamped messages, find the interpolated pose between pose1 and pose2 ''' 

    t1 = pose1.header.stamp.to_sec()
    t2 = pose2.header.stamp.to_sec()

    alpha = 0
    if (t1 != t2):
        alpha = (time-t1)/(t2-t1)

    pos1 = pose1.pose.position
    pos2 = pose2.pose.position

    rot1 = pose1.pose.orientation
    rot1 = [rot1.x,rot1.y,rot1.z,rot1.w]
    rot2 = pose2.pose.orientation
    rot2 = [rot2.x,rot2.y,rot2.z,rot2.w]

    res = PoseStamped()

    res.header.stamp = rospy.Time(time)
    res.header.frame_id = pose1.header.frame_id

    res.pose.position.x = pos1.x + (pos2.x - pos1.x)*alpha
    res.pose.position.y = pos1.y + (pos2.y - pos1.y)*alpha
    res.pose.position.z = pos1.z + (pos2.z - pos1.z)*alpha

    res_rot = tf.transformations.quaternion_slerp(rot1,rot2,alpha)
    res.pose.orientation.x = res_rot[0]
    res.pose.orientation.y = res_rot[1]
    res.pose.orientation.z = res_rot[2]
    res.pose.orientation.w = res_rot[3]

    return res

def interpolate_transform(time, trans1, trans2):
    ''' given a target time, and two TransformStamped messages, find the interpolated transform ''' 

    t1 = trans1.header.stamp.to_sec()
    t2 = trans2.header.stamp.to_sec()

    alpha = 0
    if (t1 != t2):
        alpha = (time-t1)/(t2-t1)

    pos1 = trans1.transform.translation
    pos2 = trans2.transform.translation

    rot1 = trans1.transform.rotation
    rot1 = [rot1.x,rot1.y,rot1.z,rot1.w]
    rot2 = trans2.transform.rotation
    rot2 = [rot2.x,rot2.y,rot2.z,rot2.w]

    res = TransformStamped()

    res.header.stamp = rospy.Time(time)
    res.header.frame_id = trans1.header.frame_id

    res.transform.translation.x = pos1.x + (pos2.x - pos1.x)*alpha
    res.transform.translation.y = pos1.y + (pos2.y - pos1.y)*alpha
    res.transform.translation.z = pos1.z + (pos2.z - pos1.z)*alpha

    res_rot = tf.transformations.quaternion_slerp(rot1,rot2,alpha)
    res.transform.rotation.x = res_rot[0]
    res.transform.rotation.y = res_rot[1]
    res.transform.rotation.z = res_rot[2]
    res.transform.rotation.w = res_rot[3]

    return res


def get_interpolations(target_times, trajectory, transform = True):
    '''
    given two trajectories, interploate the poses in trajectory to the times given in target_times (another trajectory)
    this modifies target_times so it stays in the range of trajectory's interpolations
    if transform = True, then trajectory stores transform messages not PoseStamped
    '''

    min_time = trajectory[0].header.stamp.to_sec()
    max_time = trajectory[-1].header.stamp.to_sec()

    res = []

    last = 0


    i = 0
    while i < len(target_times):

        target = target_times[i]
        time = target.header.stamp.to_sec()

        if (time < min_time or time > max_time):
            target_times.pop(i)
            continue
        
        lower_ind = last



        while (trajectory[lower_ind].header.stamp.to_sec() > time or trajectory[lower_ind+1].header.stamp.to_sec() < time):
            if (trajectory[lower_ind].header.stamp.to_sec() > time):
                lower_ind-=1
            else:
                lower_ind+=1
        
        #last = lower_ind +1

    

        if ((i+1) < len(target_times)):
            next_time = target_times[i+1].header.stamp.to_sec()
            if (next_time >= trajectory[lower_ind+1]):
                last = lower_ind+1
            else:
                last = lower_ind
        else:
            last = lower_ind

        #last = (lower_ind+1) if ((lower_ind+2)<len(trajectory)) else lower_ind

        if (transform):
            inter = interpolate_transform(time, trajectory[lower_ind], trajectory[lower_ind+1])
        else:
            inter = interpolate_pose(time, trajectory[lower_ind], trajectory[lower_ind+1])


        res.append(inter)
        i+=1
    
    return res


def transform_trajectory(trajectory, transformations):
    ''' translate each point in trajectory by a transformation interpolated to the correct time, return the transformed trajectory'''

    # for each point in trajectory, find the interpolated transformation, then transform the trajectory point

    matching_transforms = get_interpolations(trajectory, transformations)

    res = []

    for i in range(len(matching_transforms)):
        trans = matching_transforms[i]
        traj_pose = trajectory[i]

        transformed = tf2_geometry_msgs.do_transform_pose(traj_pose, trans)
        res.append(transformed)

    return res






def publish_costmap_msg(traj_debug=False):
    global trajectory
    
    # if traj_debug:
    #     topic_name = "/test_optim_node/teb_feedback"
    #     topic_name = rospy.get_param('~feedback_topic', topic_name)
    #     rospy.Subscriber(topic_name, FeedbackMsg, feedback_callback, queue_size=1)

    #     rospy.loginfo("Visualizing velocity profile published on '%s'.",topic_name) 
    #     rospy.loginfo("Make sure to enable rosparam 'publish_feedback' in the teb_local_planner.")
            
    #     fig, (ax_v, ax_omega) = plt.subplots(2, sharex=True)
    #     plt.ion()
    #     plt.show()

    # Rosbag path
    bag_path =  os.path.join(os.environ.get('HOME'), 'results/rosbag_data')

    # List all bag files
    bag_files = np.sort([f for f in os.listdir(bag_path) if f.endswith('.bag')])

    # Chose one
    bag_chosen = os.path.join(bag_path, bag_files[-1])

    # Read Bag file
    print("")
    print("Initializing data for file :" + bag_chosen)
    try:
        bag = rosbag.Bag(bag_chosen)
    except:
        print("ERROR: invalid filename")
    print("OK")
    print("")


    # Read SOGMs
    # Read TEB plan
    # Read TEB planned poses
    # Read trajectory

    # Select a point on the trajectory have two axes 
    #   one with the full traj an a point on the currentyl selected pose
    #   another with the first layer of predicted sogm for this timestamp

    # Publish the coressponding stuff:
    #   - Costmap centered on 0 for the test_optim_node
    #   - Include delay in the costmap publication
    #   - Coresponding TEB plan at this time

    # Add possibility of modification
    #   - translate/rotate the costmap for testing
    #   - Add reduce delay

    # Now play with parameters


    ##################
    # Reading messages
    ##################

    print("")
    print("Reading SOGMs")
    collider_preds = read_collider_preds("/plan_costmap_3D", bag)
    print("OK")
    print("")

    print("")
    print("Reading TEB plans")
    teb_local_plans = read_local_plans("/move_base/TebLocalPlannerROS/local_plan", bag)
    print("OK")
    print("")

    print("")
    print("Reading tf traj")
    map_frame = "map"
    odom_to_base = read_tf_transform("odom","base_link", bag)
    map_to_odom = read_tf_transform(map_frame,"odom", bag)



    pose = tfBuffer.lookup_transform('map', 'velodyne', stamp)
    T_q = np.array([pose.transform.translation.x,
                    pose.transform.translation.y,
                    pose.transform.translation.z,
                    pose.transform.rotation.x,
                    pose.transform.rotation.y,
                    pose.transform.rotation.z,
                    pose.transform.rotation.w], dtype=np.float64)
    self.poses[f_i] = T_q

    
    print()
    print(len(odom_to_base))
    print(odom_to_base[0])
    print(type(odom_to_base[0]))

    print()
    print(len(map_to_odom))
    print(map_to_odom[0])
    print(type(map_to_odom[0]))
    print()


    a = 1/0



    odom_to_base = transforms_to_trajectory(odom_to_base)
    tf_traj = transform_trajectory(odom_to_base, map_to_odom)
    print("OK")
    print("")


    ############
    # Create GUI
    ############

    # Use replayer.py for the loading of info
    # Use wanted_inds GUI for picking point in trajectory with slider
    # Use slider for translation / rotations


    print(len(tf_traj))
    print(tf_traj[0])
    print(type(tf_traj[0]))


    a = 1/0



    # Get annotated lidar frames
    lidar_path = join(simu_path, day, lidar_folder)
    classif_path = join(simu_path, day, classif_folder)

    f_names = [f for f in listdir(lidar_path) if f[-4:] == '.ply']
    f_times = np.array([float(f[:-4]) for f in f_names], dtype=np.float64)
    f_names = np.array([join(lidar_path, f) for f in f_names])
    ordering = np.argsort(f_times)
    f_names = f_names[ordering]
    f_times = f_times[ordering]

    # Load mapping poses
    map_traj_file = join(simu_path, day, 'logs-'+day, 'map_traj.ply')
    data = read_ply(map_traj_file)
    map_T = np.vstack([data['pos_x'], data['pos_y'], data['pos_z']]).T
    map_Q = np.vstack([data['rot_x'], data['rot_y'], data['rot_z'], data['rot_w']]).T

    # Times
    day_map_t = data['time']

    # Convert map to homogenous rotation/translation matrix
    map_R = scipyR.from_quat(map_Q)
    map_R = map_R.as_matrix()
    day_map_H = np.zeros((len(day_map_t), 4, 4))
    day_map_H[:, :3, :3] = map_R
    day_map_H[:, :3, 3] = map_T
    day_map_H[:, 3, 3] = 1

    # Filter valid frames
    f_names, day_map_t, day_map_H = filter_valid_frames(f_names, day_map_t, day_map_H)

    # Load gt_poses
    gt_t, gt_H = load_gt_poses(simu_path, day)
    
    # Init loc abd gt traj
    gt_traj = gt_H[:, :3, 3]
    gt_traj[:, 2] = gt_t
    loc_traj = day_map_H[:, :3, 3]
    loc_traj[:, 2] = day_map_t






















    # Load costmaps to publish
    # simu_path = '/home/hth/Myhal_Simulation/simulated_runs'
    simu_path = '/home/administrator/1-Deep-Collider/simulated_runs'
    #folder = '2021-06-07-21-44-58'
    #pred_file = os.path.join(simu_path, folder, 'logs-' + folder, 'collider_data.pickle')
    pred_file = os.path.join(simu_path, 'collider_data.pickle')
    collider_data = load_saved_costmaps(pred_file)

    dl = collider_data['dl'][0]
    dt0 = collider_data['dt'][0]
    dt = collider_data['dt'][0]
    pred_times = collider_data['header_stamp']

    # Init
    collision_pub = rospy.Publisher('/plan_costmap_3D', VoxGrid, queue_size=1)
    visu_pub = rospy.Publisher('/collision_visu', OccupancyGrid, queue_size=1)
    pointcloud_pub = rospy.Publisher('/colli_points', PointCloud2, queue_size=10)
    #pub = rospy.Publisher('/p3dx/move_base/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=1)
    rospy.init_node("test_obstacle_msg")

    preds_i = 50

    dims = collider_data['dims'][preds_i]
    preds = np.reshape(collider_data['preds'][preds_i], dims)
    
    origin0 = np.copy(collider_data['origin'][0])
    t0 = origin0[2]

    y_0 = -3.0
    vel_y = 0.3
    range_y = 6.0

    r = rospy.Rate(2)  # 10hz
    t = 0.0
    visu_T = 0
    while not rospy.is_shutdown():

        ####################
        # Publishing pred 3D
        ####################

        # Vary The costmap layer to publish
        #visu_T = (visu_T + 1) % preds.shape[0]

        new_origin = np.copy(origin0)

        # new_origin[0] = origin0[0] + 2.0 * np.sin(5.31 * t)
        # new_origin[1] = origin0[1] + 3.0 * np.sin(t)

        new_origin[0] = origin0[0] + 3.5
        new_origin[1] = origin0[1] + 0
        
        # Get messages
        collision_msg = get_collisions_msg(preds, t0, new_origin, dl, dt)
        visu_msg = get_collisions_visu_msg(preds, t0, new_origin, dl, visu_T)
        points, labels = get_pred_points(preds, t0, new_origin, dl, dt)
        pt_msg = get_pointcloud_msg(points, labels)

        # Publish
        collision_pub.publish(collision_msg)
        visu_pub.publish(visu_msg)
        pointcloud_pub.publish(pt_msg)

        ###################
        # Plotting feedback
        ###################

        if traj_debug:
            ts = []
            vs = []
            omegas = []
            
            for point in trajectory:
                ts.append(point.time_from_start.to_sec())
                vs.append(point.velocity.linear.x)
                omegas.append(point.velocity.angular.z)
                
            plot_velocity_profile(fig, ax_v, ax_omega, np.asarray(ts), np.asarray(vs), np.asarray(omegas))




        t = t + 0.05
        r.sleep()

if __name__ == '__main__':
    global trajectory
    try:
        trajectory = []
        publish_costmap_msg(traj_debug=False)
    except rospy.ROSInterruptException:
        pass