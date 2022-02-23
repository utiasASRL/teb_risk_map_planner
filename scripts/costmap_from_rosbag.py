#!/usr/bin/env python3

import rospy
import math
import tf
import pickle
import os
import rosbag
import tf2_ros
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
from matplotlib.widgets import Slider, Button, RadioButtons


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
    collider_data = read_collider_preds("/plan_costmap_3D", bag)
    print("OK")
    print("")

    print("")
    print("Reading TEB plans")
    teb_local_plans = read_local_plans("/move_base/TebLocalPlannerROS/local_plan", bag)
    print("OK")
    print("")

    print("")
    print("Reading tf traj")
    
    # Create tf buffer and listenner
    tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(20000))

    # Fill the buffer with tf and tf_static msgs
    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for msg_tf in msg.transforms:
            if topic == '/tf_static':
                tfBuffer.set_transform_static(msg_tf, "default_authority")
            else:
                tfBuffer.set_transform(msg_tf, "default_authority")


    # Now get transform at every timestamp we need
    all_origin_times = [None for _ in collider_data['header_stamp']]
    all_origin_poses = [None for _ in collider_data['header_stamp']]
    all_computed_times = [None for _ in collider_data['header_stamp']]
    all_computed_poses = [None for _ in collider_data['header_stamp']]
    valid_collider = np.ones((collider_data['header_stamp'].shape[0],), dtype=bool)
    for i, (stamp_secs, origin) in enumerate(zip(collider_data['header_stamp'], collider_data['origin'])):
        stamp0 = rospy.Time.from_sec(origin[2])
        stamp1 = rospy.Time.from_sec(stamp_secs)

        for stamp, time_list, pose_list in zip([stamp0, stamp1],
                                               [all_origin_times, all_computed_times],
                                               [all_origin_poses, all_computed_poses]):

            try:
                pose = tfBuffer.lookup_transform('map', 'velodyne', stamp)

            except (tf2_ros.InvalidArgumentException, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                valid_collider[i] = False
                continue

            T_q = np.array([pose.transform.translation.x,
                            pose.transform.translation.y,
                            pose.transform.translation.z,
                            pose.transform.rotation.x,
                            pose.transform.rotation.y,
                            pose.transform.rotation.z,
                            pose.transform.rotation.w], dtype=np.float64)

            time_list[i] = stamp.to_sec()
            pose_list[i] = T_q

    print("OK")
    print("")


    ##############
    # Prepare Data
    ##############

    # Filter invalid data
    valid_collider = np.array(valid_collider, dtype=bool)

    for k, v in collider_data.items():
        collider_data[k] = v[valid_collider]

    all_origin_poses = [_ for _, is_valid in zip(all_origin_poses, valid_collider) if is_valid]
    all_computed_poses = [_ for _, is_valid in zip(all_computed_poses, valid_collider) if is_valid]
    all_origin_times = [_ for _, is_valid in zip(all_origin_times, valid_collider) if is_valid]
    all_computed_times = [_ for _, is_valid in zip(all_computed_times, valid_collider) if is_valid]
    
    all_origin_times = np.array(all_origin_times, dtype=np.float64)
    all_computed_times = np.array(all_computed_times, dtype=np.float64)
    all_origin_poses = np.stack(all_origin_poses, axis=0)
    all_computed_poses = np.stack(all_computed_poses, axis=0)

    ############
    # Create GUI
    ############

    # Use replayer.py for the loading of info
    # Use wanted_inds GUI for picking point in trajectory with slider
    # Use slider for translation / rotations

    # Init ros publishers
    collision_pub = rospy.Publisher('/plan_costmap_3D', VoxGrid, queue_size=1)
    pointcloud_pub = rospy.Publisher('/colli_points', PointCloud2, queue_size=10)
    #pub = rospy.Publisher('/p3dx/move_base/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=1)
    rospy.init_node("test_obstacle_msg")


    # Figure
    global f_i
    f_i = 0
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(left=0.1, bottom=0.15)

    # Plot first frame of seq
    vmin = np.min(all_origin_times)
    vmax = np.max(all_origin_times)
    scales = np.ones_like(all_origin_times)
    scales[f_i] = 10.0
    plotsB = [axB.scatter(all_origin_poses[:, 0],
                          all_origin_poses[:, 1],
                          s=scales,
                          c=all_origin_times,
                          cmap='jet',
                          vmin=vmin,
                          vmax=vmax)]

    axB.set_aspect('equal', adjustable='box')
    
    # Make a horizontal slider to control the frequency.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.1, 0.04, 0.8, 0.02], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                         label='ind',
                         valmin=0,
                         valmax=len(all_origin_times) - 1,
                         valinit=0,
                         valstep=1)

    # The function to be called anytime a slider's value changes
    def update_PR(val):
        global f_i
        f_i = (int)(val)
        scales = np.ones_like(all_origin_times)
        scales[f_i] = 10.0
        plotsB[0].set_sizes(scales)

    # register the update function with each slider
    time_slider.on_changed(update_PR)

    plt.show()

    a = 1/0


    # Plot first frame of seq
    plotsA = [axA.scatter(all_pts[s_ind][0][:, 0],
                            all_pts[s_ind][0][:, 1],
                            s=2.0,
                            c=all_colors[s_ind][0])]

    # Show a circle of the loop closure area
    axA.add_patch(patches.Circle((0, 0), radius=0.2,
                                    edgecolor=[0.2, 0.2, 0.2],
                                    facecolor=[1.0, 0.79, 0],
                                    fill=True,
                                    lw=1))

    plt.subplots_adjust(left=0.1, bottom=0.15)

    # # Customize the graph
    # axA.grid(linestyle='-.', which='both')
    axA.set_xlim(-im_lim, im_lim)
    axA.set_ylim(-im_lim, im_lim)
    axA.set_aspect('equal', adjustable='box')
    
    # Make a horizontal slider to control the frequency.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.1, 0.04, 0.8, 0.02], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                            label='ind',
                            valmin=0,
                            valmax=len(all_pts[s_ind]) - 1,
                            valinit=0,
                            valstep=1)

    # The function to be called anytime a slider's value changes
    def update_PR(val):
        global f_i
        f_i = (int)(val)
        for plot_i, plot_obj in enumerate(plotsA):
            plot_obj.set_offsets(all_pts[s_ind][f_i])
            plot_obj.set_color(all_colors[s_ind][f_i])

    # register the update function with each slider
    time_slider.on_changed(update_PR)













    a = 1/0

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