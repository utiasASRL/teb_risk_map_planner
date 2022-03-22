#!/usr/bin/env python3

import rospy
import math
import copy
import time
import tf
import sys
import pickle
import os
import rosbag
import tf2_ros
from tf2_msgs.msg import TFMessage
from ros_numpy import point_cloud2 as pc2
import numpy as np
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32, QuaternionStamped, Quaternion, TwistWithCovariance, PoseStamped, TransformStamped
from tf.transformations import quaternion_from_euler
from vox_msgs.msg import VoxGrid
from nav_msgs.msg import OccupancyGrid, MapMetaData
from sensor_msgs.msg import PointCloud2
from teb_local_planner.msg import FeedbackMsg, TrajectoryMsg, TrajectoryPointMsg
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as scipyR


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from utils.ply import read_ply, write_ply


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


def get_collisions_msg(collision_preds, origin0, dl_2D, time_resolution):

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
    msg.origin.z = origin0[2]

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


def get_pred_points(collision_preds, origin0, dl_2D, dt):

    # Get mask of the points we want to show

    mask = collision_preds > 0.1 * 255

    nt, nx, ny = collision_preds.shape
    
    t = np.arange(0, nt, 1) + 1
    x = np.arange(0, nx, 1)
    y = np.arange(0, ny, 1)
    tv, yv, xv = np.meshgrid(t, x, y, indexing='ij')
    
    tv = tv[mask].astype(np.float32)
    xv = xv[mask].astype(np.float32)
    yv = yv[mask].astype(np.float32)

    time_factor = 0.3

    xv = origin0[0] + (xv + 0.5) * dl_2D
    yv = origin0[1] + (yv + 0.5) * dl_2D
    tv = (origin0[2] + tv * dt) * time_factor

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
    

def get_obstacle_msg(obstacles, ids, offset):


    msg = ObstacleArrayMsg()
    msg.header.stamp = rospy.get_rostime()
    msg.header.frame_id = 'odom'
    
    # Add point obstacles
    for obst_i, pos in zip(ids, obstacles):

        obstacle_msg = ObstacleMsg()
        obstacle_msg.id = obst_i
        obstacle_msg.polygon.points = [Point32(x=pos[0] + offset[0], y=pos[1] + offset[1], z=0.0)]
        msg.obstacles.append(obstacle_msg)

    return msg


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

    dims = (-1,) + tuple(collider_data['dims'][0])
    collider_data['preds'] = np.reshape(collider_data['preds'], dims)

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
    

def read_obstacles(topic_name, bagfile):
    '''returns list of local plan message'''

    obstacle_data = {}
    all_obstacle_ids = []
    all_obstacles = []
    all_header_stamp = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        all_header_stamp.append(msg.header.stamp.to_sec())

        obst_pts = []
        obst_ids = []
        for obst in msg.obstacles:
            obst_ids.append(obst.id)
            obst_pts.append([obst.polygon.points[0].x, obst.polygon.points[0].y])

        all_obstacle_ids.append(np.array(obst_ids, dtype=np.int32))
        all_obstacles.append(np.array(obst_pts, dtype=np.float64))

    obstacle_data['header_stamp'] = np.array(all_header_stamp, dtype=np.float64)
    obstacle_data['obstacles'] = all_obstacles
    obstacle_data['ids'] = all_obstacle_ids

    return obstacle_data
    

def read_static_visu(topic_name, bagfile):
    '''returns list of local plan message'''

    all_visu_msg = []

    for topic, msg, t in bagfile.read_messages(topics=[topic_name]):

        all_visu_msg.append(msg)

    return all_visu_msg


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


#
#
#
#
##########################################################################################################################################################
#
#
#
#
#################################################################################################
#
# Utils
# *****
#


def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float32, label_field=""):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
    a 3xN matrix.
    '''

    # remove crap points
    if remove_nans:
        mask = np.logical_and(np.isfinite(cloud_array['x']), np.isfinite(cloud_array['y']))
        mask = np.logical_and(mask, np.isfinite(cloud_array['z']))
        cloud_array = cloud_array[mask]

    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[..., 0] = cloud_array['x']
    points[..., 1] = cloud_array['y']
    points[..., 2] = cloud_array['z']

    if label_field:

        labels = cloud_array[label_field].astype(np.int32)

        return points, labels


    return points


def project_points_to_2D(pts, grid_L, grid_dl):
    

    # Center grid on the curent cloud
    grid_origin = np.array([[-grid_L/2, -grid_L/2]], dtype=np.float32)

    # Number of cells in the grid
    grid_N = int(np.ceil(grid_L / grid_dl))

    # Transform pts coordinates to pixel coordiantes
    pix = np.floor((pts[..., :2] - grid_origin) / grid_dl).astype(np.int32)
    pix = np.unique(pix, axis=0)

    # Get prediction shape
    supp_dims = tuple(pix.shape[:-2])
    pix = np.reshape(pix, (-1,) + tuple(pix.shape[-2:]))
    fused_D = pix.shape[0]
    
    # Get layers indices for 3 dims indexing
    pix1 = pix[:, :, 0]
    pix2 = pix[:, :, 1]
    pix_layers = np.expand_dims(np.arange(fused_D, dtype=np.int32), axis=1) * np.ones_like(pix1)

    # Grid limits
    valid_mask = np.logical_and(pix1 >= 0, pix1 < grid_N)
    valid_mask = np.logical_and(valid_mask, pix2 >= 0)
    valid_mask = np.logical_and(valid_mask, pix2 < grid_N)

    pix0 = pix_layers[valid_mask]
    pix1 = pix1[valid_mask]
    pix2 = pix2[valid_mask]

    # Fill grid
    grid_data = np.zeros((fused_D, grid_N, grid_N), np.uint8)
    grid_data[pix0, pix2, pix1] = 255

    # Reshape with original supp dimensions
    return np.reshape(grid_data, supp_dims + (grid_N, grid_N))


#################################################################################################
#
# Callback
# ********
#

class Callbacks:

    def __init__(self, tfBuffer0, tfListener0, actor_times, actor_xy):

        self.actor_times = actor_times
        self.actor_xy = actor_xy

        self.tfBuffer = tfBuffer0
        self.tfListener = tfListener0

        # Spatial dimensions
        self.in_radius = 8
        self.dl_2D = 0.12
        
        # Prediction until T=4.0s
        self.dt = 0.1
        self.n_2D_layers = 40
        self.T = self.n_2D_layers * self.dt

        # Prepare actor shape as a list of 2D points
        self.actor_r = 0.35
        shape_x = np.arange(0.0, self.actor_r + self.dl_2D, self.dl_2D)
        shape_x = np.hstack((shape_x, shape_x[1:] * -1.0))
        shape_y = np.copy(shape_x)
        shape_X, shape_Y = np.meshgrid(shape_x, shape_y)
        self.actor_shape = np.vstack((shape_X.ravel(), shape_Y.ravel())).T
        mask = np.linalg.norm(self.actor_shape, axis=1) < self.actor_r
        self.actor_shape = self.actor_shape[mask]

        self.visu_pub = rospy.Publisher('/static_visu', OccupancyGrid, queue_size=10)

        return



    def velo_callback(self, ptcloud2_msg):

        #############
        # Read points
        #############

        # convert PointCloud2 message to structured numpy array
        labeled_points = pc2.pointcloud2_to_array(ptcloud2_msg)

        # convert numpy array to Nx3 sized numpy array of float32
        xyz_points, labels = get_xyz_points(labeled_points, remove_nans=True, dtype=np.float32, label_field="intensity")

        # Safe check
        if xyz_points.shape[0] < 100:
            print('Corrupted frame not used')
            return

        # print(ptcloud2_msg.header)
        # print(ptcloud2_msg.height)
        # print(ptcloud2_msg.width)
        # print(ptcloud2_msg.fields)
        # print(ptcloud2_msg.point_step)
        # print(ptcloud2_msg.row_step)
        # print(ptcloud2_msg.is_dense)
        # print('-------------------------------------------')

        #################################
        # Get transform from frame to map
        #################################

        tries0 = 0
        pose = None
        while not rospy.is_shutdown() and tries0 < 3:
            tries0 += 1
            try:
                pose = self.tfBuffer.lookup_transform('map', 'velodyne', ptcloud2_msg.header.stamp)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                time.sleep(0.1)
                continue

        if pose is None:
            print()
            print('Could not get the pose of the frame')
            print()
            return

        # Get pose data
        T_q = np.array([pose.transform.translation.x,
                        pose.transform.translation.y,
                        pose.transform.translation.z,
                        pose.transform.rotation.x,
                        pose.transform.rotation.y,
                        pose.transform.rotation.z,
                        pose.transform.rotation.w], dtype=np.float64)
    
        # Get translation and rotation matrices
        T = T_q[:3]
        R = scipyR.from_quat(T_q[3:]).as_matrix()

        print()
        print('Got pose:')
        print(T)
        print(R)
        print()


        #####################
        # Get the static sogm
        #####################

        # Apply tranformation to align input, but recenter on p0
        p0 = np.copy(T)
        aligned_pts = np.dot(xyz_points, R.T)  # + T

        # write_ply('testesteststets.ply',
        #           [aligned_pts, labels],
        #           ['x', 'y', 'z', 'gtlabels'])

        # a = 1/0

        # Get static obstacles (not ground and not dynamic people)
        min_z = 0.2
        max_z = 1.2
        static_mask = np.logical_and(labels > 0, labels != 2)
        static_mask = np.logical_and(static_mask, aligned_pts[:, 2] > min_z)
        static_mask = np.logical_and(static_mask, aligned_pts[:, 2] < max_z)
        static_pts = aligned_pts[static_mask]
        # dyn_pts = aligned_pts[labels == 2]

        # Project on a 2D grid
        grid_L = 2 * self.in_radius / np.sqrt(2)
        static_grid = project_points_to_2D(static_pts, grid_L, self.dl_2D)


        ######################
        # Get the dynamic sogm
        ######################

        # Get actors future at this timestamp
        t0 = ptcloud2_msg.header.stamp.to_sec()
        fut_times = np.arange(t0, t0 + self.T + 0.1 * self.dt, self.dt)

        if (fut_times[0] < self.actor_times[0]):
            print()
            print('Current frame time too early compared to saved actor times')
            print()
            return

        if (fut_times[-1] > self.actor_times[-1]):
            print()
            print('Current frame time too late compared to saved actor times')
            print()
            return

        # Relevant actor times
        fut_i = np.searchsorted(self.actor_times, fut_times)

        # Actor xy positions interpolated (T, n_actors, 2)
        prev_xy = self.actor_xy[fut_i - 1, :, :]
        next_xy = self.actor_xy[fut_i, :, :]
        prev_t = self.actor_times[fut_i - 1]
        next_t = self.actor_times[fut_i]
        alpha = (fut_times - prev_t) / (next_t - prev_t)
        alpha = np.expand_dims(alpha, (1, 2))
        interp_xy = (1-alpha) * prev_xy + alpha * next_xy

        # Recenter on p0
        interp_xy = interp_xy - p0[:2]

        # Actor have a shape, we just use circle here (T, n_actors, n_shape, 2)
        dyn_xy = np.expand_dims(interp_xy, 2) + np.expand_dims(self.actor_shape, (0, 1))
        dyn_xy = np.reshape(dyn_xy, (self.n_2D_layers + 1, -1, 2))

        # Create dynamic predictions
        
        print(dyn_xy.shape)

        dyn_grid = project_points_to_2D(dyn_xy, grid_L, self.dl_2D)

        print(dyn_grid.shape, dyn_grid.dtype)
        print(static_grid.shape, static_grid.dtype)

        dyn_visu = (np.max(dyn_grid, axis=0).astype(np.float32) * 97 / 255).astype(np.int8)

        mask = static_grid > 0

        print(mask.shape, mask.dtype)
        print(dyn_visu.shape, dyn_visu.dtype)

        dyn_visu[mask] = -2
        
        print(dyn_visu.shape, dyn_visu.dtype)
        print(np.unique(dyn_visu))

        self.publish_static_visu(dyn_visu, ptcloud2_msg.header.stamp, p0)




        # TODO: Add moving points and make that cleaner





                    

        return

    def publish_static_visu(self, static_visu, t0, p0):
        '''
        0 = invisible
        1 -> 98 = blue to red
        99 = cyan
        100 = yellow
        101 -> 127 = green
        128 -> 254 = red to yellow
        255 = vert/gris
        '''

        # Get origin and orientation
        origin0 = p0 - self.in_radius / np.sqrt(2)

        # Define header
        msg = OccupancyGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'

        # Define message meta data
        msg.info.map_load_time = t0
        msg.info.resolution = self.dl_2D
        msg.info.width = static_visu.shape[0]
        msg.info.height = static_visu.shape[1]
        msg.info.origin.position.x = origin0[0]
        msg.info.origin.position.y = origin0[1]
        msg.info.origin.position.z = -0.011

        # Publish
        msg.data = static_visu.ravel().tolist()
        self.visu_pub.publish(msg)

        return


    def publish_collisions(self, collision_preds, stamp0, p0, q0):

        # Get origin and orientation
        origin0 = p0 - self.config.in_radius / np.sqrt(2)

        # Define header
        msg = VoxGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Define message
        msg.depth = collision_preds.shape[0]
        msg.width = collision_preds.shape[1]
        msg.height = collision_preds.shape[2]
        msg.dl = self.config.dl_2D
        msg.dt = self.time_resolution
        msg.origin.x = origin0[0]
        msg.origin.y = origin0[1]
        msg.origin.z = stamp0  # This is already the converted float value (message type is float64)

        #msg.theta = q0[0]
        msg.theta = 0.0

        msg.data = collision_preds.ravel().tolist()

        # Publish
        self.collision_pub.publish(msg)

        return

    def publish_collisions_visu(self, collision_preds, static_mask, t0, p0, q0, visu_T=15):
        '''
        0 = invisible
        1 -> 98 = blue to red
        99 = cyan
        100 = yellow
        101 -> 127 = green

        128 -> 254 = red to yellow
        255 = vert/gris

        -127 -> -2 = red to yellow
        -1 = vert/gris
        '''

        # Get origin and orientation
        origin0 = p0 - self.config.in_radius / np.sqrt(2)

        # Define header
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg_static = OccupancyGrid()
        msg_static.header.stamp = self.get_clock().now().to_msg()
        msg_static.header.frame_id = 'map'



        # Define message meta data
        msg.info.map_load_time = rclTime(seconds=t0.sec, nanoseconds=t0.nanosec).to_msg()
        msg.info.resolution = self.config.dl_2D
        msg.info.width = collision_preds.shape[1]
        msg.info.height = collision_preds.shape[2]
        msg.info.origin.position.x = origin0[0]
        msg.info.origin.position.y = origin0[1]
        msg.info.origin.position.z = -0.011
        #msg.info.origin.orientation.x = q0[0]
        #msg.info.origin.orientation.y = q0[1]
        #msg.info.origin.orientation.z = q0[2]
        #msg.info.origin.orientation.w = q0[3]

        msg_static.info.map_load_time = rclTime(seconds=t0.sec, nanoseconds=t0.nanosec).to_msg()
        msg_static.info.resolution = self.config.dl_2D
        msg_static.info.width = collision_preds.shape[1]
        msg_static.info.height = collision_preds.shape[2]
        msg_static.info.origin.position.x = origin0[0]
        msg_static.info.origin.position.y = origin0[1]
        msg_static.info.origin.position.z = -0.01


        # Define message data
        #   > static risk: yellow to red
        #   > dynamic risk: blue to red
        #   > invisible for the rest of the map
        # Actually separate them in two different costmaps

        dyn_v = "v2"
        dynamic_data0 = np.zeros((1, 1))
        if dyn_v == "v0":
            dynamic_data = collision_preds[visu_T, :, :].astype(np.float32)
            dynamic_data *= 1 / 255
            dynamic_data *= 126
            dynamic_data0 = np.maximum(0, np.minimum(126, dynamic_data.astype(np.int8)))
            mask = dynamic_data0 > 0
            dynamic_data0[mask] += 128

        if dyn_v == "v1":
            dynamic_mask = collision_preds[1:, :, :] > 180
            dynamic_data = dynamic_mask.astype(np.float32) * np.expand_dims(np.arange(dynamic_mask.shape[0]), (1, 2))
            dynamic_data = np.max(dynamic_data, axis=0)
            dynamic_data *= 1 / np.max(dynamic_data)
            dynamic_data *= 126
            dynamic_data0 = np.maximum(0, np.minimum(126, dynamic_data.astype(np.int8)))
            mask = dynamic_data0 > 0
            dynamic_data0[mask] += 128

        elif dyn_v == "v2":
            for iso_i, iso in enumerate([230, 150, 70]):

                dynamic_mask = collision_preds[1:, :, :] > iso
                dynamic_data = dynamic_mask.astype(np.float32) * np.expand_dims(np.arange(dynamic_mask.shape[0]), (1, 2))
                dynamic_data = np.max(dynamic_data, axis=0)
                if iso_i > 0:
                    erode_mask = dynamic_data > 0
                    close_struct = np.ones((5, 5))
                    erode_struct = np.ones((3, 3))
                    erode_mask = ndimage.binary_closing(erode_mask, structure=close_struct, iterations=2)
                    erode_mask = ndimage.binary_erosion(erode_mask, structure=erode_struct)
                    dynamic_data[erode_mask] = 0
                dynamic_data0 = np.maximum(dynamic_data0, dynamic_data)

            dynamic_data0 *= 1 / np.max(dynamic_data0)
            dynamic_data0 *= 126
            dynamic_data0 = np.maximum(0, np.minimum(126, dynamic_data0.astype(np.int8)))
            mask = dynamic_data0 > 0
            dynamic_data0[mask] += 128




        # Static risk
        static_data0 = collision_preds[0, :, :].astype(np.float32)
        static_data = static_data0 * 98 / 255
        static_data = static_data * 1.06 - 3
        static_data = np.maximum(0, np.minimum(98, static_data.astype(np.int8)))
        static_data[static_mask] = 99

        # Publish
        msg.data = dynamic_data0.ravel().tolist()
        msg_static.data = static_data.ravel().tolist()
        self.visu_pub.publish(msg)
        self.visu_pub_static.publish(msg_static)

        return

#################################################################################################
#
# Main call
# *********
#


def main():


    # Define parameters


    # First load all the actor poses


    # Then in a loop
    #   > Get velodyne points
    #       -if gt classified, get static sogm
    #       -if unclassified, WARNING and get static sogm from every point
    #   > Get current frame time
    #   > Create dynamic future from the postion i nthe future of frame time
    #   > Wait to create a custom delay
    #   > Publish the costmap, costmap visu etc
    #


    ###################
    # Define parameters
    ###################

    # Define delay (in simu time)
    sogm_delay = 0.3


    ##################
    # Init actor poses
    ##################
    
    # Get the loaded world
    rospy.init_node("gt_sogm_simu")
    load_path = rospy.get_param('load_path')
    load_world = rospy.get_param('load_world')

    # Load actor poses
    poses_path = os.path.join(load_path, load_world, "vehicles.txt")
    actor_poses = np.loadtxt(poses_path)

    # Extract xy positions and times
    actor_times = actor_poses[:, 0]
    actor_x = actor_poses[:, 1::7]
    actor_y = actor_poses[:, 2::7]
    actor_xy = np.stack((actor_x, actor_y), axis=2)

    print(actor_xy.shape)
    print(actor_times.shape)


    ##############################
    # Subscribe to velodyne points
    ##############################

    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer,
                                           queue_size=10)

    my_callbacks = Callbacks(tfBuffer, tfListener, actor_times, actor_xy)

    rospy.Subscriber("/velodyne_points", PointCloud2, my_callbacks.velo_callback)
    
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer,
                                           queue_size=10)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()







    

    a = 1/0








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
    print("Reading Obstacle messages")
    obst_data = read_obstacles("/move_base/TebLocalPlannerROS/obstacles", bag)
    print("OK")
    print("")

    print("")
    print("Reading Static Visu msgs")
    static_msgs = read_static_visu("/static_visu", bag)
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
    
    obst_data['header_stamp'] = obst_data['header_stamp'][valid_collider]
    obst_data['obstacles'] = [_ for _, is_valid in zip(obst_data['obstacles'], valid_collider) if is_valid]
    obst_data['ids'] = [_ for _, is_valid in zip(obst_data['ids'], valid_collider) if is_valid]

    static_msgs = [_ for _, is_valid in zip(static_msgs, valid_collider) if is_valid]

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

    # Init ros publishers
    collision_pub = rospy.Publisher('/plan_costmap_3D', VoxGrid, queue_size=1)
    pointcloud_pub = rospy.Publisher('/colli_points', PointCloud2, queue_size=10)
    obstacle_pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=5)
    static_pub = rospy.Publisher('/static_visu', OccupancyGrid, queue_size=10)
    rospy.init_node("test_obstacle_msg")


    # Figure
    global f_i, delay, xoff, yoff
    xoff = 0
    yoff = 0
    delay = 0
    f_i = 0
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(left=0.1, bottom=0.15)

    # Plot trajectory of seq
    vmin = np.min(all_origin_times)
    vmax = np.max(all_origin_times)
    scales = np.ones_like(all_origin_times)
    scales[f_i] = 50.0
    plotsB = [axB.scatter(all_origin_poses[:, 1],
                          all_origin_times - all_origin_times[0],
                          s=scales,
                          c=all_origin_times,
                          cmap='hsv',
                          vmin=vmin,
                          vmax=vmax)]

    # axB.set_aspect('equal', adjustable='box')
                     
    # The function to be called anytime a slider's value changes
    def publish_costmap():
        global f_i, delay, xoff, yoff
        new_origin = np.copy(collider_data['origin'][f_i])
        offset = -np.copy(all_origin_poses[f_i, :2])

        offset[0] += xoff
        offset[1] += yoff

        new_origin[0] += offset[0]
        new_origin[1] += offset[1]
        new_origin[2] += delay - collider_data['header_stamp'][f_i]

        # TEMP debug, remove static 
        dynamic_layers = np.copy(collider_data['preds'][f_i])
        dynamic0_layers = np.copy(dynamic_layers)
        dynamic0_layers[0, :, :] = 0


        # Get messages
        collision_msg = get_collisions_msg(dynamic_layers,
                                           new_origin,
                                           collider_data['dl'][f_i],
                                           collider_data['dt'][f_i])

        points, labels = get_pred_points(dynamic0_layers,
                                         new_origin,
                                         collider_data['dl'][f_i],
                                         collider_data['dt'][f_i])

        pt_msg = get_pointcloud_msg(points, labels)

        static_msg = copy.deepcopy(static_msgs[f_i])
        static_msg.header.frame_id = 'odom'
        static_msg.info.origin.position.x = new_origin[0]
        static_msg.info.origin.position.y = new_origin[1]

        # Publish
        collision_pub.publish(collision_msg)
        pointcloud_pub.publish(pt_msg)
        static_pub.publish(static_msg)
        obstacle_pub.publish(get_obstacle_msg(obst_data['obstacles'][f_i],
                                              obst_data['ids'][f_i],
                                              offset))

        return

    #######################################################################################
    # Make a horizontal slider to control the frame.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.55, 0.06, 0.4, 0.02], facecolor=axcolor)
    time_slider = Slider(ax=axtime,
                         label='frame',
                         valmin=0,
                         valmax=len(all_origin_times) - 1,
                         valinit=0,
                         valstep=1)

    # The function to be called anytime a slider's value changes
    def update_frame(val):
        global f_i
        f_i = (int)(val)
        scales = np.ones_like(all_origin_times)
        scales[f_i] = 50.0
        plotsB[0].set_sizes(scales)
        publish_costmap()
    #######################################################################################
        

    #######################################################################################
    # Make a horizontal slider to control the frame.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.55, 0.03, 0.4, 0.02], facecolor=axcolor)
    delay_slider = Slider(ax=axtime,
                         label='delay_offset',
                         valmin=-1.5,
                         valmax=2.5,
                         valinit=0,
                         valstep=0.01)
    # The function to be called anytime a slider's value changes
    def update_delay(val):
        global delay
        delay = val
        publish_costmap()
    #######################################################################################

    #######################################################################################
    # Make a horizontal slider to control the frame.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.05, 0.06, 0.4, 0.02], facecolor=axcolor)
    x_slider = Slider(ax=axtime,
                         label='x_offset',
                         valmin=-4.0,
                         valmax=4.0,
                         valinit=0,
                         valstep=0.01)
    # The function to be called anytime a slider's value changes
    def update_xoff(val):
        global xoff
        xoff = val
        publish_costmap()
    #######################################################################################

    #######################################################################################
    # Make a horizontal slider to control the frame.
    axcolor = 'lightgoldenrodyellow'
    axtime = plt.axes([0.02, 0.1, 0.01, 0.8], facecolor=axcolor)
    y_slider = Slider(ax=axtime,
                      label='y_offset',
                      valmin=-4.0,
                      valmax=4.0,
                      valinit=0,
                      valstep=0.01,
                      orientation="vertical")
    # The function to be called anytime a slider's value changes
    def update_yoff(val):
        global yoff
        yoff = val
        publish_costmap()
    #######################################################################################

    # register the update function with each slider
    time_slider.on_changed(update_frame)
    delay_slider.on_changed(update_delay)
    x_slider.on_changed(update_xoff)
    y_slider.on_changed(update_yoff)

    plt.show()

    a = 1/0


    # # Plot first frame of seq
    # plotsA = [axA.scatter(all_pts[s_ind][0][:, 0],
    #                         all_pts[s_ind][0][:, 1],
    #                         s=2.0,
    #                         c=all_colors[s_ind][0])]

    # # Show a circle of the loop closure area
    # axA.add_patch(patches.Circle((0, 0), radius=0.2,
    #                                 edgecolor=[0.2, 0.2, 0.2],
    #                                 facecolor=[1.0, 0.79, 0],
    #                                 fill=True,
    #                                 lw=1))

    # plt.subplots_adjust(left=0.1, bottom=0.15)

    # # # Customize the graph
    # # axA.grid(linestyle='-.', which='both')
    # axA.set_xlim(-im_lim, im_lim)
    # axA.set_ylim(-im_lim, im_lim)
    # axA.set_aspect('equal', adjustable='box')
    
    # # Make a horizontal slider to control the frequency.
    # axcolor = 'lightgoldenrodyellow'
    # axtime = plt.axes([0.1, 0.04, 0.8, 0.02], facecolor=axcolor)
    # time_slider = Slider(ax=axtime,
    #                         label='ind',
    #                         valmin=0,
    #                         valmax=len(all_pts[s_ind]) - 1,
    #                         valinit=0,
    #                         valstep=1)

    # # The function to be called anytime a slider's value changes
    # def update_PR(val):
    #     global f_i
    #     f_i = (int)(val)
    #     for plot_i, plot_obj in enumerate(plotsA):
    #         plot_obj.set_offsets(all_pts[s_ind][f_i])
    #         plot_obj.set_color(all_colors[s_ind][f_i])

    # # register the update function with each slider
    # time_slider.on_changed(update_PR)

    return


#################################################################################################
#
# Main call
# *********
#


if __name__ == '__main__':
    main()
