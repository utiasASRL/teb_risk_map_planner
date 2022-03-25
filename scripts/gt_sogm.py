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
import torch
from scipy import ndimage
import scipy.ndimage.filters as filters


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from utils.ply import read_ply, write_ply

from utils.pc2_numpy import array_to_pointcloud2_fast, pointcloud2_to_array

#
#
##########################################################################################################################################################
#
#

#################################################################################################
#
# Utils
# *****
#


def project_points_to_2D(pts, grid_L, grid_dl):
    

    # Center grid on the curent cloud
    grid_origin = np.array([[-grid_L/2, -grid_L/2]], dtype=np.float32)

    # Number of cells in the grid
    grid_N = int(np.ceil(grid_L / grid_dl))

    # Transform pts coordinates to pixel coordiantes
    pix = np.floor((pts[..., :2] - grid_origin) / grid_dl).astype(np.int32)
    if pix.ndim == 2:
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


def get_pointcloud_msg(new_points, stamp, frame_id, intensity=None):

    # data structure of binary blob output for PointCloud2 data type
    output_dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity'],
                             'formats': ['<f4', '<f4', '<f4', '<f4'],
                             'offsets': [0, 4, 8, 12],
                             'itemsize': 16})

    if intensity is None:
        intensity = np.zeros_like(new_points[:, 0])

    # fill structured numpy array with points and classes (in the intensity field). Fill ring with zeros to maintain Pointcloud2 structure
    c_points = np.c_[new_points, intensity]
    c_points = np.core.records.fromarrays(c_points.transpose(), output_dtype)

    # convert to Pointcloud2 message and publish
    msg = array_to_pointcloud2_fast(c_points, stamp=stamp, frame_id=frame_id)

    return msg

#################################################################################################
#
# Callback
# ********
#


class Callbacks:

    #
    # Init and Main
    # *************
    #

    def __init__(self, tfBuffer0, tfListener0, actor_times, actor_xy):
        
        ####################
        # Init environment #
        ####################

        # Set which gpu is going to be used (auto for automatic choice)
        on_gpu = True
        GPU_ID = 'auto'

        # Automatic choice (need pynvml to be installed)
        if GPU_ID == 'auto':
            print('\nSearching a free GPU:')
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.list_gpu_processes(i)
                print(torch.cuda.list_gpu_processes(i))
                a = a.split()
                if a[1] == 'no':
                    GPU_ID = a[0][-1:]

        # Safe check no free GPU
        if GPU_ID == 'auto':
            print('\nNo free GPU found!\n')
            a = 1/0

        else:
            print('\nUsing GPU:', GPU_ID, '\n')

        # Get the GPU for PyTorch
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:{:d}".format(int(GPU_ID)))
        else:
            self.device = torch.device("cpu")


        ###################
        # Init parameters #
        ###################

        self.testtest = False
        
        # Spatial dimensions
        self.in_radius = 8
        self.dl_2D = 0.12
        
        # Prediction until T=4.0s
        self.dt = 0.1
        self.n_2D_layers = 40
        self.T = self.n_2D_layers * self.dt
        self.time_resolution = self.dt

        # SOGM params
        self.static_range = 0.8
        self.dynamic_range = 1.5
        self.norm_p = 3
        self.norm_invp = 1 / self.norm_p
        self.maxima_layers = [38]
        self.visu_T = 29

        # Delay of prediction
        self.delay = 0.3

        # Time factor for visu
        self.time_factor = 0.3

        # Actor GT
        self.actor_times = actor_times
        self.actor_xy = actor_xy
        
        self.actor_r = 0.19

        # # N, n, 2
        # actor_pts = np.copy(self.actor_xy)
        # actor_ts = np.zeros_like(actor_pts[:, :, 0])
        # actor_ts += np.expand_dims(actor_times, 1)
        # print(actor_pts.shape)
        # actor_ids = np.zeros_like(actor_pts[:, :, 0], dtype=np.int32)
        # actor_ids += np.expand_dims(np.arange(actor_ids.shape[1]), 0)
        # print(actor_pts.shape)
        # actor_pts = np.reshape(actor_pts, [-1, 2])
        # actor_ids = np.reshape(actor_ids, [-1, ])
        # actor_ts = np.reshape(actor_ts, [-1, ])
        # write_ply('test_actors.ply',
        #           [actor_pts, actor_ts, actor_ids],
        #           ['x', 'y', 'z', 'gtlabels'])


        # Prepare actor shape as a list of 2D points
        shape_x = np.arange(0.0, self.actor_r + self.dl_2D, self.dl_2D)
        shape_x = np.hstack((shape_x, shape_x[1:] * -1.0))
        shape_y = np.copy(shape_x)
        shape_X, shape_Y = np.meshgrid(shape_x, shape_y)
        self.actor_shape = np.vstack((shape_X.ravel(), shape_Y.ravel())).T
        mask = np.linalg.norm(self.actor_shape, axis=1) < self.actor_r
        self.actor_shape = self.actor_shape[mask]

        # Convolution for Collision risk diffusion
        self.static_conv = self.diffusing_convolution(self.static_range)
        self.static_conv.to(self.device)
        self.dynamic_conv = self.diffusing_convolution(self.dynamic_range)
        self.dynamic_conv.to(self.device)

        ############
        # Init ROS #
        ############

        self.map_frame_id = 'map'
        self.tfBuffer = tfBuffer0
        self.tfListener = tfListener0

        self.visu_pub = rospy.Publisher('/dynamic_visu', OccupancyGrid, queue_size=10)
        self.visu_pub_static = rospy.Publisher('/static_visu', OccupancyGrid, queue_size=10)

        self.collision_pub = rospy.Publisher('/plan_costmap_3D', VoxGrid, queue_size=10)
        self.obstacle_pub = rospy.Publisher('/move_base/TebLocalPlannerROS/obstacles', ObstacleArrayMsg, queue_size=10)

        self.pointcloud_pub = rospy.Publisher('/classified_points', PointCloud2, queue_size=10)

        self.visu_pt_cloud_pub = rospy.Publisher('/colli_points', PointCloud2, queue_size=10)

        return

    def velo_callback(self, ptcloud2_msg):

        #############
        # Read points
        #############

        # convert PointCloud2 message to structured numpy array
        cloud_array = pointcloud2_to_array(ptcloud2_msg)

        # remove crap points
        mask = np.logical_and(np.isfinite(cloud_array['x']), np.isfinite(cloud_array['y']))
        mask = np.logical_and(mask, np.isfinite(cloud_array['z']))
        cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        xyz_points = np.zeros(cloud_array.shape + (3,), dtype=np.float32)
        xyz_points[..., 0] = cloud_array['x']
        xyz_points[..., 1] = cloud_array['y']
        xyz_points[..., 2] = cloud_array['z']

        labels = cloud_array['intensity'].astype(np.int32)
        f_times = cloud_array['time'].astype(np.float32)
        f_rings = cloud_array['ring'].astype(np.int32)


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
        while not rospy.is_shutdown() and tries0 < 10:
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
        t0 = ptcloud2_msg.header.stamp
        t0_sec = t0.to_sec()
        fut_times = np.arange(t0_sec, t0_sec + self.T + 0.1 * self.dt, self.dt)

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
        dyn_xy0 = np.expand_dims(interp_xy, 2) + np.expand_dims(self.actor_shape, (0, 1))
        dyn_xy = np.copy(np.reshape(dyn_xy0, (self.n_2D_layers + 1, -1, 2)))


        # print('-----------------')
        # print(dyn_xy.shape)
        # for pt, nt in zip(prev_t, next_t):
        #     print(pt, nt)
        # print('-----------------')


        # Create dynamic predictions
        dyn_grid = project_points_to_2D(dyn_xy, grid_L, self.dl_2D)

        # # To publish as SOGM and not risk_map
        # dyn_visu = (np.max(dyn_grid, axis=0).astype(np.float32) * 97 / 255).astype(np.int8)
        # mask = static_grid > 0
        # dyn_visu[mask] = -2
        # self.publish_static_visu(dyn_visu, ptcloud2_msg.header.stamp, p0)

        # Reshape gt sogm and make it torch tensor -> [T, W, H, 3]
        dyn_gts = dyn_grid.astype(np.float32) / 255
        sta_gts = np.zeros_like(dyn_gts)
        sta_gts[:, static_grid > 0] = 0.99
        collision_gts = np.stack((sta_gts, sta_gts*0, dyn_gts), axis=-1)
        collision_gts = torch.from_numpy(collision_gts).to(self.device)


        
        ##################
        # Wait for delay #
        ##################

        # Convert stamp to float
        sec1 = t0.secs
        nsec1 = t0.nsecs
        stamp_sec = float(sec1) + float(int((nsec1) * 1e-6)) * 1e-3

        # Wait until current ros time reached desired value
        now_stamp = rospy.get_rostime()
        now_sec = float(now_stamp.secs) + float(int((now_stamp.nsecs) * 1e-6)) * 1e-3
        while (now_sec < stamp_sec + self.delay):
            now_stamp = rospy.get_rostime()
            now_sec = float(now_stamp.secs) + float(int((now_stamp.nsecs) * 1e-6)) * 1e-3



        #########################
        # Publish Visu pt cloud #
        #########################

        visu_ptcloud = False
        if visu_ptcloud:

            # Recenter dynamic points  (T, n_actors, n_shape, 2)
            dyn_xy0 += p0[:2]

            # Time as third dimension
            dyn_t = np.zeros_like(dyn_xy0[:, :, :, :1]) + np.expand_dims(fut_times - t0_sec, (1, 2, 3))
            current_delay = now_sec - stamp_sec
            dyn_t = (dyn_t - current_delay) * self.time_factor
            dyn_xyt = np.concatenate((dyn_xy0, dyn_t), axis=-1)

            # One color per layer
            dyn_id = np.zeros_like(dyn_xy0[:, :, :, :1]) + np.expand_dims(np.arange(dyn_xy0.shape[0]) + 1, (1, 2, 3))

            # Reshape
            dyn_xyt = np.reshape(dyn_xyt, (-1, 3))
            dyn_id = np.reshape(dyn_id, (-1,))

            # remove points oustside area
            mask = np.linalg.norm((dyn_xyt[:, :2] - p0[:2]), axis=1) < self.in_radius

            # Get pt as msg
            pt_msg = get_pointcloud_msg(dyn_xyt[mask], now_stamp, self.map_frame_id, intensity=dyn_id[mask])

            # Publish
            self.visu_pt_cloud_pub.publish(pt_msg)

        ################
        # Publish Risk #
        ################

        # Create riskmap from gt sogm
        diffused_risk, obst_pos, static_mask = self.get_diffused_risk(collision_gts)


        ###########################
        # DEBUG RISK
        # collision_gts [T, W, H, 3] float32

        if not self.testtest:
            self.testtest = True
            self.collision_gts_visu = collision_gts
            self.diffused_risk_visu = diffused_risk

        ###########################

        # Publish collision risk in a custom message
        self.publish_collisions(diffused_risk, stamp_sec, p0)
        self.publish_collisions_visu(diffused_risk, static_mask, t0, p0, visu_T=self.visu_T)

        #####################
        # Publish obstacles #
        #####################

        # Get obstacles in world coordinates
        origin0 = p0 - self.in_radius / np.sqrt(2)

        world_obst = []
        for obst_i, pos in enumerate(obst_pos):
            world_obst.append(origin0[:2] + pos * self.dl_2D)

        # Publish obstacles
        self.publish_obstacles(world_obst)


        #####################
        # Publish 3D points #
        #####################

        # Transform gt_labels to our labels
        # -----------------
        # 0: 'uncertain',
        # 1: 'ground',
        # 2: 'still',
        # 3: 'longT',
        # 4: 'shortT'
        # -----------------
        # 0: 'ground',  => 1
        # 1: 'chair',   => 3
        # 2: 'movingp', => 4
        # 3: 'sitting', => 3
        # 4: 'table',   => 3
        # 5: 'wall',    => 2
        # 6: 'door',    => 3
        # -----------------

        gt_to_labels = np.array([1, 3, 4, 3, 3, 2, 3], dtype=np.int32)
        predictions = gt_to_labels[labels]
        
        # Get frame points re-aligned in the map
        pred_points = (aligned_pts + p0).astype(np.float32)

        # Publish pointcloud
        self.publish_pointcloud(pred_points, predictions, f_times, f_rings, t0)
        
        now_stamp = rospy.get_rostime()
        now_sec = float(now_stamp.secs) + float(int((now_stamp.nsecs) * 1e-6)) * 1e-3

        print('   >>> Publishing {:.3f} with a delay of {:.3f}s'.format(stamp_sec, now_sec - stamp_sec))



        return

    #
    # Utils
    # *****
    #

    def get_diffused_risk(self, collision_preds):
                                    
        # # Remove residual preds (hard hysteresis)
        # collision_risk *= (collision_risk > 0.06).type(collision_risk.dtype)
                    
        # Remove residual preds (soft hysteresis)
        # lim1 = 0.06
        # lim2 = 0.09
        lim1 = 0.15
        lim2 = 0.2
        dlim = lim2 - lim1
        mask0 = collision_preds <= lim1
        mask1 = torch.logical_and(collision_preds < lim2, collision_preds > lim1)
        collision_preds[mask0] *= 0
        collision_preds[mask1] *= (1 - ((collision_preds[mask1] - lim2) / dlim) ** 2) ** 2

        # Static risk
        # ***********

        # Get risk from static objects, [1, 1, W, H]
        static_preds = torch.unsqueeze(torch.max(collision_preds[:1, :, :, :2], dim=-1)[0], 1)

        # Normalize risk values between 0 and 1 depending on density
        static_risk = static_preds / (self.static_conv(static_preds) + 1e-6)

        # Diffuse the risk from normalized static objects
        diffused_0 = self.static_conv(static_risk).cpu().detach().numpy()

        # Do not repeat we only keep it for the first layer: [1, 1, W, H] -> [W, H]
        diffused_0 = np.squeeze(diffused_0)
        
        # Inverse power for p-norm
        diffused_0 = np.power(np.maximum(0, diffused_0), self.norm_invp)

        # Dynamic risk
        # ************

        # Get dynamic risk [T, W, H]
        dynamic_risk = collision_preds[..., 2]

        # Get high risk area
        high_risk_threshold = 0.7
        high_risk_mask = dynamic_risk > high_risk_threshold
        high_risk = torch.zeros_like(dynamic_risk)
        high_risk[high_risk_mask] = dynamic_risk[high_risk_mask]

        # On the whole dynamic_risk, convolution
        # Higher value for larger area of risk even if low risk
        dynamic_risk = torch.unsqueeze(dynamic_risk, 1)
        diffused_1 = np.squeeze(self.dynamic_conv(dynamic_risk).cpu().detach().numpy())

        # Inverse power for p-norm
        diffused_1 = np.power(np.maximum(0, diffused_1), self.norm_invp)

        # Rescale this low_risk at smaller value
        low_risk_value = 0.4
        diffused_1 = low_risk_value * diffused_1 / (np.max(diffused_1) + 1e-6)

        # On the high risk, we normalize to have similar value of highest risk (around 1.0)
        high_risk = torch.unsqueeze(high_risk, 1)
        high_risk_normalized = high_risk / (self.dynamic_conv(high_risk) + 1e-6)
        diffused_2 = np.squeeze(self.dynamic_conv(high_risk_normalized).cpu().detach().numpy())

        # Inverse power for p-norm
        diffused_2 = np.power(np.maximum(0, diffused_2), self.norm_invp)

        # Rescale and combine risk
        # ************************
        
        print(np.max(diffused_0), np.max(diffused_1), np.max(diffused_2))

        # Combine dynamic risks
        diffused_1 = np.maximum(diffused_1, diffused_2)

        # Rescale risk values (max should be stable around 1.0 for both)
        diffused_0 *= 1.0 / (np.max(diffused_0) + 1e-6)
        diffused_1 *= 1.0 / (np.max(diffused_1) + 1e-6)

        # merge the static risk as the first layer of the vox grid (with the delay this layer is useless for dynamic)
        diffused_1[0, :, :] = diffused_0

        # Convert to uint8 for message 0-254 = prob, 255 = fixed obstacle
        diffused_risk = np.minimum(diffused_1 * 255, 255).astype(np.uint8)
        
        # # Save walls for debug
        # debug_walls = np.minimum(diffused_risk[10] * 255, 255).astype(np.uint8)
        # cm = plt.get_cmap('viridis')
        # print(batch.t0)
        # print(type(batch.t0))
        # im_name = join(ENV_HOME, 'catkin_ws/src/collision_trainer/results/debug_walls_{:.3f}.png'.format(batch.t0))
        # imageio.imwrite(im_name, zoom_collisions(cm(debug_walls), 5))

        # Get local maxima in moving obstacles
        obst_mask = None
        for layer_i in self.maxima_layers:
            if obst_mask is None:
                obst_mask = self.get_local_maxima(diffused_1[layer_i])
            else:
                obst_mask = np.logical_or(obst_mask, self.get_local_maxima(diffused_1[layer_i]))

        # Use max pool to get obstacles in one cell over two [H, W] => [H//2, W//2]
        stride = 2
        pool = torch.nn.MaxPool2d(stride, stride=stride, return_indices=True)
        unpool = torch.nn.MaxUnpool2d(stride, stride=stride)
        output, indices = pool(static_preds.detach())
        static_preds_2 = unpool(output, indices, output_size=static_preds.shape)

        # Merge obstacles
        obst_mask[np.squeeze(static_preds_2.cpu().numpy()) > 0.3] = 1

        # Convert to pixel positions
        obst_pos = self.mask_to_pix(obst_mask)

        # Get mask of static obstacles for visu
        static_mask = np.squeeze(static_preds.detach().cpu().numpy()) > 0.3
        
        return diffused_risk, obst_pos, static_mask
        
    def get_local_maxima(self, data, neighborhood_size=5, threshold=0.1):
        
        # Get maxima positions as a mask
        data_max = filters.maximum_filter(data, neighborhood_size)
        max_mask = (data == data_max)

        # Remove maxima if their peak is not higher than threshold in the neighborhood
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        max_mask[diff == 0] = 0

        return max_mask

    def mask_to_pix(self, mask):
        
        # Get positions in world coordinates
        labeled, num_objects = ndimage.label(mask)
        slices = ndimage.find_objects(labeled)
        x, y = [], []

        mask_pos = []
        for dy, dx in slices:

            x_center = (dx.start + dx.stop - 1) / 2
            y_center = (dy.start + dy.stop - 1) / 2
            mask_pos.append(np.array([x_center, y_center], dtype=np.float32))

        return mask_pos

    def diffusing_convolution(self, obstacle_range):
        
        k_range = int(np.ceil(obstacle_range / self.dl_2D))
        k = 2 * k_range + 1
        dist_kernel = np.zeros((k, k))
        for i, vv in enumerate(dist_kernel):
            for j, v in enumerate(vv):
                dist_kernel[i, j] = np.sqrt((i - k_range) ** 2 + (j - k_range) ** 2)
        dist_kernel = np.clip(1.0 - dist_kernel * self.dl_2D / obstacle_range, 0, 1) ** self.norm_p
        fixed_conv = torch.nn.Conv2d(1, 1, k, stride=1, padding=k_range, bias=False)
        fixed_conv.weight.requires_grad = False
        fixed_conv.weight *= 0
        fixed_conv.weight += torch.from_numpy(dist_kernel)

        return fixed_conv

    #
    # Publishing
    # **********
    #

    def publish_collisions(self, collision_preds, stamp0, p0):

        # Get origin and orientation
        origin0 = p0 - self.in_radius / np.sqrt(2)

        # Define header
        msg = VoxGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'

        # Define message
        msg.depth = collision_preds.shape[0]
        msg.width = collision_preds.shape[1]
        msg.height = collision_preds.shape[2]
        msg.dl = self.dl_2D
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

    def publish_collisions_visu(self, collision_preds, static_mask, t0, p0, visu_T=15):
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
        origin0 = p0 - self.in_radius / np.sqrt(2)

        # Define header
        msg = OccupancyGrid()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'
        msg_static = OccupancyGrid()
        msg_static.header.stamp = rospy.get_rostime()
        msg_static.header.frame_id = 'map'

        # Define message meta data
        msg.info.map_load_time = t0
        msg.info.resolution = self.dl_2D
        msg.info.width = collision_preds.shape[1]
        msg.info.height = collision_preds.shape[2]
        msg.info.origin.position.x = origin0[0]
        msg.info.origin.position.y = origin0[1]
        msg.info.origin.position.z = -0.011
        #msg.info.origin.orientation.x = q0[0]
        #msg.info.origin.orientation.y = q0[1]
        #msg.info.origin.orientation.z = q0[2]
        #msg.info.origin.orientation.w = q0[3]

        msg_static.info.map_load_time = t0
        msg_static.info.resolution = self.dl_2D
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
            # for iso_i, iso in enumerate([230, 150, 70]):
            for iso_i, iso in enumerate([230]):

                dynamic_mask = collision_preds[1:, :, :] > iso
                dynamic_data = dynamic_mask.astype(np.float32) * np.expand_dims(np.arange(dynamic_mask.shape[0]) + 1, (1, 2))
                max_v = np.max(dynamic_data)
                dynamic_data[np.logical_not(dynamic_mask)] = max_v + 2
                dynamic_data = np.min(dynamic_data, axis=0)
                dynamic_data[dynamic_data > max_v + 1] = 0

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

    def publish_obstacles(self, obstacle_list):


        msg = ObstacleArrayMsg()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = 'map'
        
        # Add point obstacles
        for obst_i, pos in enumerate(obstacle_list):

            obstacle_msg = ObstacleMsg()
            obstacle_msg.id = obst_i
            obstacle_msg.polygon.points = [Point32(x=pos[0], y=pos[1], z=0.0)]

            msg.obstacles.append(obstacle_msg)

        self.obstacle_pub.publish(msg)

        return

    def publish_pointcloud(self, new_points, predictions, f_times, f_rings, t0):

        t = [time.time()]

        # data structure of binary blob output for PointCloud2 data type
        output_dtype = np.dtype({'names': ['x', 'y', 'z', 'classif', 'time', 'ring'],
                                 'formats': ['<f4', '<f4', '<f4', '<i4', '<f4', '<u2']})

        # new_points = np.hstack((new_points, predictions))
        structured_pc2_array = np.empty([new_points.shape[0]], dtype=output_dtype)

        t += [time.time()]

        structured_pc2_array['x'] = new_points[:, 0]
        structured_pc2_array['y'] = new_points[:, 1]
        structured_pc2_array['z'] = new_points[:, 2]
        structured_pc2_array['classif'] = predictions.astype(np.int32)
        structured_pc2_array['time'] = f_times.astype(np.float32)
        structured_pc2_array['ring'] = f_rings.astype(np.uint16)

        # structured_pc2_array['frame_id'] = (features[:, -1] > 0.01).astype(np.uint8)
        
        t += [time.time()]

        # convert to Pointcloud2 message and publish
        msg = array_to_pointcloud2_fast(structured_pc2_array,
                                        t0,
                                        self.map_frame_id,
                                        True)


        t += [time.time()]

        self.pointcloud_pub.publish(msg)
        
        t += [time.time()]

        # print(35 * ' ',
        #       35 * ' ',
        #       '{:^35s}'.format('Publish pointcloud {:.0f} + {:.0f} + {:.0f} + {:.0f} ms'.format(1000 * (t[1] - t[0]),
        #                                                                                         1000 * (t[2] - t[1]),
        #                                                                                         1000 * (t[3] - t[2]),
        #                                                                                         1000 * (t[4] - t[3]))))

        return

#################################################################################################
#
# Main call
# *********
#


if __name__ == '__main__':
    
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
    t1 = time.time()
    poses_path = os.path.join(load_path, load_world, "vehicles.txt")
    actor_poses = np.loadtxt(poses_path)
    t2 = time.time()

    print('Loaded actor positions in {:.3f}s'.format(t2 - t1))

    # Extract xy positions and times
    actor_times = actor_poses[:, 0]
    actor_x = actor_poses[:, 1::7]
    actor_y = actor_poses[:, 2::7]
    actor_xy = np.stack((actor_x, actor_y), axis=2)

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


    debug = False
    if debug:
        while not rospy.is_shutdown():


            # Wait for matplotlib to be called
            print('matplotlib called: ', my_callbacks.testtest)
            time.sleep(0.5)

            if (my_callbacks.testtest):

                collision_gts = my_callbacks.collision_gts_visu
                diffused_risk = my_callbacks.diffused_risk_visu
    
                # Figure
                global dl, xoff, yoff, fake_sogm, valuee
                xoff = 0
                yoff = 0
                dl = 2
                valuee = 0.99
                fake_sogm = np.copy(collision_gts.cpu().numpy())
                fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 7))
                plt.subplots_adjust(left=0.1, bottom=0.15)

                images = []
                images.append(axA.imshow(fake_sogm[2]))
                images.append(axB.imshow(diffused_risk[2]))
                                
                # The function to be called anytime a slider's value changes
                def update_image_1():
                    global dl, xoff, yoff, fake_sogm

                    # Add obstacle in the image at the wanted postion
                    fake_sogm2 = np.copy(fake_sogm[2])

                    yoff1 = (fake_sogm2.shape[0] - 1) - yoff
                    u0 = max(yoff1 - dl, 0)
                    u1 = min(yoff1 + dl, fake_sogm2.shape[0])
                    v0 = max(xoff - dl, 0)
                    v1 = min(xoff + dl, fake_sogm2.shape[1])
                    fake_sogm2[u0:u1, v0:v1, -1] = valuee
                    
                    fake_sogm2[45:50, 40:60, -1] = np.maximum(fake_sogm2[45:50, 40:60, -1], 0.39)
                    
                    # Update images
                    images[0].set_array(fake_sogm2)
                    
                    plt.draw()

                    return images
                            
                # The function to be called anytime a slider's value changes
                def update_image_2():
                    
                    # Add obstacle in the image at the wanted postion
                    fake_sogm_visu = np.copy(fake_sogm)

                    yoff1 = (fake_sogm_visu.shape[1] - 1) - yoff
                    u0 = max(yoff1 - dl, 0)
                    u1 = min(yoff1 + dl, fake_sogm_visu.shape[1])
                    v0 = max(xoff - dl, 0)
                    v1 = min(xoff + dl, fake_sogm_visu.shape[2])
                    fake_sogm_visu[2, u0:u1, v0:v1, -1] = valuee
                    
                    fake_sogm_visu[2, 45:50, 40:60, -1] = np.maximum(fake_sogm_visu[2, 45:50, 40:60, -1], 0.39)
                    
                    diffused_fake, obst_pos, static_mask = my_callbacks.get_diffused_risk(torch.from_numpy(fake_sogm_visu).to(my_callbacks.device))

                    images[1].set_array(diffused_fake[2])

                    plt.draw()

                    return images

                #######################################################################################
                # Make a horizontal slider to control x.
                axcolor = 'lightgoldenrodyellow'
                axtime = plt.axes([0.05, 0.06, 0.4, 0.02], facecolor=axcolor)
                x_slider = Slider(ax=axtime,
                                  label='frame',
                                  valmin=0,
                                  valmax=collision_gts.shape[2],
                                  valinit=0,
                                  valstep=1)

                # The function to be called anytime a slider's value changes
                def update_xoff(val):
                    global xoff
                    xoff = int(val)
                    return update_image_1()
                #######################################################################################

                #######################################################################################
                # Make a vertical slider to control y.
                axcolor = 'lightgoldenrodyellow'
                axtime = plt.axes([0.02, 0.1, 0.01, 0.8], facecolor=axcolor)
                y_slider = Slider(ax=axtime,
                                  label='y_offset',
                                  valmin=0,
                                  valmax=collision_gts.shape[1],
                                  valinit=0,
                                  valstep=1,
                                  orientation="vertical")

                # The function to be called anytime a slider's value changes
                def update_yoff(val):
                    global yoff
                    yoff = int(val)
                    return update_image_1()
                #######################################################################################

                x_slider.on_changed(update_xoff)
                y_slider.on_changed(update_yoff)

                #######################################################################################
                # Key press events

                def onkey(event):
                    global dl, xoff, yoff, valuee
                    
                    if event.key == 'm':
                        dl += 1
                        return update_image_1()

                    elif event.key == 'n':
                        dl -= 1
                        return update_image_1()

                    if event.key == 'g':
                        valuee += 0.1
                        return update_image_2()

                    elif event.key == 'h':
                        valuee -= 0.1
                        return update_image_2()

                    if event.key == 'right':
                        xoff += 1
                        return update_image_1()

                    elif event.key == 'left':
                        xoff -= 1
                        return update_image_1()
                    if event.key == 'up':
                        yoff += 1
                        return update_image_1()

                    elif event.key == 'down':
                        yoff -= 1
                        return update_image_1()

                    elif event.key == 'enter':
                        return update_image_2()

                    return None

                        
                cid = fig.canvas.mpl_connect('key_press_event', onkey)
                print('\n---------------------------------------\n')
                print('Instructions:\n')
                print('> Use right and left arrows to make area smaller/bigger.')
                print('> Use enter to compute riskmap.')
                print('\n---------------------------------------\n')


                #######################################################################################

                plt.show()

                a = 1/0

    else:

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
