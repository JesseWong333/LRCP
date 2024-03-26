
import pickle
import numpy as np
import cv2
import os
from collections import OrderedDict
from opencood.utils import box_utils as box_utils
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose
from data_preprocess_tools.twoD_tools import isRectsOverlap, getTransform, getPointsInQuad, applyTransform
from data_preprocess_tools import custom_vis
from numpy.linalg import inv
from PIL import Image
import math
import uuid
from tqdm import tqdm
import torch

max_time_delay = 5
pc_range = [-32, -32, -3, 32, 32, 2]
# pc_range = [-64, -64, -3, 64, 64, 2] # 
# 160 * 160; -> 80*80
bev_shape = [80, 80]

def save_numpy(path, arr):
    with open(path, 'wb') as f:
        np.save(f, arr)

def vis_point_track(points_tracks, lidars, path):
    # 将points_track 和 lidar转换为， 图像 和 序列
    # trajs is S, N, 2
    # rgbs is S, C, H, W
    
    S = len(lidars)  
    point_hash = {}
    for one_box_track in points_tracks.values():
        if len(one_box_track) == 0:
            continue
        anchor_points = one_box_track[0]
        for i in range(anchor_points.shape[0]):
            point_hash[str(int(anchor_points[i][0]))+ '_' + str(int(anchor_points[i][1])) + '_0' ]  = anchor_points[i][:2]  # 元素为2
        
        for n_frame, point in enumerate(one_box_track):
            for i in range(point.shape[0]):    
                point_hash[str(int(point[i][0]))+ '_' + str(int(point[i][1])) + '_' + str(n_frame+1) ]  = point[i][2:]

    # BEV是[100, 252]， 雷达图可以画得很大，之前是 2000,800, 画为8倍[800, 2016]
    BEV_images = [custom_vis.convert_lidar_to_BEV_image(lidar, pc_range, bev_shape, scale=8).canvas for lidar in lidars]
    
    images = custom_vis.summ_traj2ds_on_rgbs(point_hash, BEV_images, bev_shape=bev_shape, scale=8)
    images[0].save(path, format='gif', save_all=True, append_images=images[1:], duration=500,loop=0)

def vis_boxes(lidars, boxes_dict, path, scale=8):
    # point_list: [lidar1, lidar2, ...]
    # boxes_dict: {box_id1:N*4*2, box_id2:N*4*2}
    BEV_images = [custom_vis.convert_lidar_to_BEV_image(lidar, pc_range, bev_shape, scale=scale).canvas for lidar in lidars]
    for id, boxes in boxes_dict.items():
        for time in range(boxes.shape[0]):
            box = (boxes[time]* scale).astype(np.int32)  # 4*2
            pts = box.reshape((-1, 1, 2))
            cv2.polylines(BEV_images[time], [pts], True, (255, 0, 0), 2)
          
    vis_images = [Image.fromarray(img) for img in BEV_images]
    if len(vis_images) >= 1:
        vis_images[0].save(path, format='gif', save_all=True, append_images=vis_images[1:], duration=500,loop=0)
    else:
        vis_images[0].save(path, format='png')

def filter_points(points):
    # x_boolean = (points[:, 2] <= (bev_shape[1]-1)) & (points[:, 2] <= (bev_shape[1]-1)) & (points[:, 2] >= 0)
    # y_boolean = np.logical_and(points[:, 3] <= bev_shape[0]-1,  points[:, 3] >= 0)
    # xy_boolean = np.logical_and(x_boolean, y_boolean)
    boolean_1 = (points[:, 0] >= 0) & (points[:, 0] <= (bev_shape[1]-1))
    boolean_2 = (points[:, 1] >= 0) & (points[:, 1] <= (bev_shape[0]-1))
    boolean_3 = (points[:, 2] >= 0) & (points[:, 2] <= (bev_shape[1]-1))
    boolean_4 = (points[:, 3] >= 0) & (points[:, 3] <= (bev_shape[0]-1))
    boolean_all = boolean_1 & boolean_2 & boolean_3 & boolean_4
    points =points[boolean_all, :]
    return points

def scale_boxes(box):
    # boxes: 4, 2
    # scale_y = 100 / (40*2)
    # scale_x = 252 / (100.8*2)
    scale_y = bev_shape[0] / (pc_range[4] - pc_range[1])
    scale_x = bev_shape[1] / (pc_range[3] - pc_range[0])

    box[:, :, 0] = (box[:, :, 0] - pc_range[0])* scale_x
    box[:, :, 1] = (box[:, :, 1] - pc_range[1])* scale_y
    return box

def load_lidar_points(file_path):
    nbr_dims = 4  # x,y,z,intensity
    scan = np.fromfile(file_path, dtype='float32')
    points = scan.reshape((-1, 5))[:, :nbr_dims]
    return points

def track_boxes_through_id(boxes_list):
    # boxes_list  --> {object_id1: [box, box, ...], object_id2: [], }
    track_dict = { object_id: [] for object_id in boxes_list[0] }
    for frame_boxes in boxes_list:
        for object_id in frame_boxes:
            if object_id in track_dict:
                track_dict[object_id].append(frame_boxes[object_id])
    return track_dict

# def get_trans_2d(box1, box2):
#     # box1 -> box2: [1, 7] # [x,y,z, hwl, yaw] 
#     diff = box2[0] - box1[0]
#     trans_2d = np.array(
#         [
#             [math.cos(diff[-1]), -math.sin(diff[-1]), diff[0]],
#             [math.sin(diff[-1]), math.cos(diff[-1]), diff[1]],
#             [0, 0, 1],
#         ]
#     )
#     return trans_2d

# def get_transformations(track_boxes_list):
#     # {object_id1: [box, box, ...], object_id2: [], }
#     transformation_list = { object_id: [] for object_id in track_boxes_list }
#     for object_id, boxes_list in track_boxes_list.items():
#         for i in range(1, len(boxes_list)):
#             if i > max_time_delay:
#                 break
#             transformation_list[object_id].append(get_trans_2d(boxes_list[i], boxes_list[0]))
            
#     return transformation_list

def get_center_yaw(box):
    # box [4, 2]
    center = np.mean(box, axis=0)
    diff_y = box[1, 1] - box[2, 1]
    diff_x = box[1, 0] - box[2, 0]
    yaw = math.atan2(diff_y, diff_x)
    return np.array([center[0], center[1], yaw])

def get_trans_2d(box1, box2):
    # box1 -> box2, [4, 2]
    pose1 = get_center_yaw(box1)
    pose2 = get_center_yaw(box2) # 表示从原点 -> 当前位置
    T_pose1 = np.array(
        [
            [math.cos(pose1[-1]), -math.sin(pose1[-1]), pose1[0]],
            [math.sin(pose1[-1]), math.cos(pose1[-1]), pose1[1]],
            [0, 0, 1],
        ]
    )
    T_pose2 = np.array(
        [
            [math.cos(pose2[-1]), -math.sin(pose2[-1]), pose2[0]],
            [math.sin(pose2[-1]), math.cos(pose2[-1]), pose2[1]],
            [0, 0, 1],
        ]
    )
    trans_2d = np.dot(T_pose2, inv(T_pose1))
    return trans_2d

def convert_scale_boxes(track_boxes_list):
    track_boxes_2d_list = { object_id: [] for object_id in track_boxes_list }
    for object_id in track_boxes_list:
        for box in track_boxes_list[object_id]:
            track_boxes_2d_list[object_id].append(np.asarray(get_3d_8points(box[0]))[:4,:2]) # box is (1, 7) -> (8, 3) --> (4, 2)
    
    track_boxes_2d_list = { object_id: np.stack(track_boxes_2d_list[object_id]) for object_id in track_boxes_2d_list}
    # scale box into bev_size
    track_boxes_2d_list = { object_id: scale_boxes(track_boxes_2d_list[object_id]) for object_id in track_boxes_2d_list}
    return track_boxes_2d_list

def get_transformations(track_boxes_2d_list):
    # {object_id1: (N, 4, 2), object_id2: (N, 4, 2) }
    transformation_list = { object_id: [] for object_id in track_boxes_2d_list }
    for object_id, boxes_np in track_boxes_2d_list.items():
        for i in range(1, boxes_np.shape[0]):
            if i > max_time_delay:
                break
            transformation_list[object_id].append(get_trans_2d(boxes_np[0], boxes_np[i]))   
    return transformation_list

def get_point_transformation(rects, transformations):
    # 得到每个rect中的点，并应用transformations
    points_ = getPointsInQuad(rects[0])
    point_trans = []
    for i in range(len(transformations)):
        points = points_.copy()
        transfomed_points = applyTransform(points, transformations[i]) # N,2
        # 转换为以左上角为原点的坐标
        # points[:, 1] = -(points[:, 1] - bev_shape[0])
        # transfomed_points[:, 1] = -(transfomed_points[:, 1] - bev_shape[0])
        # filter out points
        transform_tuple = np.concatenate([points, transfomed_points], axis=1) # 原坐标 -> 新坐标
        transform_tuple = filter_points(transform_tuple)
        point_trans.append(transform_tuple)
    return point_trans

def get_3d_8points(U):
    x,y,z, h, w, l, yaw_lidar = U[0], U[1], U[2], U[3], U[4], U[5],U[6],
    center_lidar = [x, y, z]
    liadr_r = np.matrix(
        [
            [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
            [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
            [0, 0, 1],
        ]
    )
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T
    return corners_3d_lidar.T

#
# 循环1         循环2    循环3
# ego  ------> agent1: [time_index1, time_index2, ...]
#              agent2:
#              agent3:
# 80000 * 5*5*10

def update_offset_map(transform_tuple, offset_map):
    # transform_tuple: 原始坐标， anchor的坐标点, N,2
    # transform_tuple: 变换后的坐标点, N, 2
    points = transform_tuple[:, :2].astype(np.int64)
    transformed_points = transform_tuple[:, 2:]
    offsets = transformed_points - points # (N, 2)
    # 在 transform_tuple[0]的位置记录下这个offset
    offset_map[points[:,1], points[:,0], :] = offsets
    return offset_map

def get_offset_maps(points_tracks, n_frame=max_time_delay):
    # n_frame = max([ len(one_boxes_points) for one_boxes_points in points_tracks.values() ])
    offset_maps = [ np.zeros((bev_shape[0], bev_shape[1], 2), dtype=np.float32) for _ in range(n_frame)]
    for one_boxes_points in points_tracks.values(): # 对每一个boxes
        for i, point_trans in enumerate(one_boxes_points):  # 对每一帧
            offset_maps[i] = update_offset_map(point_trans, offset_maps[i])
    return offset_maps

def label_one_ego_frame(ego_agent_index, frame_info, dataset_info_dict):
    ego_pose_t = frame_info['lidar_pose_{}'.format(ego_agent_index)]  # ego pose是当前的
    ego_pose = tfm_to_pose(ego_pose_t)
    prev_tokens = frame_info['prev_samples']
    prev_tokens[0] = frame_info['token'] # zero means no delay
    # anchor = xx
    offsets_l = []
    for agent_index in range(1, frame_info['agent_num'] + 1):

        point_list = []
        boxes_list = []
        #for time_delay, token in prev_tokens.items(): # 把当前帧也加进去
        for time_delay in range(max_time_delay+1):
            token = prev_tokens[time_delay]
            if token is None:
                break
            hist_frame = dataset_info_dict[token]
            points = load_lidar_points(hist_frame['lidar_path_{}'.format(agent_index)])
            pose_t = hist_frame['lidar_pose_{}'.format(agent_index)] # 
            # r_pose = np.dot(inv(ego_pose), pose)  # world->agent1, agent2->world
            r_pose = np.linalg.solve(ego_pose_t, pose_t)
            projected_points = box_utils.project_points_by_matrix_torch(points[:, :3], r_pose)
            point_list.append(projected_points)

            gt_boxes = {}
            output_dict = {} # [x,y,z, hwl, yaw] 
            gt_boxes['gt_boxes'] = hist_frame['labels_{}'.format(agent_index)]['gt_boxes_global'] # 
            gt_boxes['object_ids'] = hist_frame['labels_{}'.format(agent_index)]['gt_object_ids']
            # Project the objects under world coordinates into another coordinate
            box_utils.project_world_objects_v2x(gt_boxes, output_dict, ego_pose, pc_range, 'hwl', None)
            boxes_list.append(output_dict)
               
        # {object_id1: [box, box, ...], object_id2: [], }
        track_boxes_list = track_boxes_through_id(boxes_list)
        track_boxes_2d_list = convert_scale_boxes(track_boxes_list)
        transformation_list = get_transformations(track_boxes_2d_list) # 2d boxes transformation
        
        point_trans_list = {}
        for object_id in track_boxes_2d_list:
            point_trans_list[object_id] = get_point_transformation(track_boxes_2d_list[object_id], transformation_list[object_id])
        
        offset_maps = get_offset_maps(point_trans_list)
        offset_maps = np.concatenate([np.expand_dims(x, axis=0) for x in offset_maps], axis=0) # time_delay, h, w, 2
        offsets_l.append(offset_maps)

        # vis boxes
        vis_save_path =  os.path.join('tmp', '{}_ego_{}_index_{}.gif'.format(frame_info['token'], ego_agent_index, agent_index))
        vis_boxes(point_list, track_boxes_2d_list, vis_save_path)

        # vis points
        vis_save_path =  os.path.join('tmp', '{}_ego_{}_index_{}_flow.gif'.format(frame_info['token'], ego_agent_index, agent_index))
        vis_point_track(point_trans_list, point_list, vis_save_path)

    offsets_l =  np.concatenate([np.expand_dims(x, axis=0) for x in offsets_l], axis=0) # agent, time_delay, h, w, 2
    return offsets_l


def label_flow(token, dataset_info_dict):
    # for one frame
    frame_info = dataset_info_dict[token]
    agent_num = frame_info['agent_num']
    offsets_l = []
    for agent_index in range(1, agent_num+1):
        # all other agents will project to current agent
        offset_maps = label_one_ego_frame(agent_index, frame_info, dataset_info_dict)
        offsets_l.append(offset_maps)
    offsets_l =  np.concatenate([np.expand_dims(x, axis=0) for x in offsets_l], axis=0) # ego_agent, agent, time_delay, h, w, 2
    return offsets_l

if __name__ == '__main__':
    root_dir = '/data/datasets/V2X-smi/V2X-Sim-2.0'
    save_dir = 'offset_maps'
    if not os.path.exists(os.path.join(root_dir, save_dir)):
        os.mkdir(os.path.join(root_dir, save_dir))

    pkl_files = ['/hd_cache/users/junjie/projects/coperception/v2xsim2_info_generated/v2xsim_infos_train.pkl',
                 '/hd_cache/users/junjie/projects/coperception/v2xsim2_info_generated/v2xsim_infos_val.pkl',
                 '/hd_cache/users/junjie/projects/coperception/v2xsim2_info_generated/v2xsim_infos_test.pkl'
                 ]
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            dataset_info = pickle.load(f)  

        dataset_info_dict = OrderedDict()
        for data in dataset_info:
            dataset_info_dict[data['token']] = data
        
        for token in tqdm(dataset_info_dict):
            offset_maps = label_flow(token, dataset_info_dict)
            save_numpy(os.path.join(root_dir, save_dir, token + '.npy'), offset_maps)
           
