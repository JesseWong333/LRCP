import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

class DAIRV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']

        self.split_info = read_json(split_dir)
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info_processed_updated.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")  # 使用vehicle作为帧的ID
            self.co_data[veh_frame_id] = frame_info

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        if "time_delay" in self.params:
            self.max_time_delay = params["time_delay"] # can only be [0, 1, 2, 3, 4, 5]
        else:
            self.max_time_delay = 0
        
        if "load_history" in self.params and params["load_history"]:
            self.load_history = True
        else:
            self.load_history = False

        if "train_stage" in self.params:
            self.train_stage = self.params["train_stage"]

        if "history_frame" in self.params:
            self.history_frame = self.params["history_frame"] 
        else:
            self.history_frame = 0
        
        if "bev_h" in self.params:
            self.bev_h = self.params["bev_h"]

        if "bev_w" in self.params:
            self.bev_w = self.params["bev_w"]
    
    def reinitialize(self):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        veh_frame_id = self.split_info[idx]
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # pose of agent 
        lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)
        # data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir, frame_info['cooperative_label_path']))
        data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'].replace("label_world", "label_world_backup"))) 
        data[0]['params']['vehicles_all'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'])) 

        data[1]['params']['vehicles_front'] = [] # we only load cooperative label in vehicle side
        data[1]['params']['vehicles_all'] = [] # we only load cooperative label in vehicle side

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))
            
            data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))

 
        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'] = self.load_lidar_timeindex(0, frame_info)
            
            if self.load_history:
                history_frame_index_l, time_index = self.frame_select(frame_info)
                if history_frame_index_l is None:
                    if self.train:
                        return self.retrieve_base_data(random.randint(0, len(self.split_info)-1))
                    else:
                        return None
                data[1]['lidar_np'] = self.load_lidar_timeindex(time_index, frame_info) # over write lidar_np
                data[1]['lidar_np_history'] = [self.load_lidar_timeindex(index, frame_info) for index in history_frame_index_l]
                data[1]['time_delay'] = time_index * 100
                offset_path = os.path.join(self.root_dir, 'offset_maps_fix_flip', 'offset_'+veh_frame_id+'.npy')
                offset_map = torch.from_numpy(np.load(offset_path))   
                if time_index == 0:
                    data[1]['params']["offset"] = torch.zeros((1, self.bev_h, self.bev_w, 2))
                else:
                    data[1]['params']["offset"] = offset_map[time_index-1].unsqueeze(0)
              
        # Label for single side
        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id)))
        data[0]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        if self.load_history:
            # for single supervised,  we should also load the history
            inf_frame_id = self.get_infid_with_timeindex(time_index, frame_info)
        
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
        return data
    
    @staticmethod
    def is_frame_exits(index, frame_info):
        if index == 0:
            return True
        if frame_info["previous_inf_"+str(index)] is not None:
            return True
        return False
    
    def get_infid_with_timeindex(self, time_index, frame_info):
        if time_index == 0:
            inf_id = frame_info["infrastructure_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        else:
            inf_id = frame_info["previous_inf_"+str(time_index)][0].split("/")[-1].replace(".pcd", "")
        return inf_id

    def load_lidar_timeindex(self, time_index, frame_info):
        if not self.is_frame_exits(time_index, frame_info):
            return None
        if time_index == 0:
            lidar_np, _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
        else:
            lidar_np, _ = \
                    pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["previous_inf_"+str(time_index)][0]))
        return lidar_np
    
    def frame_select(self, frame_info):
        # return a list of Ids 
        def is_track_frame_exits(track_n_frame):
            for i in range(1, track_n_frame):
                if not self.is_frame_exits(i, frame_info):
                    return False
            return True

        if self.train:
            # 训练时随机选择: time_delay:time_delay + history_frame
            max_trial = 5
            for _ in range(max_trial): 
                # 有可能当前的帧全部不满足 todo
                if self.train_stage == "stage1":
                    time_index = np.floor(np.random.exponential(scale=2.0)).astype(np.int32) # 均值为2
                    if time_index > self.max_time_delay:
                        time_index = self.max_time_delay
                else:
                    time_index = random.randint(0, self.max_time_delay) # 随机在最大time_delay中选一帧
                history_index_list = [index for index in range(time_index, time_index + self.history_frame)] # a list of size, history_frame

                if is_track_frame_exits(time_index + self.history_frame): # 要求每一帧都在，不能缺帧
                    return history_index_list, time_index
                else:
                    continue  
            return None, None
        else:
            # when do test, self.max_time_delay means all the
            history_index_list = [index for index in range(self.max_time_delay, self.max_time_delay + self.history_frame)]
            if is_track_frame_exits(self.max_time_delay  + self.history_frame):
                return history_index_list, self.max_time_delay
            else:
                return None, None

    def __len__(self):
        return len(self.split_info)

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
        veh or inf 's coordinate
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)

    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask
    