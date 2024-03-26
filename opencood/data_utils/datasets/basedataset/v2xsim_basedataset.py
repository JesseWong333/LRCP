# Author: Yangheng Zhao <zhaoyangheng-sjtu@sjtu.edu.cn>
import os
import pickle
from collections import OrderedDict
from typing import Dict
from abc import abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
import random

class V2XSIMBaseDataset(Dataset):
    """
        First version.
        Load V2X-sim 2.0 using yifan lu's pickle file. 
        Only support LiDAR data.
    """

    def __init__(self,
                 params: Dict,
                 visualize: bool = False,
                 train: bool = True):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.data_augmentor = DataAugmentor(params['data_augment'], train)

        self.data_dir = params['data_dir']
        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
        self.root_dir = root_dir

        print("Dataset dir:", root_dir)

        if 'train_params' not in params or \
                'max_cav' not in params['train_params']:
            self.max_cav = 5
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        assert self.label_type in ['lidar', 'camera']

        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                            else self.generate_object_center_camera
        self.generate_object_center_single = self.generate_object_center

        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
        
        with open(self.root_dir, 'rb') as f:
            dataset_info = pickle.load(f)
        self.dataset_info_pkl = dataset_info

        if "time_delay" in self.params:
            self.max_time_delay = params["time_delay"] # can only be [0, 1, 2, 3, 4, 5]
        else:
            self.max_time_delay = 0

        if "load_history" in self.params and params["load_history"]:
            self.load_history = True
        else:
            self.load_history = False
        
        if "history_frame" in self.params:
            self.history_frame = self.params["history_frame"] 
        else:
            self.history_frame = 0
        
        if "bev_h" in self.params:
            self.bev_h = self.params["bev_h"]

        if "bev_w" in self.params:
            self.bev_w = self.params["bev_w"]

        # TODO param: one as ego or all as ego?
        self.ego_mode = 'one'  # "all"

        self.reinitialize()

    def reinitialize(self):
        self.scene_database = OrderedDict()
        if self.ego_mode == 'one':
            self.len_record = len(self.dataset_info_pkl)
        else:
            raise NotImplementedError(self.ego_mode)
        
        # {token: id} look up dict
        self.token_id_dict = {}
        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.token_id_dict[scene_info['token']] = i
        self.id_token_dict = {self.token_id_dict[key]:key for key in self.token_id_dict}

        for i, scene_info in enumerate(self.dataset_info_pkl):
            self.scene_database.update({i: OrderedDict()}) # scene_database, frame_info 用数字做key
            cav_num = scene_info['agent_num']
            assert cav_num > 0

            if self.train:
                cav_ids = 1 + np.random.permutation(cav_num)
            else:
                cav_ids = list(range(1, cav_num + 1))
            
            self.scene_database[i]['prev'] = OrderedDict()
            for key in scene_info['prev_samples']:
                if scene_info['prev_samples'][key] is not None:
                    self.scene_database[i]['prev'][key] = self.token_id_dict[scene_info['prev_samples'][key]]
                else:
                    self.scene_database[i]['prev'][key] = None

            for j, cav_id in enumerate(cav_ids):
                if j > self.max_cav - 1:
                    print('too many cavs reinitialize')
                    break
                # 随机选择一个做ego 每个agent的字段有lidar， lidar_pose， gt_boxes_global， cav_id
                self.scene_database[i][cav_id] = OrderedDict()

                self.scene_database[i][cav_id]['ego'] = j==0 # cav_ids shuffle 了， 相当于随机选了一个做ego

                self.scene_database[i][cav_id]['lidar'] = scene_info[f'lidar_path_{cav_id}']
                # need to delete this line is running in /GPFS
                # self.scene_database[i][cav_id]['lidar'] = \
                #     self.scene_database[i][cav_id]['lidar'].replace("/GPFS/rhome/yifanlu/workspace/dataset/v2xsim2-complete", "/data/datasets/V2X-smi/V2X-Sim-2.0") # change here

                self.scene_database[i][cav_id]['params'] = OrderedDict()
                self.scene_database[i][cav_id][
                    'params']['lidar_pose'] = tfm_to_pose(
                        scene_info[f"lidar_pose_{cav_id}"]
                    )  # [x, y, z, roll, pitch, yaw]
                self.scene_database[i][cav_id]['params'][
                    'vehicles'] = scene_info[f'labels_{cav_id}'][
                        'gt_boxes_global']
                self.scene_database[i][cav_id]['params'][
                    'object_ids'] = scene_info[f'labels_{cav_id}'][
                        'gt_object_ids'].tolist()

    def __len__(self) -> int:
        return self.len_record

    @abstractmethod
    def __getitem__(self, index):
        pass

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.

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

        data = OrderedDict()
        # {     
        #     'prev':
        #     'cav_id0':{
        #         'ego': bool,
        #         'params': {
        #           'lidar_pose': [x, y, z, roll, pitch, yaw],
        #           'vehicles':{
        #                   'id': {'angle', 'center', 'extent', 'location'},
        #                   ...
        #               }
        #           },# 包含agent位置信息和object信息
        #         'camera_data':,
        #         'depth_data':,
        #         'lidar_np':,
        #         ...
        #     }
        #     'cav_id1': ,
        #     ...
        # }
        scene = self.scene_database[idx]  # 这里的scene指一帧
        # v2x-sim的pari-wise_transformation有, ego_index, agent_index, time_index, 
        # lidar-pose, box-pose; 要跟原来的兼容
        ego_id = -1
        for cav_id, cav_content in scene.items():
            if cav_id == 'prev':
                continue
            data[f'{cav_id}'] = OrderedDict()
            data[f'{cav_id}']['ego'] = cav_content['ego']
            if cav_content['ego']:
                ego_id = cav_id
            data[f'{cav_id}']['params'] = cav_content['params']  # used for cooperative label
            # load the corresponding data into the dictionary
            data[f'{cav_id}']['lidar_np'] = self.read_lidar(cav_content['lidar'])

        if self.load_history:
            for cav_id in scene:
                if cav_id == 'prev':
                    continue
                if data[f'{cav_id}']['ego']:
                    # for ego, no time_delay
                    continue
                history_frame_index_l, time_index = self.frame_select(scene)

                history_info = self.load_info_history(time_index, scene, cav_id) # lidar, pose, box_label
                data[f'{cav_id}']['lidar_np'] = history_info[0]  # over write lidar_np
                data[f'{cav_id}']['params_single'] = history_info[1]  # only used for single supervision
                history_infos = [self.load_info_history(index, scene, cav_id) for index in history_frame_index_l]
                data[f'{cav_id}']['lidar_np_history'] = [history_info[0] for history_info in history_infos]
                data[f'{cav_id}']['params_history'] = [history_info[1] for history_info in history_infos]

                data[f'{cav_id}']['time_delay'] = time_index * 100
                offset_path = os.path.join(self.data_dir, 'offset_maps', self.id_token_dict[idx] + '.npy')
                offset_map = torch.from_numpy(np.load(offset_path)) # ego_agent, agent, time_delay, h, w, 2
                if time_index == 0:
                    data[f'{cav_id}']['params']['offset'] = torch.zeros((1, self.bev_h, self.bev_w, 2))
                else:
                    data[f'{cav_id}']['params']['offset'] = offset_map[ego_id-1, cav_id-1, time_index-1, :, :, :].unsqueeze(0)
        return data
    
    @staticmethod
    def read_lidar(path):
        nbr_dims = 4  # x,y,z,intensity
        scan = np.fromfile(path, dtype='float32')
        points = scan.reshape((-1, 5))[:, :nbr_dims]
        return points
        
    def load_info_history(self, time_index, frame_info, cav_id):
        if not self.is_frame_exits(time_index, frame_info):
            return None, None

        if time_index == 0:
            frame_extract = frame_info
        else:
            frame_extract = self.scene_database[frame_info['prev'][time_index]]

        lidar_np = self.read_lidar( frame_extract[cav_id]['lidar'])
        frame_params = frame_extract[cav_id]['params']
        return lidar_np, frame_params
    
    @staticmethod
    def is_frame_exits(index, frame_info):
        if index == 0:
            return True
        if frame_info['prev'][index] is not None:
            return True
        return False
    
    def frame_select(self, frame_info):
        def is_track_frame_exits(track_n_frame):
            for i in range(1, track_n_frame):
                if not self.is_frame_exits(i, frame_info):
                    return False
            return True
         
        if self.train:
            # 训练时随机选择
            max_trial = 5
            for _ in range(max_trial):
                time_index = random.randint(0, self.max_time_delay)
                history_index_list = [index for index in range(time_index, time_index + self.history_frame)]
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
        
    def generate_object_center_lidar(self, cav_contents, reference_lidar_pose):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.

        Notice: it is a wrap of postprocessor function

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """

        return self.post_processor.generate_object_center_v2x(
            cav_contents, reference_lidar_pose)

    def generate_object_center_camera(self, cav_contents, reference_lidar_pose):
        raise NotImplementedError()

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