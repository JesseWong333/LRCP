# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.defor_encoder_multi_scale import DeforEncoderMultiScale
from opencood.models.sub_modules.defor_encoder_multi_scale_single_agent import DeforEncoderMultiScaleSingleAgent
from opencood.models.sub_modules.defor_encoder import DeforEncoder
from opencood.models.sub_modules.auto_encoder import AutoEncoder

DEBUG = False

class DeforBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.compress = False

        self.discrete_ratio = model_cfg['voxel_size'][0]
        self.downsample_rate = 1

        if 'compression' in model_cfg and model_cfg['compression'] > 0:
            self.compress = True
            self.compress_layer = model_cfg['compression']

        if 'layer_nums' in self.model_cfg:

            assert len(self.model_cfg['layer_nums']) == \
                   len(self.model_cfg['layer_strides']) == \
                   len(self.model_cfg['num_filters'])

            layer_nums = self.model_cfg['layer_nums']
            layer_strides = self.model_cfg['layer_strides']
            num_filters = self.model_cfg['num_filters']
        else:
            layer_nums = layer_strides = num_filters = []

        if 'upsample_strides' in self.model_cfg:
            assert len(self.model_cfg['upsample_strides']) \
                   == len(self.model_cfg['num_upsample_filter'])

            num_upsample_filters = self.model_cfg['num_upsample_filter']
            upsample_strides = self.model_cfg['upsample_strides']

        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.fuse_modules = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        if self.compress:
            self.compression_modules = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]

            if self.compress and self.compress_layer - idx > 0:
                self.compression_modules.append(AutoEncoder(num_filters[idx],
                                                            self.compress_layer-idx))

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        if "multi_scale" in model_cfg and model_cfg['multi_scale']:
            self.multi_scale = True
            self.defor_encoder = DeforEncoderMultiScale(model_cfg['defor_encoder'])
            self.defor_encoder_single = DeforEncoderMultiScaleSingleAgent(model_cfg['defor_encoder_single'])
        else:
            self.multi_scale = False
            self.defor_encoder = DeforEncoder(model_cfg['defor_encoder'])

        # project multi-feature to the same dimention
        if self.multi_scale:
            input_proj_list = []
            for i, _ in enumerate(range(num_levels)):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(num_filters[i], num_upsample_filters[i], kernel_size=1),  # we use the same size filters as the privious upsample filters
                    nn.BatchNorm2d(num_upsample_filters[i]),
                ))

            self.input_proj = nn.ModuleList(input_proj_list)

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        x = spatial_features
        H, W = x.shape[2:]
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        ups = []
        ups_multi_scale = []
        # if we choose multi-scale, both the single supervised branch and fused branch use multi-scale
       
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)  # 2*B, C, H, W

            if self.compress and i < len(self.compression_modules):
                x = self.compression_modules[i](x)

            if self.multi_scale:
                ups_multi_scale.append(self.input_proj[i](x)) # # project all the features to the same dimmension
            else:
                if len(self.deblocks) > 0:
                    ups.append(self.deblocks[i](x)) # # 2*B, C, H, W
                else:
                    ups.append(x)

        if self.multi_scale:
            single_features = self.defor_encoder_single(ups_multi_scale)
            fused_features = self.defor_encoder(ups_multi_scale, record_len, pairwise_t_matrix, data_dict['offset_GT'], data_dict['pred_offset']) # dim=128
            data_dict['single_features'] = single_features
            data_dict['fused_features'] = fused_features
        else:
            if len(ups) > 1:
                x = torch.cat(ups, dim=1)
            elif len(ups) == 1:
                x = ups[0]
            if len(self.deblocks) > len(self.blocks):
                x = self.deblocks[-1](x)
            data_dict['single_features'] = x
            fused_features = self.defor_encoder(x, record_len, pairwise_t_matrix, data_dict['offset'], data_dict['offset_mask'])  # dim=384
            data_dict['fused_features'] = fused_features
        return data_dict    
