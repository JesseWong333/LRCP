# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>
#
# DeforAttFusion， Modified by Junjie
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32 as MultiScaleDeformableAttnFunction
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch # 使用pytorch版本做调试

DEBUG=False

class DeformableAttentionV2(nn.Module):
    # attention weight的值不预测，按照传统的attention来计算， 如果相差不多，说明改融合没什么关系; 结果，相差不多
    def __init__(self, embed_dims, max_num_levels=2, num_heads=1, num_points=9):
        super(DeformableAttentionV2, self).__init__()
        self.sqrt_dim = np.sqrt(embed_dims)

        self.max_num_levels = max_num_levels  # number_level, 
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, max_num_levels*num_heads*num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, max_num_levels*num_heads * num_points)       
    
    def scale_dot_attention(self, q, k, v):
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # # 3, B, num_heads, N, C
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q: 1, 25200, 64   k: 1, 64, 25200, 18 -> 1, 25200, 18, 64 [0, 2, 3, 1]
        attn = (q.unsqueeze(2) * k).sum(-1) * self.sqrt_dim
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn.unsqueeze(-1) * v).sum(2)
       
        return x
    
    def sample_value(self, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
            sampling_locations: torch.Tensor):
        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, num_levels, num_points, _ =\
            sampling_locations.shape
        value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                                dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (H_, W_) in enumerate(value_spatial_shapes):
            # bs, H_*W_, num_heads, embed_dims ->
            # bs, H_*W_, num_heads*embed_dims ->
            # bs, num_heads*embed_dims, H_*W_ ->
            # bs*num_heads, embed_dims, H_, W_
            value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
                bs * num_heads, embed_dims, H_, W_)
            # bs, num_queries, num_heads, num_points, 2 ->
            # bs, num_heads, num_queries, num_points, 2 ->
            # bs*num_heads, num_queries, num_points, 2
            sampling_grid_l_ = sampling_grids[:, :, :,
                                            level].transpose(1, 2).flatten(0, 1)
            # bs*num_heads, embed_dims, num_queries, num_points
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (bs, num_queries, num_heads, num_levels, num_points) ->
        # (bs, num_heads, num_queries, num_levels, num_points) ->
        # (bs, num_heads, 1, num_queries, num_levels*num_points)
        output = torch.stack(sampling_value_list, dim=-2).flatten(-2).permute(0, 2, 3, 1)
        return output

    def forward(self, query, key, value, reference_points, spatial_shapes):
        number_levels, num_value, C = value.shape
   
        value = value.unsqueeze(0).contiguous().view(1, number_levels*num_value, C)

        bs, num_query, C = query.shape
        
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.max_num_levels, self.num_points, 2)  # 
        sampling_offsets = sampling_offsets[:, :, :, :number_levels, :, :].contiguous()

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        
        sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]  # (bs ,num_queries, num_heads, num_levels, num_points, 2)
        
        assert self.num_heads == 1
        value = value.view(bs, number_levels*num_value, self.num_heads, -1)
        sampled_value = self.sample_value(value, spatial_shapes, sampling_locations,)

        # 做传统的attention
        out = self.scale_dot_attention(query, sampled_value, sampled_value)
        return out

class DeformableAttention(nn.Module):
    def __init__(self, embed_dims, max_num_levels=2, num_heads=1, num_points=4):
        super(DeformableAttention, self).__init__()
        self.sqrt_dim = np.sqrt(embed_dims)

        self.max_num_levels = max_num_levels  # number_level, 
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dims, max_num_levels*num_heads*num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, max_num_levels*num_heads * num_points)         

    def forward(self, query, key, value, reference_points, spatial_shapes):
        """A customal version of deformable attention with resiual connection

        Args:
            query (_type_): (bs, num_query, embed_dims)
            key (_type_): _description_
            value (_type_): _description_
            reference_points (_type_): _description_
            spatial_: (num_levels, 2)
        Returns:    
            _type_: _description_
        """
        # value 是将多有的拉直到
        number_levels, num_value, C = value.shape
   
        value = value.unsqueeze(0).contiguous().view(1, number_levels*num_value, C)

        bs, num_query, C = query.shape
        
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.max_num_levels, self.num_points, 2)  # 
        sampling_offsets = sampling_offsets[:, :, :, :number_levels, :, :].contiguous()
        
        # sampling_offsets = torch.zeros(bs, num_query, self.num_heads, number_levels, self.num_points, 2, device=query.device)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                self.max_num_levels,
                                                self.num_points)
        attention_weights = attention_weights[:, :, :, :number_levels, :]
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, number_levels*self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query,
                                                self.num_heads,
                                                number_levels,
                                                self.num_points).contiguous()
        # attention_weights = torch.ones(bs, num_query,
        #                                         self.num_heads,
        #                                         number_levels,
        #                                         self.num_points, device=query.device) * (1/8)

        offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        
        sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]  # (bs ,num_queries, num_heads, num_levels, num_points, 2)
        
        assert self.num_heads == 1
        value = value.view(bs, number_levels*num_value, self.num_heads, -1)

        # output = MultiScaleDeformableAttnFunction.apply(
        #         value, spatial_shapes, torch.tensor([0], device=query.device), sampling_locations, attention_weights)
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        return output


class DeforAttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(DeforAttFusion, self).__init__()
        self.att = DeformableAttention(feature_dim)
        # self.att = DeformableAttentionV2(feature_dim)
    
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


    def forward(self, x, record_len, pairwise_t_matrix):
        """
        pairwise_t_matrix : [N,N,2,3]
        """
   
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        out = []
  
        for b, xx in enumerate(split_x):
            N = xx.shape[0]  # 当有的数据 N=1 时出错
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))

            cav_num = xx.shape[0]
            # 注意cam可能为1
            # generate reference points
            ref_2d = self.get_reference_points(
                H, W, dim='2d', bs=1, device=xx.device, dtype=xx.dtype)
            # ref_2d = torch.cat([ref_2d, ref_2d], dim=2)  # level层面
            ref_2d = ref_2d.repeat(1, 1, cav_num,1)

            spatial_shapes = [(H, W)] * cav_num  # the values has two level with the same shapes
            spatial_shapes = torch.as_tensor(
                spatial_shapes, dtype=torch.long, device=split_x[0].device)

            xx = xx.view(cav_num, C, -1).permute(0, 2, 1) # cav_num, H*W, C
            h = self.att(xx[:1, ...], xx, xx, ref_2d, spatial_shapes)  # 切片保持维度
            # h = h.permute(0, 2, 1).view(cav_num, C, H, W)[0, ...].unsqueeze(0)  # C, W, H before, 只取了前面的
            # h = xx.mean(dim=0, keepdim=True).permute(0, 2, 1)  # 没有permute也可运行，最后结果不对
            # h = xx[:1, :, :].permute(0, 2, 1)  # 发现的bug点1， 一定要在原来的数据连续的时候view； 可以先permute在view
            h = h.permute(0, 2, 1).view(1, C, H, W)  # 就是这个问题，其他的不行也是因为我没有permute
            out.append(h)
        return torch.cat(out, dim=0)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x



    def forward_debug(self, x, origin_x, record_len, pairwise_t_matrix):
        split_x = self.regroup(x, record_len)
        split_origin_x = self.regroup(origin_x, record_len)
        batch_size = len(record_len)
        C, H, W = split_x[0].shape[1:]  # C, W, H before
        H_origin, W_origin = split_origin_x[0].shape[2:]
        out = []
        from matplotlib import pyplot as plt
        for b, xx in enumerate(split_x):
          N = xx.shape[0]
          t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
          i = 0
          xx = warp_affine_simple(xx, t_matrix[i, :, :, :], (H, W))
          origin_xx = warp_affine_simple(split_origin_x[b], t_matrix[i, :, :, :], (H_origin, W_origin))

          for idx in range(N):
            plt.imshow(torch.max(xx[idx],0)[0].detach().cpu().numpy())
            plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/feature_{b}_{idx}")
            plt.clf()
            plt.imshow(torch.max(origin_xx[idx],0)[0].detach().cpu().numpy())
            plt.savefig(f"/GPFS/rhome/yifanlu/workspace/OpenCOOD/vis_result/debug_warp_feature/origin_feature_{b}_{idx}")
            plt.clf()
        raise