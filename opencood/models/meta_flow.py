# -*- coding: utf-8 -*-
# Author: Junjie Wang <junjie.wang@umu.se>

# give the moudle a cool name


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet.models.utils import LearnedPositionalEncoding
from opencood.models.sub_modules.defor_encoder_multi_scale_single_agent import Block
from torch.nn.init import normal_

class SineTimeEncoding(nn.Module):
    """Time encoding with sine and cosine functions.
    """

    def __init__(self,
                 num_feats,
                 temperature=10000):
        super(SineTimeEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature

    def forward(self, time, device, shape=None):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        bs = time.shape[0]
        time = time.view(bs, 1, 1) * torch.ones(shape).to(device)
        
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = time.unsqueeze(-1) / dim_t
     
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        pos_x = torch.concat(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3)

        return pos_x

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class CNNBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.deblocks = nn.ModuleList()
        upsample_strides = args['upsample_strides']
        num_levels = args['num_levels']
        num_filters = args['num_filters']
        num_upsample_filters = args['num_upsample_filters']
        embed_dims = args['embed_dims']
        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]

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

        self.shrink_conv = nn.Sequential(
                    nn.Conv2d(embed_dims * num_levels, embed_dims, kernel_size=1),  # we use the same size filters as the privious upsample filters
                    nn.BatchNorm2d(embed_dims)
                )

    def forward(self, x):
        # x: [ (b, t, c0, h0, w0), (b, t, c1, h1, w1), (b, t, c2, h2, w2) ]
        x1, x2, x3 = x[0], x[1], x[2]
        b,t,c1,h1,w1 = x1.shape
        _,_,c2,h2,w2 = x2.shape
        _,_,c3,h3,w3 = x3.shape

        # deconv 到同一个维度
        x1 = self.deblocks[0](x1.view(b*t,c1,h1,w1))
        x2 = self.deblocks[1](x2.view(b*t,c2,h2,w2))
        x3 = self.deblocks[2](x3.view(b*t,c3,h3,w3))

        x = torch.cat([x1, x2, x3], dim=1)  # b*t, c, h, w
        x = self.shrink_conv(x) # b*t, 128, h, c
        x = x.view(b, t, 128, h1, w1)
        return x

class MetaFlow(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = CNNBackbone(args['backbone'])

        self.bev_h = 100
        self.bev_w = 252

        embed_dims = args['embed_dims']


        self.positional_encoding = LearnedPositionalEncoding(        
            num_feats=embed_dims//2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w)
        self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, embed_dims)
        self.temporal_encoding = SineTimeEncoding(embed_dims) 
        self.frame_embeds = nn.Parameter(
            torch.FloatTensor(args['n_frame'], embed_dims))  # torch.Tensor默认初始化会出现Nan?
        normal_(self.frame_embeds)
        # decoder block
        # self.block1 = Block(embed_dims, num_heads_self=8, num_points_self=8, 
        #                     num_heads_cross=8, num_points_cross=8, 
        #                     dropout=0.1, max_num_agent=1, feature_level=5, 
        #                     cfgs = ["cross_attn", "norm", "ffn", "norm", "cross_attn", "norm", "ffn", "norm", "cross_attn", "norm", "ffn", "norm"])
        # self.block2 = Block(embed_dims, num_heads_self=8, num_points_self=8, 
        #                     num_heads_cross=8, num_points_cross=8, 
        #                     dropout=0.1, max_num_agent=1, feature_level=5,
        #                     cfgs = ["self_attn", "norm", "ffn", "norm", "self_attn", "norm", "ffn", "norm"])

        self.blocks = nn.ModuleList()
        block_cfgs = args["block_cfgs"]
        for block_cfg in block_cfgs:
            self.blocks.append(Block(*block_cfg))

        self.classifier = nn.Linear(embed_dims, 2)

    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        # H, W is
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
        
    def forward(self, x, time):
        # time: N
        xx = self.encoder(x) # N, C, H, W
        B, N, C, H, W = xx.shape
        mulframe_feats = torch.chunk(xx, chunks=N, dim=1)
        feat_flatten = []
        for lvl, feat in enumerate(mulframe_feats):
            feat = feat.squeeze(1)
            _, c, h, w = feat.shape
            feat = feat.flatten(2).transpose(1, 2) # Bs, h*w, C
            feat = feat + self.frame_embeds[None, lvl:lvl + 1, :].to(feat.dtype)
            feat_flatten.append(feat)  # [Bs, h*w, C]
        feat_flatten = torch.cat(feat_flatten, 1) 

        temporal_encoding = self.temporal_encoding(time, xx.device, [B, H, W]).view(B, H*W, -1) # B, H*W, C
        # xx = xx + temporal_encoding

        bev_queries = self.bev_embedding.weight.to(xx.dtype).repeat(B, 1, 1) # h*w, C -> B, H*W, C
        bev_queries = bev_queries + temporal_encoding
        bev_mask = torch.zeros((B, self.bev_h, self.bev_w),
                                device=bev_queries.device).to(xx.dtype)
        bev_pos = self.positional_encoding(bev_mask).to(xx.dtype).flatten(2, 3).permute(0, 2, 1) # B, C, h*w -> B, h*w, C

        # bev_pos = bev_pos + temporal_encoding

        ref_2d = self.get_reference_points(
                H, W, bs=B, device=xx.device, dtype=xx.dtype)
        ref_2d = ref_2d.repeat(1,1,N,1)

        spatial_shapes = [(H, W)] * N  # the values has two levels with the same shape
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=xx.device) # N*2
        
        spatial_shapes_self = [(self.bev_h, self.bev_w)]
        spatial_shapes_self = torch.as_tensor(spatial_shapes_self, dtype=torch.long, device=feat.device)
        for block in self.blocks:
            bev_queries = block(bev_queries, bev_pos, feat_flatten, ref_2d, spatial_shapes, spatial_shapes_self)
        # bev_queries = self.block3(bev_queries, bev_pos, feat_flatten, ref_2d, spatial_shapes, spatial_shapes_self)
        x = self.classifier(bev_queries)
        x = x.view(B, H, W, 2)
        return x
   