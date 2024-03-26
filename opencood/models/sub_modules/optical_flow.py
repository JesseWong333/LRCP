# modified by Junjie Wang

# 显示光流预测来校准定位误差
# 使用pip的方案

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def get_3d_embedding(xyz, C, cat_coords=True):
    B, N, D = xyz.shape
    assert(D==3)

    x = xyz[:,:,0:1]
    y = xyz[:,:,1:2]
    z = xyz[:,:,2:3]
    div_term = (torch.arange(0, C, 2, device=xyz.device, dtype=torch.float32) * (1000.0 / C)).reshape(1, 1, int(C/2))
    
    pe_x = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    pe_z = torch.zeros(B, N, C, device=xyz.device, dtype=torch.float32)
    
    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)
    
    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)
    
    pe_z[:, :, 0::2] = torch.sin(z * div_term)
    pe_z[:, :, 1::2] = torch.cos(z * div_term)
    
    pe = torch.cat([pe_x, pe_y, pe_z], dim=2) # B, N, C*3
    if cat_coords:
        pe = torch.cat([pe, xyz], dim=2) # B, N, C*3+3
    return pe

def MLPMixer(S, input_dim, dim, output_dim, depth, expansion_factor=4, dropout=0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        nn.Linear(input_dim, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(S, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, output_dim)
    )

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    # go to 0,1 then 0,2 then -1,1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

class FlowPred(nn.Module):

    def __init__(self, max_iters, embeding_dim, dim_scale=4) -> None:
        super().__init__()
        self.max_iters = max_iters
        self.radius = 4
        self.S = 2
        # 先对embeding project，减少计算量
        # 投影之后要norm一下
        self.feat_project = nn.Linear(embeding_dim, embeding_dim // dim_scale)
        self.norm = nn.LayerNorm(embeding_dim // dim_scale)
        
        self.embeding_dim = embeding_dim // dim_scale

        # 预测模型相关
        kitchen_dim =   (2*self.radius + 1)**2 + self.embeding_dim + 64*3 + 3  # embeding_dim是传输过来的feature C
        self.to_delta = MLPMixer(
            S=self.S,
            input_dim=kitchen_dim,
            dim=512,
            output_dim=self.S*(self.embeding_dim+2),
            depth=4,  # 层数， 默认12还很多
        )

        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.embeding_dim, self.embeding_dim),
            nn.GELU(),
        )
    
    def predict(self, fhid, fcorr, flow):
        """
        Args:
            fhid (_type_): B*N, S, C
            fcorr (_type_): B*N, S, R*R  # 这个传输进来的尺寸不对
            flow (_type_): # B*N,S,3
        """
        B_N = fhid.shape[0]
        flow_sincos = get_3d_embedding(flow, 64, cat_coords=True) # B*N, S, 64*3 + 3
        x = torch.cat([fhid, fcorr, flow_sincos], dim=2) # B*N, S, C+R*R+64*3+3 # 最后一项是多维特征
        delta = self.to_delta(x) # # B*N, S*(self.embeding_dim+2)
        delta = delta.reshape(B_N, self.S, self.embeding_dim+2)
        return delta

    def cal_corr(self, init_f, target_f):
        # 计算相关度矩阵; init_f(原特征), target_f(维护的一个特征序列)
        # [B, S, N, C] # 
        B, S, N, C = target_f.shape
        init_f = init_f.permute(0, 1, 3, 2)
        corrs = torch.matmul(target_f, init_f) # B, S, N, N  # 这个相关性矩阵太消耗计算资源了；必须仅计算周围一定点的个数， 后面也是只采样指定个
        corrs =  corrs / torch.sqrt(torch.tensor(C).float())
        return corrs
    
    def sample_coor(self, corrs, coords):
        """
        Args:
            corrs (_type_): [B, S, N, H, W]
            coords (_type_): [B, S, N, 2]
        """
        B, S, N, H, W = corrs.shape
        r = self.radius
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device) # R*R**2 R=2*r+1
        centroid = coords.flatten(start_dim=0, end_dim=2)[:, None, None, :] # B*S*N, 1, 1, 2
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        coords = centroid + delta
        sampled_corrs = bilinear_sampler(corrs.flatten(start_dim=0, end_dim=2).unsqueeze(1), coords)
        sampled_corrs = sampled_corrs.view(B, S, N, -1) # B, S, N, R*R
        return sampled_corrs
    

    def forward(self, X, ref_2d):
        """ 简化版pips, 不使用特征金字塔    

        Args:
            X: S, C, H, W  S=2为两车情形
            ref_2d: # (1, H*W, 2) # 注意后两维是先w,再h, 一样，就是标准做法
        """
        
        # featue projection
        X = X.permute(0, 2, 3, 1)
        X = self.feat_project(X)
        X = self.norm(X)
        X = X.permute(0, 3, 1, 2)

        # 第一帧 是ego，就是密集预测
        S, C, H, W = X.shape

        ref_2d = ref_2d.permute(0, 2, 1).view(1, 2, H, W)
        ref_2d = F.interpolate(ref_2d, scale_factor=0.5)
        ref_2d = ref_2d.permute(0, 2, 3, 1).view(1, )

        B, N, D = ref_2d.shape  # B=1 

        assert H*W == N

        # init coords
        coords = ref_2d[:, None, :, :].repeat(1, S, 1, 1) # B, S, N, 2
        ffeats  = X[:1].permute(0, 2, 3, 1).reshape(B, H*W, C).unsqueeze(1).repeat(1, S, 1, 1)  # [B, S, N, C]
        X = X.permute(0, 2, 3, 1).reshape(S, H*W, C).unsqueeze(0) # [B, S, N, C]

        # 开始迭代
        coord_predictions = []
        for itr in range(self.max_iters):
            coords = coords.detach()
            # 计算相关矩阵
            corrs = self.cal_corr(X, ffeats).view(B, S, N, H, W) # 原文 多尺度对应fcps
           
            # 采样coords附近的相关矩阵
            fcorrs = self.sample_coor(corrs, coords)  # B, S, N, R*R
            fcorrs_ = fcorrs.permute(0, 2, 1, 3).flatten(start_dim=0, end_dim=1)  # B*N, S, R*R

            flows_ = (coords - coords[:,0:1]).permute(0,2,1,3).reshape(B*N, S, 2)  #  B*N, S, 2 减去初始坐标
            times_ = torch.linspace(0, S, S, device=X.device).reshape(1, S, 1).repeat(B*N, 1, 1) # B*N, S, 1 
            flows_ = torch.cat([flows_, times_], dim=2) # B*N,S,3
            ffeats_ = ffeats.permute(0,2,1,3).flatten(start_dim=0, end_dim=1) # B, N, S, C -> B*N, S, C

            delta_all_ = self.predict(ffeats_, fcorrs_, flows_) # B*N, S, C+2
            delta_coords_ = delta_all_[:,:,:2]
            delta_feats_ = delta_all_[:,:,2:]

            ffeats_ = delta_feats_ + ffeats_ #  [B*N, S, C] + [B*N, S, C]
            ffeats = ffeats_.reshape(B, N, S, self.embeding_dim).permute(0,2,1,3) # B,S,N,C  # 这里写回去了
            
            coords = coords + delta_coords_.reshape(B, N, S, 2).permute(0,2,1,3) # [B, S, N, 2] + 
            coord_predictions.append(coords)
        
        # 这个loss就是一个sequence, 我们要最后一个时刻的
        return coord_predictions


