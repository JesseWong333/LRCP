import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from functools import partial

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,3,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

class FlowPred(nn.Module):

    def __init__(self, embeding_dim, batchNorm=True, dim_scale=4) -> None:
        super().__init__()
        self.batchNorm = batchNorm

        self.feat_project = nn.Linear(embeding_dim, embeding_dim // dim_scale)
        self.norm = nn.LayerNorm(embeding_dim // dim_scale)

        self.conv1   = conv(self.batchNorm,   embeding_dim // dim_scale * 2,   64, kernel_size=3)
        self.conv2   = conv(self.batchNorm, 64, 64) # 开始预测， 最多scale 两次

        self.conv3   = conv(self.batchNorm, 64,  128, kernel_size=3, stride=2)
        self.conv3_1   = conv(self.batchNorm, 128,  128, kernel_size=3, ) # predict

        self.conv4 = conv(self.batchNorm, 128,  256, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(259, 64)

        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(259)
        self.predict_flow2 = predict_flow(131)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        # X: S, C, H, W  S=2为两车情形
        x = x.permute(0, 2, 3, 1)
        x = self.feat_project(x)
        x = self.norm(x) # S, H, W, C
        x = x.permute(0, 3, 1, 2).flatten(start_dim=0, end_dim=1).unsqueeze(0)  # 卷积通道在前

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        flow4 = self.predict_flow4(out_conv4)  # 输出多一维度， 为mask
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_conv4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)  # [1, 3, H, W]
     
        return flow2,flow3,flow4
