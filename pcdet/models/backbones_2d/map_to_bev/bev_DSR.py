from .bev_convRes import RepVGGBlock
from .bev_convS import DepthwiseSeparableConvWithShuffle,SE,points_to_bevs_two
import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
class DSRBlock(nn.Module):
    def __init__(self,input_channels,output_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise=nn.Sequential(
            DepthwiseSeparableConvWithShuffle(input_channels,input_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            )
        self.se=SE(input_channels)
        self.repvgg=nn.Sequential(
            RepVGGBlock(input_channels,output_channels,kernel_size, stride, padding, deploy=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            RepVGGBlock(output_channels,output_channels,kernel_size, stride, padding, deploy=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            )
    def forward(self,x):
        x=self.depthwise(x)
        x=self.se(x)
        x=self.repvgg(x)
        return x
class BEVDSR(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1), #b*8*1504*1504
            nn.BatchNorm2d(4),
            nn.ReLU(),
            DSRBlock(4, 8, kernel_size=3, stride=1, padding=1), 
            DSRBlock(8, 8, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*752*752
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DSRBlock(16, 32, kernel_size=3, stride=1, padding=1), 
            DSRBlock(32, 32, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DSRBlock(64, self.num_bev_features, kernel_size=3, stride=1, padding=1), 
            DSRBlock(64, self.num_bev_features, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.num_bev_features, self.num_bev_features, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
        )
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        bev_combined=points_to_bevs_two(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev'] = bev_combined
        if(self.training==False):
            for module in self.conv_layers.modules():
                #print("switch to deploy")
                if module.__class__.__name__=='RepVGGBlock':
                    module.switch_to_deploy()
        spatial_features = self.conv_layers( bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
