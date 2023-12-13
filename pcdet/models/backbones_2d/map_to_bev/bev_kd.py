import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
def points_to_bevs_two(points, point_range, batch_size,size):
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*-10
    bev_intensity= torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    z_values = points[:, 3]
    i_values=points[:, 4]
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask]
    y_indices = y_pixel_values[mask]
    z_vals=z_values[mask] 
    i_vals=i_values[mask]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], z_vals)
    bev_intensity[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_intensity[batch_indices, 0, y_indices, x_indices], i_vals)
    return torch.cat([bev,bev_intensity], dim=1)  
def points_to_bevs_all(points, point_range, batch_size,size):
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bev_height_sum = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bev_zmin = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*10
    bev_zmax = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*-10
    bev_i = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    z_values = points[:, 3]
    i_values=points[:, 4]
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask]
    y_indices = y_pixel_values[mask]
    z_vals = z_values[mask]
    i_vals=i_values[mask]
    indices = torch.stack([batch_indices, torch.zeros_like(batch_indices), y_indices, x_indices], dim=1)
    # 计算每个网格的点数
    bev_height_sum = bev_height_sum.index_put_(tuple(indices.t()), z_vals.unsqueeze(1), accumulate=True)
    ones = torch.ones_like(batch_indices, dtype=torch.float32)
    bev = bev.index_put_(tuple(indices.t()), ones, accumulate=True)   
    bev[bev == 0] = 1
    bev_average_z = bev_height_sum / bev 
    bev/=50
    bev_zmax[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_zmax[batch_indices, 0, y_indices, x_indices], z_vals)
    bev_zmin[batch_indices, 0, y_indices, x_indices] = torch.minimum(bev_zmin[batch_indices, 0, y_indices, x_indices], z_vals)
   
    bev_i[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_i[batch_indices, 0, y_indices, x_indices], i_vals)
    return torch.cat([bev,bev_zmin,bev_zmax,bev_i], dim=1)  
class BEVKD(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.max_pool2d=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_layers_1=nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )#b*8*1600*1408
        self.conv_layers_2=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv_layers_3=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_layers_4=nn.Sequential(
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1),
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
        x_conv1=self.conv_layers_1(bev_combined)
        x_conv1_maxpool=self.max_pool2d(x_conv1)
        x_conv2=self.conv_layers_2(x_conv1_maxpool)
        x_conv2_maxpool=self.max_pool2d(x_conv2)
        x_conv3=self.conv_layers_3(x_conv2_maxpool)
        x_conv3_maxpool=self.max_pool2d(x_conv3)
        spatial_features=self.conv_layers_4(x_conv3_maxpool)
        batch_dict['spatial_features'] = (spatial_features)
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv3_maxpool,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict   
    
class BEVKDV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.max_pool2d=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_layers_1=nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )#b*8*1600*1408
        self.conv_layers_2=nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv_layers_3=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_layers_4=nn.Sequential(
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1),
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
        bev_combined=points_to_bevs_all(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev'] = bev_combined
        x=self.conv_layers_1(bev_combined)
        x=self.max_pool2d(x)
        x=self.conv_layers_2(x)
        x=self.max_pool2d(x)
        x=self.conv_layers_3(x)
        x=self.max_pool2d(x)
        spatial_features=self.conv_layers_4(x)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict   