import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
from tqdm import tqdm
def points_to_bev(points, point_range, batch_size,size):
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*-10
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    z_values = points[:, 3]
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask].long()
    y_indices = y_pixel_values[mask].long()
    #z_vals = z_values_normalized[mask]
    z_vals=z_values[mask] #取消z正则化                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], z_vals)
    return bev
def points_to_bev_zmin(points, point_range, batch_size,size):
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))*10
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    z_values = points[:, 3]
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask].long()
    y_indices = y_pixel_values[mask].long()
    z_vals=z_values[mask] #取消z正则化                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    bev[batch_indices, 0, y_indices, x_indices] = torch.minimum(bev[batch_indices, 0, y_indices, x_indices], z_vals)
    return bev
def points_nums_to_bev(points, point_range, batch_size,size):
    
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    bev_height_sum = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    z_values = points[:, 3]
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask].long()
    y_indices = y_pixel_values[mask].long()
    z_vals = z_values[mask]
    indices = torch.stack([batch_indices, torch.zeros_like(batch_indices), y_indices, x_indices], dim=1)
    
    # 计算每个网格的点数
    bev_height_sum = bev_height_sum.index_put_(tuple(indices.t()), z_vals.unsqueeze(1), accumulate=True)
    ones = torch.ones_like(batch_indices, dtype=torch.float32)
    bev = bev.index_put_(tuple(indices.t()), ones, accumulate=True)   
    bev[bev == 0] = 1
    bev_average_z = bev_height_sum / bev 
    bev/=50
    return bev,bev_average_z                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
def intensity_to_bev(points, point_range, batch_size,size):
    x_scale_factor = size[0]/ (point_range[3] - point_range[0])
    y_scale_factor = size[1]/ (point_range[4] - point_range[1])
    bev_shape = (batch_size, 1, size[1] , size[0] )
    bev = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if(point_range[0]==0):
        x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
    else:    
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
    intensity = points[:, 4]
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask].long()
    y_indices = y_pixel_values[mask].long()
    i_vals=intensity[mask]
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], i_vals)
    return bev
class DepthwiseSeparableConvWithShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.shuffle = ChannelShuffle(groups=in_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.shuffle(x)
        x = self.pointwise(x)
        return x

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)
        return x
def points_to_bevs(points, point_range, batch_size,size):
    
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
    bev_zmin[batch_indices, 0, y_indices, x_indices] = torch.minimum(bev_zmin[batch_indices, 0, y_indices, x_indices], z_vals)
    bev_zmax[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_zmax[batch_indices, 0, y_indices, x_indices], z_vals)
    bev_i[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_i[batch_indices, 0, y_indices, x_indices], i_vals)
    return bev,bev_average_z,bev_zmin,bev_zmax,bev_i

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
    x_indices = x_pixel_values[mask].long()
    y_indices = y_pixel_values[mask].long()
    #z_vals = z_values_normalized[mask]
    z_vals=z_values[mask] #取消z正则化 
    i_vals=i_values[mask]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], z_vals)
    bev_intensity[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_intensity[batch_indices, 0, y_indices, x_indices], i_vals)
    return torch.cat([bev,bev_intensity], dim=1)  


class BEVConvS(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConvWithShuffle(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        
        """import numpy as np
        np.save('/mnt/16THDD/yw/Fast_det/bev.npy',bev_combined.cpu().numpy())
        np.save('/mnt/16THDD/yw/Fast_det/points.npy',batch_dict['points'].cpu().numpy())
        exit() """
        for i, layer in enumerate(self.conv_layers):
            bev_combined = layer( bev_combined)
            if i==2:
                x_conv1=bev_combined
            if i==6:
                x_conv2=bev_combined
            if i==13:
                x_conv3=bev_combined
            if i==14:
                x_conv4=bev_combined
        batch_dict['spatial_features'] = (bev_combined)
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
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
class BEVConvSNormal(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict   
class BEVConvSNormalV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Final layers
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
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
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict  
class BEVConvSV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            DepthwiseSeparableConvWithShuffle(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConvWithShuffle(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
    
class BEVConvSV3(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Depthwise separable convolution
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
class BEVConvSV3One(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            # Depthwise separable convolution
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        spatial_features = self.conv_layers(bev)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict    
class BEVConvSV4(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        spatial_features = self.conv_layers(bev)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict


class BEVConvSV5(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            DepthwiseSeparableConvWithShuffle(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            DepthwiseSeparableConvWithShuffle(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConvWithShuffle(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Final layers
            DepthwiseSeparableConvWithShuffle(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict        
    
    
class BEVConvSV8(nn.Module):#多种特征
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(5, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            nn.Conv2d(12, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        #self.size=[200,176]
        bev,bev_average_z,bev_zmin,bev_zmax,bev_i =points_to_bevs(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_combined = torch.cat([bev,bev_average_z,bev_zmin,bev_zmax,bev_i], dim=1)  # Stack along the channel dimensionW
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
    
class BEVConvSV7(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConvWithShuffle(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Final layers
            nn.Conv2d(32, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
    
class BEVConvSV10(nn.Module):#多种特征
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        
        self.maxpool=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
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
        #self.size=[200,176]
       
        bev_points,bev_zmean=points_nums_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_zmin=points_to_bev_zmin(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_combined = torch.cat([bev_points, bev_intensity,bev_zmean,bev_zmin], dim=1)  # Stack along the channel dimensionW
        batch_dict['bev'] = bev_combined
        batch_dict['spatial_features'] = (self.maxpool(bev_combined))
        return batch_dict
    
    
class BEVConvSV4_3(nn.Module):#多种特征
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(5, 8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev,bev_average_z,bev_zmin,bev_zmax,bev_i =points_to_bevs(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_combined = torch.cat([bev,bev_average_z,bev_zmin,bev_zmax,bev_i], dim=1)  # Stack along the channel dimensionW
        batch_dict['bev'] = bev_combined
        spatial_features = self.conv_layers(bev_combined)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
    
class BEVConvSV4Wise(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            DepthwiseSeparableConvWithShuffle(1, 4, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(4),
            nn.ReLU(),
            DepthwiseSeparableConvWithShuffle(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            DepthwiseSeparableConvWithShuffle(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            DepthwiseSeparableConvWithShuffle(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        spatial_features = self.conv_layers(bev)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict

class BEVConvSV4WiseV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(4),
            nn.ReLU(),
            DepthwiseSeparableConvWithShuffle(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            DepthwiseSeparableConvWithShuffle(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Final layers
            DepthwiseSeparableConvWithShuffle(16, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*n*200*176
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        spatial_features = self.conv_layers(bev)
        batch_dict['spatial_features'] = (spatial_features)
        return batch_dict
