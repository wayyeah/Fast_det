import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
import sparseconvnet as scn

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
    z_values_normalized = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask]
    y_indices = y_pixel_values[mask]
    #z_vals = z_values_normalized[mask]
    z_vals=z_values[mask] #取消z正则化                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], z_vals)
    return bev

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
    x_indices = x_pixel_values[mask]
    y_indices = y_pixel_values[mask]
    i_vals=intensity[mask]
    bev[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev[batch_indices, 0, y_indices, x_indices], i_vals)
    return bev
def convert_bev_to_sparse(bev):
    mask = bev != -10
    coords = mask.nonzero(as_tuple=False)  # 获取非-10元素的索引
    features = bev[mask].unsqueeze(1)  # 提取特征并增加一个特征维度
    print("coords.shape",coords.shape)
    print(features.shape)
    exit()
    return coords, features


class BEVSPConv(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
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
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev']=bev
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        coords,combined_features=convert_bev_to_sparse(bev)
        
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict