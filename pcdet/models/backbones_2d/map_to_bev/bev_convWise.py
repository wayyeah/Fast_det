import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
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

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class BEVConvWise(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = nn.Sequential(
            # Initial convolution to increase channels
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Further convolutions to reduce spatial dimensions and increase channels
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            DepthwiseSeparableConv(128, self.num_bev_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    

    def points_to_bev(self,points, point_range, batch_size,size):
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

    def intensity_to_bev(self,points, point_range, batch_size,size):
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        batch_dict['bev']=bev
        batch_dict['spatial_features'] =self.conv_layers(bev)
        return batch_dict


class BEVConvWiseV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Depthwise separable convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise convolution
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bottleneck layer before next deep convolution
            nn.Conv2d(128, 64, kernel_size=1),  # Reduce the number of features
            nn.ReLU(),

            # Depthwise separable convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise convolution
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Final layers
            nn.Conv2d(128, self.num_bev_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


    

    def points_to_bev(self,points, point_range, batch_size,size):
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

    def intensity_to_bev(self,points, point_range, batch_size,size):
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        batch_dict['bev']=bev
        batch_dict['spatial_features'] =self.conv_layers(bev)
        return batch_dict
    
class BEVConvWiseV3(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), #b*64*1600*1408
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*64*800*704
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*64*400*352
            # Depthwise separable convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise convolution #b*128*400*352
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Final layers
            nn.Conv2d(128, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*256*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def points_to_bev(self,points, point_range, batch_size,size):
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

    def intensity_to_bev(self,points, point_range, batch_size,size):
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        batch_dict['bev']=bev
        batch_dict['spatial_features'] =self.conv_layers(bev)
        return batch_dict
    
    
class BEVConvWiseWithI(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1), #b*64*1600*1408
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*64*800*704
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*64*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(128, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*256*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*256*200*176
        )


    

    def points_to_bev(self,points, point_range, batch_size,size):
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

    def intensity_to_bev(self,points, point_range, batch_size,size):
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        batch_dict['bev']=bev
        bev_intensity = self.intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict


class BEVConvWiseWithIV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_layers = nn.Sequential(
            # Existing layers
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1), #b*32*1600*1408
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*32*800*704
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*32*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(128, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*256*400*352
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #b*256*200*176
        )


    

    def points_to_bev(self,points, point_range, batch_size,size):
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

    def intensity_to_bev(self,points, point_range, batch_size,size):
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

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        batch_dict['bev']=bev
        bev_intensity = self.intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict
    
    
class BEVConvWiseWithIV3(nn.Module):
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
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*32*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(128, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict
    
class BEVConvWiseWithIV4(nn.Module):
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
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*32*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Final layers
            nn.Conv2d(64, self.num_bev_features, kernel_size=3, stride=1, padding=1), #b*n*400*352
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict
    
class BEVConvWiseWithIV5(nn.Module):
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
            # Depthwise separable convolution
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1),
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict
    
class BEVConvWiseWithIV6(nn.Module):
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
            # Depthwise separable convolution
            DepthwiseSeparableConv(16, self.num_bev_features, kernel_size=3, stride=1, padding=1),
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict    
    
class BEVConvWiseWithIV7(nn.Module):
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
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*8*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(8, self.num_bev_features, kernel_size=3, stride=1, padding=1),
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict    
    
    
class BEVConvWiseWithIV8(nn.Module): #全部使用深度可分离卷积
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_layers = nn.Sequential(
            # Existing layers
            DepthwiseSeparableConv(2,8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
            DepthwiseSeparableConv(8,16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
            # Depthwise separable convolution
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Final layers
            DepthwiseSeparableConv(32,self.num_bev_features,kernel_size=3,stride=1,padding=1),
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
        batch_dict['spatial_features'] = self.conv_layers(bev_combined)
        return batch_dict
    
    
class BEVBase(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        #1*1*1600*1408
        self.conv_1=nn.Sequential(
            # Existing layers
            DepthwiseSeparableConv(2,8, kernel_size=3, stride=1, padding=1), #b*8*1600*1408
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),#b*8*800*704
        )
        self.conv_2=nn.Sequential(
            DepthwiseSeparableConv(8,16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  #b*16*400*352
        )
        self.conv_3=nn.Sequential(
            # Depthwise separable convolution
            DepthwiseSeparableConv(16, self.num_bev_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_bev_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
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
        output_1=self.conv_1(bev_combined)
        output_2=self.conv_2(output_1)
        output_3=self.conv_3(output_2)
        batch_dict['spatial_features'] = (output_3)
        return batch_dict