import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))


class BEVConvResH(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.block=BasicBlock
        self.num_blocks=[2,2,2]
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.in_channels = 64
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 752x752
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 376x376
            self._make_layer(self.block, 64, self.num_blocks[0], stride=1),  # 376x376
            self._make_layer(self.block, 128, self.num_blocks[1], stride=2),  # 188x188
            self._make_layer(self.block, 256, self.num_blocks[2], stride=1), 
        )# 188x188
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def voxel_to_bev_batch(self,voxel, voxel_coords, voxel_num_points):
        # Ensure long type for coordinates and num_points
        voxel_coords = voxel_coords.long()
        voxel_num_points = voxel_num_points.long()
        
        # Determine the size of the BEV based on the max coordinates in voxel_coords
        bev_shape = (int(voxel_coords[:, 0].max()) + 1, voxel_coords[:, 2].max() + 1, voxel_coords[:, 3].max() + 1)
        bev = torch.zeros(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Mask to filter out voxels with zero points
        mask = voxel_num_points > 0
        
        # Compute average heights for voxels with points
        avg_heights = (voxel[mask, :, 2] * (voxel_num_points[mask].unsqueeze(-1) > torch.arange(voxel.shape[1], device=voxel.device))).sum(dim=1) / voxel_num_points[mask]
        
        # Update bev using advanced indexing
        bev[voxel_coords[mask, 0], voxel_coords[mask, 2], voxel_coords[mask, 3]] = avg_heights

        return F.interpolate(bev.unsqueeze(1), size=(1504, 1504), mode='bilinear', align_corners=False)
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


    def boxes_to_bev(self, bboxes_3d, point_range):
        bboxes_3d[:,:,0]+=(point_range[3] - point_range[0]) / 2
        bboxes_3d[:,:,1]+=(point_range[4] - point_range[1]) / 2
        
        return bboxes_3d               
        
    def height_difference_bev(self, points, point_range, batch_size):
        x_scale_factor = 1504 / (point_range[3] - point_range[0])
        y_scale_factor = 1504 / (point_range[4] - point_range[1])
        z_scale_factor=255/(point_range[5]-point_range[2])
        bev_shape = (batch_size, 1, 1504, 1504)
        
        # Initialize BEV with negative infinity for max and positive infinity for min.
        bev_max = torch.full(bev_shape, float('-inf'), dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        bev_min = torch.full(bev_shape, float('inf'), dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        if(point_range[0]==0):
            x_pixel_values = ((points[:, 1] ) * x_scale_factor).to(torch.int)
        else:    
            x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
            
        y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)
        z_values = points[:, 3]
        z_pixel_values = ((points[:, 3] - point_range[2]) * z_scale_factor).to(torch.float)
      
        mask = (x_pixel_values < 1504) & (x_pixel_values >= 0) & (y_pixel_values < 1504) & (y_pixel_values >= 0)
        batch_indices = points[mask, 0].long()
        x_indices = x_pixel_values[mask]
        y_indices = y_pixel_values[mask]
        z_vals =  z_pixel_values[mask]
        
        # Update max and min height values
        bev_max[batch_indices, 0, y_indices, x_indices] = torch.maximum(bev_max[batch_indices, 0, y_indices, x_indices], z_vals)
        bev_min[batch_indices, 0, y_indices, x_indices] = torch.minimum(bev_min[batch_indices, 0, y_indices, x_indices], z_vals)
        
        # Compute the height difference
        bev_diff = bev_max - bev_min
        bev_diff[torch.isnan(bev_diff)] = 0
        bev_diff[torch.isinf(bev_diff)] = 0
        return bev_diff



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        import time 
        st=time.time()
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'])
        batch_dict['bev']=bev
        height_bev = self.height_difference_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'])
        combined_bev = torch.cat([bev, height_bev], dim=1)
        batch_dict['spatial_features'] = self.conv_layers(combined_bev)
        #batch_dict['gt_boxes']=self.boxes_to_bev(batch_dict['gt_boxes'],self.point_range)
        #bev=self.voxel_to_bev_batch(batch_dict['voxels'],batch_dict['voxel_coords'],batch_dict['voxel_num_points'])
        return batch_dict
