import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
#Unet架构
class BEVConvU(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # Encoding layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2, dilation=2),  # Using dilated convolutions
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Decoding layers
        self.dec2 = nn.Sequential(
            nn.Conv2d(256+128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128+64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)  # Can be adjusted based on desired output channels

        
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






    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """

        bev=self.voxel_to_bev_batch(batch_dict['voxels'],batch_dict['voxel_coords'],batch_dict['voxel_num_points'])
        # Encoding path
        enc1_out = self.enc1(bev)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)

        # Decoding path
        dec2_out = self.dec2(torch.cat([enc2_out, enc3_out], dim=1))
        dec1_out = self.dec1(torch.cat([enc1_out, dec2_out], dim=1))
        
        spatial_features = self.final_conv(dec1_out)
        batch_dict['spatial_features'] = spatial_features
        return batch_dict
