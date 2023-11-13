import torch
import torch.nn as nn



class UniBEVBackbone(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range = self.model_cfg.POINT_CLOUD_RANGE
        self.size = self.model_cfg.SIZE
        
        # Initial downsample
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> batch x 32 x 800 x 704
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> batch x 64 x 400 x 352
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> batch x 128 x 200 x 176
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Bottleneck layers
        self.bottleneck_layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Upsample layers
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # -> batch x 128 x 400 x 352
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),  # -> batch x 256 x 800 x 704
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Output layer to match the target shape batch x 256 x 200 x 176
        self.output_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1),  # -> batch x 256 x 200 x 176
            nn.BatchNorm2d(256),
            nn.ReLU(),
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
    def forward(self, batch_dict):
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        x = self.downsample_layers(bev)
        x = self.bottleneck_layers(x)
        x = self.upsample_layers(x)
        x = self.output_layers(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict
    
class UniBEVBackboneV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range = self.model_cfg.POINT_CLOUD_RANGE
        self.size = self.model_cfg.SIZE
        # Initial downsample
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> batch x 32 x 800 x 704
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> batch x 64 x 400 x 352
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> batch x 128 x 200 x 176
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Bottleneck layers
        self.bottleneck_layers = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Upsample layers
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),  # -> batch x 128 x 400 x 352
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),  # -> batch x 256 x 800 x 704
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Output layer to match the target shape batch x 256 x 200 x 176
        self.output_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1),  # -> batch x 256 x 200 x 176
            nn.BatchNorm2d(256),
            nn.ReLU(),
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
    def forward(self, batch_dict):
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        x = self.downsample_layers(bev)
        x = self.bottleneck_layers(x)
        x = self.upsample_layers(x)
        x = self.output_layers(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict    