import torch
import torch.nn as nn

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
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        padding: int = 1,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out
class UniBEVBackbone(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range = self.model_cfg.POINT_CLOUD_RANGE
        self.size = self.model_cfg.SIZE
        # Initial downsample
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),  # -> batch x 32 x 800 x 704
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
         
    def forward(self, batch_dict):
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
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
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            BasicBlock(4, 8,downsample=True),
            BasicBlock(8, 16,downsample=True),
            BasicBlock(16, 32,downsample=True),
            BasicBlock(32, 64, stride=2,downsample=True),
            BasicBlock(64, 128, stride=2,downsample=True)
        )
        
        # Bottleneck layers
        self.bottleneck_layers = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        
        # Upsample layers
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Output layer to match the target shape batch x 256 x 200 x 176
        self.output_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    def forward(self, batch_dict):
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        x = self.downsample_layers( bev_combined)
        x = self.bottleneck_layers(x)
        x = self.upsample_layers(x)
        x = self.output_layers(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict
    
class UniBEVBackboneV3(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range = self.model_cfg.POINT_CLOUD_RANGE
        self.size = self.model_cfg.SIZE
        self.downsample_layers = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(32, 32,downsample=True),
            BasicBlock(32, 64,downsample=True),
            BasicBlock(64, 128,downsample=True),
           
        )
        
        # Bottleneck layers
        self.bottleneck_layers = nn.Sequential(
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        
        # Upsample layers
        self.upsample_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        # Output layer to match the target shape batch x 256 x 200 x 176
        self.output_layers = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    def forward(self, batch_dict):
        bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        bev_intensity = intensity_to_bev(batch_dict['points'], self.point_range, batch_dict['batch_size'], self.size)
        bev_combined = torch.cat([bev, bev_intensity], dim=1)  # Stack along the channel dimension
        batch_dict['bev'] = bev_combined
        x = self.downsample_layers( bev_combined)
        x = self.bottleneck_layers(x)
        x = self.upsample_layers(x)
        x = self.output_layers(x)
        batch_dict['spatial_features_2d'] = x
        return batch_dict    