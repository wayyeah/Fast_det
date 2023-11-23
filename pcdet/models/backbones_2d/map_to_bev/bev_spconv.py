import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
import sparseconvnet as scn

def points_to_bev(points, point_range, batch_size, size):
    x_scale_factor = size[0] / (point_range[3] - point_range[0])
    y_scale_factor = size[1] / (point_range[4] - point_range[1])
    bev_shape = (batch_size, size[1], size[0])  # 更新形状
    bev = torch.ones(bev_shape, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) * -10

    if point_range[0] == 0:
        x_pixel_values = (points[:, 1] * x_scale_factor).to(torch.int)
    else:
        x_pixel_values = ((points[:, 1] + (point_range[3] - point_range[0]) / 2) * x_scale_factor).to(torch.int)
    y_pixel_values = ((points[:, 2] + (point_range[4] - point_range[1]) / 2) * y_scale_factor).to(torch.int)

    z_values = points[:, 3]
    mask = (x_pixel_values < size[0]) & (x_pixel_values >= 0) & (y_pixel_values < size[1]) & (y_pixel_values >= 0)
    
    batch_indices = points[mask, 0].long()
    x_indices = x_pixel_values[mask]
    y_indices = y_pixel_values[mask]
    z_vals = z_values[mask]  
    bev[batch_indices, y_indices, x_indices] = torch.maximum(bev[batch_indices, y_indices, x_indices], z_vals)
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

    all_coords = mask.nonzero(as_tuple=False)[:,[1,2,0]]  # 获取所有非-10元素的索引
    features = bev[mask]  # 提取所有非-10元素的特征
    features = features.unsqueeze(-1)  # 增加一个维度
    return all_coords, features




class BEVSPConv(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.conv_layers = scn.Sequential().add(
        scn.SubmanifoldConvolution(2, 1, 8, 3, False)  # 维持稀疏性的卷积
        ).add(
        scn.BatchNormReLU(8)
        ).add(
        scn.MaxPooling(2, 3, 2),
        ).add(
        scn.BatchNormReLU(8)
        ).add(
        scn.SubmanifoldConvolution(2, 8, 16, 3, False)  # 再次维持稀疏性的卷积
        ).add(
        scn.BatchNormReLU(16)
        ).add(
        scn.MaxPooling(2, 3, 2),
        ).add(
        scn.BatchNormReLU(16)
        ).add(
        scn.SubmanifoldConvolution(2, 16, 32, 3, False)  # 继续维持稀疏性的卷积
        ).add(
        scn.BatchNormReLU(32)
        ).add(
        scn.MaxPooling(2, 3, 2),
        ).add(
        scn.SparseToDense(2, 32)  # 最终转换为密集张量
        )

        self.inputSpatialSize=self.conv_layers.input_spatial_size(torch.LongTensor([self.size[1],self.size[0]]))
        self.input_layer=scn.InputLayer(2,self.inputSpatialSize)
        self.dense=scn.SparseToDense(2, 8)
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
        coords,combined_features=convert_bev_to_sparse(bev)
       
        input_tensor=self.input_layer([coords,combined_features])
        
        batch_dict['spatial_features'] = self.conv_layers(input_tensor)[:,:,:self.size[1]//8,:self.size[0]//8]
        """ print("batch_dict['spatial_features'].shape",batch_dict['spatial_features'].shape)
        np.save("/home/xmu/projects/xmuda/yw/Fast_det/bev_features.npy",batch_dict['spatial_features'].cpu().detach().numpy())
        np.save("/home/xmu/projects/xmuda/yw/Fast_det/bev.npy",batch_dict['bev'].cpu().detach().numpy())
        np.save("/home/xmu/projects/xmuda/yw/Fast_det/points.npy",batch_dict['points'].cpu().detach().numpy())
        exit()
        exit() """
        return batch_dict
    
    
#1123
# class BEVSPConv(nn.Module):
#     def __init__(self, model_cfg, **kwargs):
#         super().__init__()
#         self.model_cfg = model_cfg
#         self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
#         self.point_range=self.model_cfg.POINT_CLOUD_RANGE
#         self.size=self.model_cfg.SIZE
#         self.conv_layers = scn.Sequential().add(
#         scn.SubmanifoldConvolution(2, 1, 8, 3, False)  # 类似于 nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(8)
#         ).add(
#         scn.SubmanifoldConvolution(2, 8, 16, 3, False)  # 类似于 nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(16)
#         ).add(
#         scn.SubmanifoldConvolution(2, 16, 32, 3, False)  # 类似于 nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(32)
#         ).add(
#         scn.SparseToDense(2, 32)  # 将稀疏张量转换为密集张量
#         )
#         self.pooling_layer=nn.Sequential(
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         """ self.conv_layers = scn.Sequential().add(
#         scn.SubmanifoldConvolution(2, 1, 8, 3, False)  # 类似于 nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(8)
#         ).add(
#         scn.MaxPooling(2, 3, 2)  # 类似于 nn.MaxPool2d(kernel_size=2, stride=2)
#         ).add(
#         scn.SubmanifoldConvolution(2, 8, 16, 3, False)  # 类似于 nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(16)
#         ).add(
#         scn.MaxPooling(2, 3, 2)  # 再次 MaxPooling
#         ).add(
#         scn.SubmanifoldConvolution(2, 16, 32, 3, False)  # 类似于 nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         ).add(
#         scn.BatchNormReLU(32)
#         ).add(
#         scn.SparseToDense(2, 32)  # 将稀疏张量转换为密集张量
#         ) """
#         self.inputSpatialSize=self.conv_layers.input_spatial_size(torch.LongTensor([self.size[1],self.size[0]]))
#         self.input_layer=scn.InputLayer(2,self.inputSpatialSize)

#     def forward(self, batch_dict):
#         """
#         Args:
#             batch_dict:
#                 encoded_spconv_tensor: sparse tensor
#         Returns:
#             batch_dict:
#                 spatial_features:

#         """
#         bev=points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
#         batch_dict['bev']=bev
#         coords,combined_features=convert_bev_to_sparse(bev)
#         input_tensor=self.input_layer([coords,combined_features])
#         batch_dict['spatial_features'] = self.pooling_layer(self.conv_layers(input_tensor))
#         """ print("batch_dict['spatial_features'].shape",batch_dict['spatial_features'].shape)
#         np.save("/home/xmu/projects/xmuda/yw/Fast_det/bev_features.npy",batch_dict['spatial_features'].cpu().detach().numpy())
#         np.save("/home/xmu/projects/xmuda/yw/Fast_det/bev.npy",batch_dict['bev'].cpu().detach().numpy())
#         np.save("/home/xmu/projects/xmuda/yw/Fast_det/points.npy",batch_dict['points'].cpu().detach().numpy())
#         exit()
#         exit() """
#         return batch_dict