import torch.nn as nn
import numpy as np
import torch 
import torch.nn.functional as F
from pcdet.utils.bev_utils import points_to_bevs_two
class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

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


class BEVConvRes(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.block=BasicBlock
        self.num_blocks=[2,2,2]
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.in_channels = 64
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  
            self._make_layer(self.block, 64, self.num_blocks[0], stride=1),  
            self._make_layer(self.block, 128, self.num_blocks[1], stride=2), 
            self._make_layer(self.block, 256, self.num_blocks[2], stride=1), 
        )
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    def voxel_to_bev_batch(self,voxel, voxel_coords, voxel_num_points):
        exit()
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
        exit()
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
        bev=self.points_to_bev(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        import time 
        import numpy as np
        st=time.time()
        batch_dict['bev']=bev
        #batch_dict['gt_boxes']=self.boxes_to_bev(batch_dict['gt_boxes'],self.point_range)
        #bev=self.voxel_to_bev_batch(batch_dict['voxels'],batch_dict['voxel_coords'],batch_dict['voxel_num_points'])
        ''' np.save("/home/xmu/yw/Fast_det/bev.npy",bev.cpu().numpy())
        exit() '''
        batch_dict['spatial_features'] =self.conv_layers(bev)
        return batch_dict


class BEVConvVGG16(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super(BEVConvVGG16, self).__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        self.features = nn.Sequential(
            # input: batch_size x 3 x 224 x 224             b*2*1408*1600
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # output: batch_size x 64 x 224 x 224          b*64*1408*1600
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # output: batch_size x 64 x 224 x 224          b*64*1408*1600
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),      # output: batch_size x 64 x 112 x 112           b*64*704*800

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# output: batch_size x 128 x 112 x 112         b*128*704*800
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),# output: batch_size x 128 x 112 x 112         b*128*704*800
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),       # output: batch_size x 128 x 56 x 56           b*128*352*400

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# output: batch_size x 256 x 56 x 56          b*256*352*400
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1),# output: batch_size x 256 x 56 x 56         b*256*352*400
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# output: batch_size x 256 x 56 x 56          b*256*352*400
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),        # output: batch_size x 256 x 28 x 28         b*256*176*200

            nn.Conv2d(128, 256, kernel_size=3, padding=1),# output: batch_size x 512 x 28 x 28           b*512*176*200
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# output: batch_size x 512 x 28 x 28         b*512*176*200
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# output: batch_size x 512 x 28 x 28          b*512*176*200
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_dict):
        bev_combined=points_to_bevs_two(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        batch_dict['bev'] = bev_combined
        bev_combined = self.features(bev_combined)
        batch_dict['spatial_features'] =bev_combined
    
        return batch_dict
    
    

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    # 该模块负责建立一个卷积层和BN层
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
 
class RepVGGBlock(nn.Module):
    # 该模块用来产生RepVGGBlock，当deploy=False时，产生三个分支，当deploy=True时，产生一个结构重参数化后的卷积和偏置
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
 
        assert kernel_size == 3
        assert padding == 1
 
        padding_11 = padding - kernel_size // 2
 
        self.nonlinearity = nn.ReLU()
 
        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
 
        if deploy:
            # 当deploy=True时，产生一个结构重参数化后的卷积和偏置
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
 
        else:
            # 当deploy=False时，产生三个分支
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)
 
 
    def forward(self, inputs):
        # 当结构重参数化时，卷积和偏置之后跟上一个SE模块和非线性激活模块
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        # 如果没有线性映射shortcut时，则第三个分支输出为0
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # 训练时输出为三个分支输出结果相加，再加上SE模块和非线性激活
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
 
 
    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
 
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle
 
 
 
#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        # 用来将三个分支中的卷积算子和BN算子都转化为3x3卷积算子和偏置，然后将3x3卷积核参数相加，偏置相加
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # 输出将三个分支转化后的的3x3卷积核参数相加，偏置相加
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
 
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将第二个分支中的1x1卷积核padding为3x3的卷积核
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])
 
    def _fuse_bn_tensor(self, branch):
        # 将BN层的算子转化为卷积核的乘积和偏置
        if branch is None:
            return 0, 0
        # 当输入的分支是序列时，记录该分支的卷积核参数、BN的均值、方差、gamma、beta和eps（一个非常小的数）
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        # 当输入是第三个分支只有BN层时，添加一个只进行线性映射的3x3卷积核和一个偏置
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 输出新的卷积核参数（kernel * t），新的偏置（beta - running_mean * gamma / std）
        return kernel * t, beta - running_mean * gamma / std
 
    def switch_to_deploy(self):
        # 该模块用来进行结构重参数化，输出由三个分支重参数化后的只含有主分支的block
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        # 用self.__delattr__删除掉之前的旧的三个分支
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class BEVRepVGG(nn.Module):
    def __init__(self,  model_cfg, **kwargs  ):
        super(BEVRepVGG, self).__init__()
        width_multiplier=[0.75, 0.75, 2]
        num_blocks=[2, 4, 14]
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.point_range=self.model_cfg.POINT_CLOUD_RANGE
        self.size=self.model_cfg.SIZE
        # 定义RepVGG阶段
        self.in_planes = int(64 * width_multiplier[0])
        #self.training=False
        if self.training:
            deploy=False
        else:
            deploy=True
        self.stage0 = RepVGGBlock(in_channels=2, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=deploy)
        self.stage1 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[0], stride=2, deploy=deploy)
        self.stage2 = self._make_stage(int(128 * width_multiplier[2]), num_blocks[1], stride=2, deploy=deploy)

    def _make_stage(self, planes, num_blocks, stride, deploy):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, deploy=deploy))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, batch_dict):
        bev_combined=points_to_bevs_two(batch_dict['points'],self.point_range,batch_dict['batch_size'],self.size)
        if self.training==False:
            self.stage0.switch_to_deploy()
            for modules in self.stage1.modules():
                for module in modules.modules():
                    if(module.__class__.__name__=='RepVGGBlock'):
                        module.switch_to_deploy()
               
            for modules in self.stage2.modules():
                for module in modules.modules():
                    if(module.__class__.__name__=='RepVGGBlock'):
                        module.switch_to_deploy()
        out = self.stage0(bev_combined)
        out = self.stage1(out)
        out = self.stage2(out)
        batch_dict['bev'] = bev_combined
        batch_dict['spatial_features'] =out
    
        return batch_dict