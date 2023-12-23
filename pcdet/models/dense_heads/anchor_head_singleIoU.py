import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner_add_gt import AxisAlignedTargetAssigner

class AnchorHeadSingleIoU(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        self.conv_iou = nn.Conv2d(input_channels, self.num_anchors_per_location, kernel_size=3, stride=1, padding=1, bias=False,)
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
        self.top_labels = torch.zeros([70400], dtype=torch.long, ).cuda() 

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
    def get_rdiou(self, bboxes1, bboxes2):
        x1u, y1u, z1u = bboxes1[:,:,0], bboxes1[:,:,1], bboxes1[:,:,2]
        l1, w1, h1 =  torch.exp(bboxes1[:,:,3]), torch.exp(bboxes1[:,:,4]), torch.exp(bboxes1[:,:,5])
        t1 = torch.sin(bboxes1[:,:,6]) * torch.cos(bboxes2[:,:,6])
        x2u, y2u, z2u = bboxes2[:,:,0], bboxes2[:,:,1], bboxes2[:,:,2]
        l2, w2, h2 =  torch.exp(bboxes2[:,:,3]), torch.exp(bboxes2[:,:,4]), torch.exp(bboxes2[:,:,5])
        t2 = torch.cos(bboxes1[:,:,6]) * torch.sin(bboxes2[:,:,6])

        # we emperically scale the y/z to make their predictions more sensitive.
        x1 = x1u
        y1 = y1u * 2
        z1 = z1u * 2
        x2 = x2u
        y2 = y2u * 2
        z2 = z2u * 2

        # clamp is necessray to aviod inf.
        l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)
        j1, j2 = torch.ones_like(h2), torch.ones_like(h2)

        volume_1 = l1 * w1 * h1 * j1
        volume_2 = l2 * w2 * h2 * j2

        inter_l = torch.max(x1 - l1 / 2, x2 - l2 / 2)
        inter_r = torch.min(x1 + l1 / 2, x2 + l2 / 2)
        inter_t = torch.max(y1 - w1 / 2, y2 - w2 / 2)
        inter_b = torch.min(y1 + w1 / 2, y2 + w2 / 2)
        inter_u = torch.max(z1 - h1 / 2, z2 - h2 / 2)
        inter_d = torch.min(z1 + h1 / 2, z2 + h2 / 2)
        inter_m = torch.max(t1 - j1 / 2, t2 - j2 / 2)
        inter_n = torch.min(t1 + j1 / 2, t2 + j2 / 2)

        inter_volume = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0) \
            * torch.clamp((inter_d - inter_u),min=0) * torch.clamp((inter_n - inter_m),min=0)
        
        c_l = torch.min(x1 - l1 / 2,x2 - l2 / 2)
        c_r = torch.max(x1 + l1 / 2,x2 + l2 / 2)
        c_t = torch.min(y1 - w1 / 2,y2 - w2 / 2)
        c_b = torch.max(y1 + w1 / 2,y2 + w2 / 2)
        c_u = torch.min(z1 - h1 / 2,z2 - h2 / 2)
        c_d = torch.max(z1 + h1 / 2,z2 + h2 / 2)
        c_m = torch.min(t1 - j1 / 2,t2 - j2 / 2)
        c_n = torch.max(t1 + j1 / 2,t2 + j2 / 2)

        inter_diag = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2 + (t2 - t1)**2
        c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2 + torch.clamp((c_d - c_u),min=0)**2  + torch.clamp((c_n - c_m),min=0)**2

        union = volume_1 + volume_2 - inter_volume
        u = (inter_diag) / c_diag
        rdiou = inter_volume / union
        return u, rdiou
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)
        iou_preds=self.conv_iou(spatial_features_2d)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        iou_preds=iou_preds.permute(0,2,3,1).contiguous()
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['iou_preds'] = iou_preds
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        self.predict_boxes_when_training=True
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds,iou_preds=iou_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
        self.add_module(
            'iou_pred_loss_func',
            loss_utils.WeightedSmoothL1LossIoU(sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0)
        )
    def get_iou_pred_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        iou_preds = self.forward_ret_dict["iou_preds"]
        
        batch_size = int(box_preds.shape[0])
        box_cls_labels = box_cls_labels.view(batch_size, -1)
        iou_pos_preds = iou_preds.view(batch_size, -1, 1)
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        
        
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
      
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
       
        u, rdiou = self.get_rdiou(box_preds, box_reg_targets)
        rdiou=rdiou*2-1
        rdiou=rdiou.view(batch_size, -1,1)
        iou_pos_preds=iou_pos_preds.view(batch_size, -1,1)
        iou_preds_loss=self.iou_pred_loss_func(iou_pos_preds, rdiou, reg_weights)
        iou_preds_loss = iou_preds_loss.sum() / batch_size
        return iou_preds_loss
    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds,iou_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)
            iou_preds: (N, H, W, C1)
        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        batch_iou_preds=iou_preds.view(batch_size, num_anchors, -1) if not isinstance(iou_preds, list) \
            else torch.cat(iou_preds, dim=1).view(batch_size, num_anchors, -1)
        # 找到每个预测中的最大值的索引
        _, indices = torch.max(batch_cls_preds, dim=2)

        # 将IOU值插入到最大值位置
        batch_cls_preds[torch.arange(batch_size)[:, None], torch.arange(batch_cls_preds.shape[1]), indices] = batch_iou_preds.squeeze(-1)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds      
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        iou_pred_loss=self.get_iou_pred_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss+iou_pred_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict