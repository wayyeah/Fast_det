from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning
from pcdet.datasets.once.once_eval.iou_utils import rotate_iou_gpu_eval
def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    #inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    #eps = 1e-6
    #union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi] # constrain to [0-pi]
    iou3d[diff_rot > np.pi/2] = 0 # unmatched if diff_rot > 90
    return iou3d
# Suppress only NumbaPerformanceWarning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)
def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

gt_path='/mnt/4tssd1/yw/Fast_det/data/kitti/kitti_infos_val.pkl'
result_path=None
try:
    gt_pkl=read_pkl(gt_path)
except:
    gt_path=None
try:
    det_pkl=read_pkl(result_path)
except:
    result_path=None
if gt_path is None:
    print("输入gt路径")
    gt_path=input()
if result_path is None:
    print("输入result路径")
    result_path=input()


class_names=['Car','Pedestrian','Cyclist']
gt_pkl=read_pkl(gt_path)
det_pkl=read_pkl(result_path)

print("Over All:")
eval_det_annos = copy.deepcopy(det_pkl)
eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
#eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
#result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
#print(result)
print(det_pkl[0]['name'])
exit()
#统计不同IOU<0.5,<0.7的数量
for i in range (len(gt_pkl)):
    gt_boxes=gt_pkl[i]['annos']['gt_boxes_lidar']
    pred_boxes=det_pkl[i]['boxes_lidar']
    
    ious=iou3d_kernel_with_heading(gt_boxes,pred_boxes)
    for row in ious:
        if(max(row)<0.5):
            print("iou<0.5")
        elif(max(row)<0.7 and max(row)>=0.5):
            print("iou<0.7")
        elif (max(row)>=0.7):
            print("iou>=0.7")
    print(ious)