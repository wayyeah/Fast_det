from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
import copy
import numpy as np
import pickle
from numba import cuda
import warnings
from numba.core.errors import NumbaPerformanceWarning

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
dis=[0,1,2,3]
for i in dis:
    if(i==0):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        
        for j in range(len(eval_det_annos)):
            mask=eval_det_annos[j]['boxes_lidar'][:,0]<30
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            mask=eval_gt_annos[j]['gt_boxes_lidar'][:,0]<30
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]
                 
        print("0-30m:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    elif(i==1):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        for j in range(len(eval_det_annos)):
            mask = (eval_det_annos[j]['boxes_lidar'][:, 0] >= 30) & (eval_det_annos[j]['boxes_lidar'][:, 0] < 50)
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            mask=(eval_gt_annos[j]['gt_boxes_lidar'][:,0]>=30) & (eval_gt_annos[j]['gt_boxes_lidar'][:,0]<50)
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]
        print("30-50m:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    elif(i==2):
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        for j in range(len(eval_gt_annos)):
            while(len(eval_gt_annos[j]['gt_boxes_lidar'])<len(eval_gt_annos[j]['name'])):
                new_row = np.array([[0, 0, 0, 0, 0, 0, 0]])
                eval_gt_annos[j]['gt_boxes_lidar'] = np.vstack([eval_gt_annos[j]['gt_boxes_lidar'], new_row])
        for j in range(len(eval_det_annos)):
            mask=eval_det_annos[j]['boxes_lidar'][:,0]>=50
            for key in eval_det_annos[j].keys():
                if key!='frame_id':
                    eval_det_annos[j][key]= eval_det_annos[j][key][mask]
        for j in range(len(eval_gt_annos)):
            mask=eval_gt_annos[j]['gt_boxes_lidar'][:,0]>=50
            for key in eval_gt_annos[j].keys():
                if key!='frame_id':
                    eval_gt_annos[j][key]= eval_gt_annos[j][key][mask]        
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        print("50~inf:")
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)
    else:
        print("Over All:")
        eval_det_annos = copy.deepcopy(det_pkl)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        #eval_gt_annos = [copy.deepcopy(info['annos']) for info in gt_pkl]
        result,str=kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
        print(result)