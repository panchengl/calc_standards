from __future__ import division, print_function
from utils.common_util import read_class_names, get_classes_standard_dict

import numpy as np

#tf1_yolov3: https://github.com/wizyoung/YOLOv3_TensorFlow
#tf2_yolov3: https://github.com/ShuiXianhua/YOLO_V3
#darknet_yolov3:  must use cfg file and weights filecd ..
#pytorch1_yolov3: https://github.com/ultralytics/yolov3
#pytorch2_yolov3: https://github.com/search?q=yolov3&type=Repositories
#keras_yolov3:  https://github.com/qqwweee/keras-yolo3
inference_type = "centernet"# only support darknet yolov3, tf2_yolov3_mnn, keras_yolov3, pytorch_yolov3, pytorch_centernet

# common params
root_data_dir = "./data"
names_file = './data/coco.names'
img_dir = "./data/image/"
val_file = './data/aa.txt'

names_class = read_class_names(names_file)
class_dict = get_classes_standard_dict(names_class)
class_num = len(names_class)
print("class is", class_num)


single_img_result_dir = "./data/result_txt/"
transform_result_dir = "./data/result_class/"
# dic = {'DiaoChe':0, 'TaDiao':1, 'TuiTuJi':2, 'BengChe': 3, 'WaJueJi':4, 'ChanChe':5}

img_size = [512, 512]
score_th = 0.01
via_th = 0.01
standard_max_iou = 0.1

use_voc_07_metric = False
letterbox_resize = False
single_infere_img_via_flag = False
single_infere_img_save_flag = False
single_infere_img_save_dir = "./"

# mnn param
mnn_model_dir = "./dianli_608_three_label_third_prune_yang.mnn"
mnn_nms_th = 0.5

# tf1_yolov3 param
anchor_path = "./data/yolo_anchors.txt"
tf_model_dir = "./dianli_608//best_model_Epoch_3_step_17895_mAP_0.5809_loss_3.4707_lr_0.0001"
tf_write_img_dir = './data/tf_visual_results_service/'
tf_nms_th = 0.5
prune_cnt = 5
tf_via_flag = False
tf_save_flag = False

#darknet params
darknet_cfg_file = "./weights/yolov3.cfg"
darknet_weights = "./weights/yolov3.weights"
darknet_write_img_dir = './data/darknet_visual_results_service/'
darknet_via_flag = False
darknet_save_flag = False
darknet_nms_th = 0.5

#pytorch params
torch_model = "./yolov3.pt"



#centernet params
flip_test = False
cat_spec_wh = False
K = 100
nms = False
down_ratio = 4
center_thresh = 0.01
pause = False
gpus = [4,5]

arch = 'dla_34'
# arch = 'hourglass'
heads = {'hm': class_num, 'wh': 2 , 'reg': 2}
head_conv = 256 if 'dla' in arch else 64
load_model ='./weights/model_best_dla34.pth'
# load_model ='./weights/model_best_hg.pth'
ori_mean = np.array([0.40789654, 0.44719302, 0.47026115],
                dtype=np.float32).reshape(1, 1, 3)
ori_std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
mean = np.array(ori_mean, dtype=np.float32).reshape(1, 1, 3)
std = np.array(ori_std, dtype=np.float32).reshape(1, 1, 3)

test_scales = [1.0]
fix_res = False
pad = 127 if 'hourglass' in arch else 31
num_stacks = 2 if arch == 'hourglass' else 1
dataset = 'dianli'
debugger_theme = 'white'
debug = 0
