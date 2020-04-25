from __future__ import print_function

from utils.mnn_util import mnn_inference_write_results
from utils.tf_util import tf1_inference_write_results
from utils.dark_util import darknet_inference_write_results
from utils.common_util import create_directory
import cfg


def inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th,  via_flag, save_flag , save_dir):
    model_type = model_dir.split(".")[-1]
    assert create_directory(cfg.single_img_result_dir)
    if model_type == "mnn":
        print("current model type is mnn")
        return mnn_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th,  via_flag, save_flag , save_dir)
    elif model_type == "ckpt" or model_type == "0001":
        print("current model type is tf1")
        return tf1_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th,  via_flag, save_flag , save_dir)
    elif model_type == "weights":
        return darknet_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th,  via_flag, save_flag , save_dir, cfg.darknet_cfg_file)
    else:
        raise ValueError("type not support")


