# encoding=utf-8
import os
from utils.static_util import transform_data
from utils.standard_util import calculate_performance_index,calculate_performance_index_3label
from inference.inference import inference_write_results
import cfg


if __name__ == "__main__":
    # inference_write_results(img_dirs=cfg.img_dir, model_dir=cfg.mnn_model_dir, INPUTSIZE=cfg.img_size[0], conf_th = cfg.score_th,
    #                         classes=cfg.names_class, via_flag = cfg.single_infere_img_via_flag, save_flag = cfg.single_infere_img_save_flag, save_dir=cfg.single_infere_img_save_dir)
    transform_data(result_dir=cfg.single_img_result_dir, transform_result_dir=cfg.transform_result_dir, classes=cfg.names_class)
    print(cfg.class_dict)
    calculate_performance_index_3label(cfg.root_data_dir, transform_result_dir=cfg.transform_result_dir, dict_class=cfg.class_dict)



