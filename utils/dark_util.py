from __future__ import division

from models.darknet_model import Darknet
from utils.common_util import load_classes, non_max_suppression, rescale_boxes, pad_to_square, resize, create_directory
from PIL import Image
from  utils.visulize_util import plot_one_box, get_color_table
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import time
import argparse
import cv2
import torch
import cfg
import numpy as np

def darknet_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './', cfg_file='./models/yolov3.cfg'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg_file, img_size=INPUTSIZE).to(device)
    if model_dir.endswith(".weights"):
        model.load_darknet_weights(model_dir)
    else:
        model.load_state_dict(torch.load(model_dir))
    model.eval()
    file_list = os.listdir(img_dirs)
    color_table = get_color_table(80)
    for img_name in file_list:
        txt_name = img_name.split('.')[0]
        img_paths = os.path.join(img_dirs, img_name)
        assert create_directory(cfg.single_img_result_dir)
        txt_file = open(cfg.single_img_result_dir + txt_name, 'w')
        img_ori = cv2.imread(img_paths)
        img = transforms.ToTensor()(Image.open(img_paths))
        img, _ = pad_to_square(img, 0)
        input_imgs = resize(img, INPUTSIZE)
        input_imgs = Variable(torch.unsqueeze(input_imgs, dim=0).float(), requires_grad=False).cuda()
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_th, cfg.darknet_nms_th)
            if detections[0] is not None:
                detections = rescale_boxes(detections[0], INPUTSIZE, img_ori.shape[:2])
                for i, (x0, y0, x1, y1, conf, cls_conf, label) in enumerate(detections.numpy()):
                    score = cls_conf*conf
                    if score >= conf_th:
                        src = classes[int(label)] + " " + str(round(score, 2)) + " " + str(int(x0)) + " " + str(int(y0)) + " " + str(int(x1)) + " " + str(int(y1))
                        if i != len(detections) - 1:
                            src += '\n'
                        txt_file.write(src)
                        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[int(label)] + ', {:.2f}%'.format(score * 100),
                                     color=color_table[int(label)])
                if via_flag == True:
                    cv2.namedWindow('Detection result', 0)
                    cv2.resizeWindow('Detection result', 800, 800)
                    cv2.imshow('Detection result', img_ori)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                if save_flag == True:
                    cv2.imwrite(save_dir + img_name, img_ori)
            else:
                print("current img detect no obj ")
                print("img name is", img_name)
    return 1

def darknet_model_init(model_dir, INPUTSIZE, cfg_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(cfg_file, img_size=INPUTSIZE).to(device)
    if model_dir.endswith(".weights"):
        model.load_darknet_weights(model_dir)
    else:
        model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model

def darknet_inference_single_img(model,  img_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True):
    color_table = get_color_table(80)
    img_ori = cv2.imread(img_dir)
    img = transforms.ToTensor()(Image.open(img_dir))
    img, _ = pad_to_square(img, 0)
    input_imgs = resize(img, INPUTSIZE)
    input_imgs = Variable(torch.unsqueeze(input_imgs, dim=0).float(), requires_grad=False).cuda()
    with torch.no_grad():
        results = []
        detections = model(input_imgs)
        detections = non_max_suppression(detections, conf_th, cfg.darknet_nms_th)
        if detections[0] is not None:
            detections = rescale_boxes(detections[0], INPUTSIZE, img_ori.shape[:2])
            for i, (x0, y0, x1, y1, conf, cls_conf, label) in enumerate(detections.numpy()):
                score = cls_conf * conf
                if score >= conf_th:
                    results.append( np.array( [x0, y0, x1, y1, score, label] ) )
                    plot_one_box(img_ori, [x0, y0, x1, y1],
                                 label=classes[int(label)] + ', {:.2f}%'.format(score * 100),
                                 color=color_table[int(label)])
            if via_flag == True:
                cv2.namedWindow('Detection result', 0)
                cv2.resizeWindow('Detection result', 800, 800)
                cv2.imshow('Detection result', img_ori)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("current img detect no obj ")
            print("img name is", img_dir)
    return results

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # darknet_inference_write_results(cfg.img_dir, cfg.darknet_weights, cfg.img_size[0], cfg.names_class, cfg.score_th, cfg.darknet_via_flag, cfg.darknet_save_flag, cfg.darknet_write_img_dir, cfg.darknet_cfg_file)
    model = darknet_model_init(cfg.darknet_weights, 416, cfg.darknet_cfg_file)
    results = darknet_inference_single_img(model, "/home/pcl/tf_work/map/data/image_shandong/val00002.jpg", 416, cfg.names_class, cfg.score_th, cfg.darknet_via_flag)
    print(results)