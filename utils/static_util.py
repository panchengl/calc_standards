# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from utils.common_util import create_directory
import cv2
from utils.visulize_util import get_color_table, plot_one_box


def static_class_num(anno_dir, class_id_dict):
    class_num_dict = {}
    for root, folders, files in os.walk(anno_dir):
        for f in files:
            if (f.find(".xml") != -1):
                in_file = open(root + "/" + f)
                tree = ET.parse(in_file)
                r = tree.getroot()
                for obj in r.iter('object'):
                    cls = obj.find('name').text
                    if cls in class_id_dict.keys():
                        if cls in ["ShanHuo", "YanWu", "SuLiaoBu"]:
                            print(f)
                        class_id_dict[cls] = class_id_dict[cls] + 1
                in_file.close()
    for key in class_id_dict:
        class_num_dict[key] = class_id_dict[key]
    return class_num_dict


def check_img_anno(img_dir, anno_dir):
    print("img_dir has imgs: ", len(os.listdir(img_dir)))
    print("anno_dir has xmls: ", len(os.listdir(anno_dir)))

    imgs = os.listdir(img_dir)
    annos = os.listdir(anno_dir)
    ann_names = []
    img_names = []
    for ann in annos:
        name = ann.split('.')[0]
        ann_names.append(name)
    for im in imgs:
        a = im.split('.')[0]
        img_names.append(a)
    for b in img_names:
        if b not in ann_names:
            print("this file has jpg, but no xml: ", b)
    for b in ann_names:
        if b not in img_names:
            print("this file has xml, but no jpg: ", b)


def transform_data(result_dir, transform_result_dir, classes):
    assert create_directory(transform_result_dir)
    files = os.listdir(result_dir)
    files.sort()
    for name in classes:
        filename = os.path.join(transform_result_dir, name + ".txt")
        with open(filename, 'w') as f:
            for file in files:
                with open(os.path.join(result_dir, file), 'r') as ori:
                    lines = ori.readlines()
                    for line in lines:
                        if line.startswith(name):
                            line = line.replace(name, file)
                            if "\n" not in line:
                                line = line + '\n'
                            f.writelines(line)
    print("transform finished")


def get_txt_inference_single_img(img_id, txt_dir, scale_list, img_name, number, classes, conf_th=0.3, via_flag=False, use_orignal_scale=True):
    color_table = get_color_table(len(classes))
    output_box = []
    #### shandong txt code ################
    # current_result_dir = os.path.join(txt_dir, img_name[number].split('/')[-1].replace('.jpg', ''))
    # assert os.path.isfile(current_result_dir)
    ########################################

    #### v5 yolo txt code ###################
    # current_result_dir = os.path.join(txt_dir, img_name[number].split('/')[-2], img_name[number].split('/')[-1].replace('.jpg', '.txt'))
    # assert os.path.isfile(current_result_dir)
    #########################################

    #### v5 ssd txt code #####################
    current_result_dir = os.path.join(txt_dir, img_name[number].split('/')[-2],img_name[number].split('/')[-1].replace('.jpg', '.txt'))
    if not os.path.isfile(current_result_dir):
        return output_box
    ##########################################

    scale_w = scale_list[number][0]
    scale_h = scale_list[number][1]
    shigongjixie = ["TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    all_obj_list = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "ShiGongJiXie"]
    with open(current_result_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result = line.split(' ')
            if result[0] in all_obj_list:
                # if result[0] == "DiaoChe" or result[0] == "ShiGongJiXie":
                #     conf_th = 0.4
                if float(result[1]) >= conf_th:
                    true_label = result[0]
                    if result[0] in shigongjixie:
                        print("id is %s, obj is %s, transfer ShiGongJiXie"%(img_name[number], result[0]))
                        true_label = "ShiGongJiXie"
                        print(result[0])
                    x0_ori, y0_ori, x1_ori, y1_ori, score, label = float(result[2]), float(result[3]), float(result[4]), float(result[5]), float(result[1]), classes.index(true_label)
                    x0, y0, x1, y1, score, label = float(result[2])/ float(scale_w) , float(result[3])/float(scale_h), float(result[4])/float(scale_w), float(result[5])/float(scale_h), float(result[1]), classes.index(true_label)
                    if use_orignal_scale:
                        box = [int(img_id[number]), x0_ori, y0_ori, x1_ori, y1_ori, score, label]
                    else:
                        box = [int(img_id[number]), x0, y0, x1, y1, score, label]
                    if via_flag == True:
                        img_ori = cv2.imread(os.path.join(img_name[number]))
                        plot_one_box(img_ori, [x0_ori, y0_ori, x1_ori, y1_ori], label=result[0] + ', {:.2f}%'.format(score * 100),
                                     color=color_table[int(label)])
                    output_box.append(box)
    if via_flag == True:
        cv2.namedWindow('Detection result', 0)
        cv2.resizeWindow('Detection result', 800, 800)
        cv2.imshow('Detection result', img_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return output_box


if __name__ == "__main__":
    class_id_dict = {"DiaoChe": 0, "TaDiao": 0, "TuiTuJi": 0, "BengChe": 0, "WaJueJi": 0, "ChanChe": 0, "ShanHuo": 0,
                     "YanWu": 0, "SuLiaoBu": 0}
    names = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    # anno_dir = "/home/pcl/tf_work/map/data/Annotations"
    # anno_dir = "/home/pcl/data/dianli_zhongkeyuan/"
    anno_dir = "/home/pcl/data/VOC2007/Annotations"
    all_dict = static_class_num(anno_dir, class_id_dict)
    print(all_dict)
# transform_data(result_dir="/home/pcl/tf_work/map/data/mnn_txt", transform_result_dir='/home/pcl/tf_work/map/data/mnn_class', classes=names)
