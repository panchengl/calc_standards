# coding: utf-8

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os

from utils.common_util import parse_anchors
from utils.visulize_util import get_color_table, plot_one_box
import datetime
from models.tf_model_sliming import sliming_yolov3
from utils.common_util import postprocess_doctor_yang, create_directory
import cfg


def tf1_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './'):
    anchors = parse_anchors(cfg.anchor_path)
    color_table = get_color_table(cfg.class_num)
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, cfg.img_size[1], cfg.img_size[0], 3], name='input_data')
        yolo_model = sliming_yolov3(cfg.class_num, anchors)
        with tf.variable_scope('yolov3'):
            boxes = yolo_model.forward_include_res_with_prune_factor_docktor_yang( input_data, 0.8, prune_cnt=cfg.prune_cnt)
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/pred_boxes_last"])
        # with tf.gfile.FastGFile("./pb_model/dianli_608_no_prune_20200311_doctor_yang_no_stride_07.pb", mode='wb') as f:
        #     f.write(constant_graph.SerializeToString())
        ############################################################################################
        img_list = os.listdir(img_dirs)
        for m in img_list:
            print(m)
            txt_name = m.split('.')[0]
            txt_file =  open(cfg.single_img_result_dir+ txt_name, 'w')
            img_dir = os.path.join(cfg.img_dir, m)
            img_ori = cv2.imread(img_dir)
            img = cv2.resize(img_ori, tuple((INPUTSIZE, INPUTSIZE)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            starttime = datetime.datetime.now()
            boxes_ = sess.run(boxes, feed_dict={input_data: img})
            endtime = datetime.datetime.now()
            last_result = postprocess_doctor_yang(boxes_, cfg.img_size[0], img_ori.shape[:2], conf_th, cfg.tf_nms_th)
            print("sess cost time is ", endtime - starttime)
            print("boxes_, scores_, labels is ", last_result)
            print("box coords:")
            print('*' * 30)
            for i, box in enumerate(last_result):
                x0, y0, x1, y1, score, label  = box
                src = classes[int(label)] + " " + str(round(score, 2)) + " " + str(int(x0)) + " " + str(int(y0)) + " " + str(int(x1)) + " " +str(int(y1))
                if i != len(last_result) - 1:
                    src += '\n'
                txt_file.write(src )
                print("src is", src)
                plot_one_box(img_ori, [x0, y0, x1, y1],  label=classes[int(label)] + ', {:.2f}%'.format(score * 100),
                             color=color_table[int(label)])
            if via_flag == True:
                cv2.namedWindow('Detection result', 0)
                cv2.resizeWindow('Detection result', 800, 800)
                cv2.imshow('Detection result', img_ori)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_flag == True:
                cv2.imwrite(save_dir + m, img_ori)
        return last_result

def tf1_model_init(sess, model_dir, INPUTSIZE):
    anchors = parse_anchors(cfg.anchor_path)
    input_data = tf.placeholder(tf.float32, [1, INPUTSIZE, INPUTSIZE, 3], name='input_data')
    yolo_model = sliming_yolov3(cfg.class_num, anchors)
    with tf.variable_scope('yolov3'):
        boxes = yolo_model.forward_include_res_with_prune_factor_docktor_yang(input_data, 0.8,
                                                                              prune_cnt=cfg.prune_cnt)
    saver = tf.train.Saver()
    saver.restore(sess, model_dir)
    return boxes, input_data

def tf1_inference_single_img(sess, boxes, input_data,  img_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './'):
    color_table = get_color_table(cfg.class_num)
    img_ori = cv2.imread(img_dir)
    img = cv2.resize(img_ori, tuple((INPUTSIZE, INPUTSIZE)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    starttime = datetime.datetime.now()
    boxes_ = sess.run(boxes, feed_dict={input_data: img})
    endtime = datetime.datetime.now()
    last_result = postprocess_doctor_yang(boxes_, INPUTSIZE, img_ori.shape[:2], conf_th, cfg.tf_nms_th)
    print("sess cost time is ", endtime - starttime)
    print("boxes_, scores_, labels is ", last_result)
    print("box coords:")
    print('*' * 30)
    for i, box in enumerate(last_result):
        x0, y0, x1, y1, score, label = box
        plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[int(label)] + ', {:.2f}%'.format(score * 100),
                     color=color_table[int(label)])
    if via_flag == True:
        cv2.namedWindow('Detection result', 0)
        cv2.resizeWindow('Detection result', 800, 800)
        cv2.imshow('Detection result', img_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # if save_flag == True:
    #     cv2.imwrite(save_dir + m, img_ori)
    return last_result


def tf1_inference_write_results(img_dirs, model_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './'):
    anchors = parse_anchors(cfg.anchor_path)
    color_table = get_color_table(cfg.class_num)
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, cfg.img_size[1], cfg.img_size[0], 3], name='input_data')
        yolo_model = sliming_yolov3(cfg.class_num, anchors)
        with tf.variable_scope('yolov3'):
            boxes = yolo_model.forward_include_res_with_prune_factor_docktor_yang( input_data, 0.8, prune_cnt=cfg.prune_cnt)
        saver = tf.train.Saver()
        saver.restore(sess, model_dir)
        img_list = os.listdir(img_dirs)
        for m in img_list:
            print(m)
            txt_name = m.split('.')[0]
            txt_file =  open(cfg.single_img_result_dir+ txt_name, 'w')
            img_dir = os.path.join(cfg.img_dir, m)
            img_ori = cv2.imread(img_dir)
            img = cv2.resize(img_ori, tuple((INPUTSIZE, INPUTSIZE)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.
            starttime = datetime.datetime.now()
            boxes_ = sess.run(boxes, feed_dict={input_data: img})
            endtime = datetime.datetime.now()
            last_result = postprocess_doctor_yang(boxes_, cfg.img_size[0], img_ori.shape[:2], conf_th, cfg.tf_nms_th)
            print("sess cost time is ", endtime - starttime)
            print("boxes_, scores_, labels is ", last_result)
            print("box coords:")
            print('*' * 30)
            for i, box in enumerate(last_result):
                x0, y0, x1, y1, score, label  = box
                src = classes[int(label)] + " " + str(round(score, 2)) + " " + str(int(x0)) + " " + str(int(y0)) + " " + str(int(x1)) + " " +str(int(y1))
                if i != len(last_result) - 1:
                    src += '\n'
                txt_file.write(src )
                print("src is", src)
                plot_one_box(img_ori, [x0, y0, x1, y1],  label=classes[int(label)] + ', {:.2f}%'.format(score * 100),
                             color=color_table[int(label)])
            if via_flag == True:
                cv2.namedWindow('Detection result', 0)
                cv2.resizeWindow('Detection result', 800, 800)
                cv2.imshow('Detection result', img_ori)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_flag == True:
                cv2.imwrite(save_dir + m, img_ori)
        return last_result



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # results = tf1_inference_write_results(cfg.img_dir, cfg.tf_model_dir, cfg.img_size[0], cfg.names_class, cfg.score_th, cfg.tf_via_flag, cfg.tf_save_flag, cfg.tf_write_img_dir)
    with tf.Session() as sess:
        boxes, input_data = tf1_model_init(sess, cfg.tf_model_dir, cfg.img_size[0])
        results = tf1_inference_single_img(sess, boxes, input_data, img_dir="/home/pcl/tf_work/map/data/image/val00000.jpg", INPUTSIZE=cfg.img_size[0], classes=cfg.names_class, conf_th = 0.3,  via_flag = True)