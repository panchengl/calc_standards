from __future__ import print_function

import numpy as np
import MNN
import cv2
import os
from utils.visulize_util import draw_bbox
from utils.common_util import postprocess_doctor_yang
import cfg

def inference_mnn(img_dir, mnn_model_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './'):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(mnn_model_dir)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    img_ori = cv2.imread(img_dir)
    orishape = img_ori.shape
    originimg=img_ori.copy()
    img = cv2.resize(img_ori, tuple([INPUTSIZE, INPUTSIZE]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    image = img[np.newaxis, :] / 255.
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1,INPUTSIZE, INPUTSIZE,3), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Tensorflow)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    output_data=np.array(output_tensor.getData())
    output_data=output_data.reshape((-1,len(classes) + 5))
    print("output data is", output_data)
    outbox = np.array(postprocess_doctor_yang(output_data, INPUTSIZE, orishape[:2], conf_thres=conf_th))
    originimg = draw_bbox(originimg, outbox, classes)
    if via_flag == True:
        cv2.namedWindow('result', 0)
        cv2.resizeWindow('result', 800, 800)
        cv2.imshow("result", originimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_flag == True:
        cv2.imwrite(save_dir + 'result.jpg', originimg)
    # print("mnn inferece finished")
    return outbox

def mnn_inference_write_results(img_dirs, mnn_model_dir, INPUTSIZE, classes, conf_th = 0.3,  via_flag = True, save_flag = True, save_dir = './'):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(mnn_model_dir)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    img_list = os.listdir(img_dirs)
    for m in img_list:
        print(m)
        txt_name = m.split('.')[0]
        txt_file = open(cfg.single_img_result_dir + txt_name, 'w')
        img_dir = os.path.join(img_dirs, m)
        img_ori = cv2.imread(img_dir)
        orishape = img_ori.shape
        originimg = img_ori.copy()
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple([INPUTSIZE, INPUTSIZE]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        image = img[np.newaxis, :] / 255.

        #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
        tmp_input = MNN.Tensor((1,INPUTSIZE, INPUTSIZE,3), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Tensorflow)
        #construct tensor from np.ndarray
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
        output_tensor = interpreter.getSessionOutput(session)
        output_data=np.array(output_tensor.getData())
        output_data=output_data.reshape((-1,len(classes) + 5))
        # print('output data is', output_data)
        outbox = np.array(postprocess_doctor_yang(output_data, INPUTSIZE, orishape[:2], cfg.score_th, cfg.mnn_nms_th))
        print('result box is', outbox)
        for i, box in enumerate(outbox):
            x0, y0, x1, y1, score, label  = box
            src = classes[int(label)].replace('\n', '') + " " + str(round(score, 2)) + " " + str(int(x0)) + " " + str(int(y0)) + " " + str(int(x1)) + " " +str(int(y1))
            if i != len(outbox) - 1:
                src += '\n'
            txt_file.write(src )
        originimg = draw_bbox(originimg, outbox, classes)
        if via_flag == True:
            cv2.namedWindow('result', 0)
            cv2.resizeWindow('result', 800, 800)
            cv2.imshow("result", originimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_flag == True:
            cv2.imwrite(save_dir + m, originimg)
    return outbox
if __name__ == "__main__":
    INPUTSIZE = 608
    CLASSES = []
    num_classes = 3
    with open('/home/pcl/tf_work/map/data/cheliang_3label.names', 'r') as f:
        CLASSES = f.readlines()
    img_dir = '/home/pcl/tf_work/map/data/image_shandong/'
    # model_dir = "/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/mnn/convertor-code/v3/dianli_608_three_label_third_prune_yang.mnn"
    model_dir = "/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/mnn/convertor-code/v3/dianli_608_three_label_third_prune_yang_quan.mnn"
    # result = inference_mnn(img_dir, model_dir, INPUTSIZE, CLASSES, conf_th=0.3)
    result = mnn_inference_write_results(img_dir, model_dir, INPUTSIZE, CLASSES, conf_th=0.3)
    print(result)
