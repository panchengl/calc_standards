from tqdm import trange
from utils.mnn_util import inference_mnn
from utils.eval_util import parse_gt_rec,  voc_eval, get_preds_gpu_tf1, AverageMeter, get_preds_gpu_mnn, get_preds_gpu_darknet, get_preds_gpu_centernet
from utils.tf_util import tf1_model_init
from utils.dark_util import darknet_model_init
# from utils.centernet_utils import
from centernet.detector_factory import detector_factory

# from
import cfg
import tensorflow as tf

eval_threshold = cfg.score_th
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

if __name__ =="__main__":
    assert cfg.inference_type in ["tf1_yolov3", "tf2_yolov3", "darknet_yolov3", "pytorch1_yolov3", "pytorch2_yolov3", "keras_yolov3", "tf1_yolov3_mnn", 'centernet']

    val_img_cnt = len(open(cfg.val_file, 'r').readlines())
    img_id = []
    img_name = []
    img_scale = []
    for line in open(cfg.val_file, 'r').readlines():
        # print(line)
        img_id.append(line.split(' ')[0])
        img_name.append(line.split(' ')[1])
        img_scale.append([float(int(line.split(' ')[2]) / cfg.img_size[0]), float(int(line.split(' ')[3]) / cfg.img_size[0])])
    val_preds = []
    if cfg.inference_type == "tf1_yolov3":
        print("INFO: model type is tensorflow yolov3")
        with tf.Session() as sess:
            boxes, input_data = tf1_model_init(sess, cfg.tf_model_dir, cfg.img_size[0])
            for j in trange(val_img_cnt):
                pred_content = get_preds_gpu_tf1(sess, boxes, input_data , img_id, img_name, img_scale, j, model_dir=cfg.mnn_model_dir, score_th=eval_threshold)
                val_preds.extend(pred_content)
    elif cfg.inference_type == "tf1_yolov3_mnn":
        print("INFO: current mnn just support inference and model init at the same time")
        print("INFO: model type is mnn_tf yolov3")
        for j in trange(val_img_cnt):
            pred_content = get_preds_gpu_mnn(img_id, img_name, img_scale, j, model_dir=cfg.mnn_model_dir, score_th=eval_threshold)
            val_preds.extend(pred_content)
    elif cfg.inference_type == "darknet_yolov3":
        print("INFO: model type is darknet yolov3")
        model = darknet_model_init(cfg.darknet_weights, 416, cfg.darknet_cfg_file)
        for j in trange(val_img_cnt):
            pred_content = get_preds_gpu_darknet(model, img_id, img_name, img_scale, j, model_dir=cfg.mnn_model_dir, score_th=eval_threshold)
    elif cfg.inference_type == "pytorch_yolov3":
        print("INFO: model type is pytorch yolov3")
        model = darknet_model_init(cfg.torch_model, 416, cfg.darknet_cfg_file)
        for j in trange(val_img_cnt):
            pred_content = get_preds_gpu_darknet(model, img_id, img_name, img_scale, j, model_dir=cfg.mnn_model_dir, score_th=eval_threshold)
            val_preds.extend(pred_content)
    elif cfg.inference_type == "centernet":
        print("INFO: model type is pytorch centernet")
        model = detector_factory['ctdet']()
        print(model)
        for j in trange(val_img_cnt):
            pred_content =get_preds_gpu_centernet(model, img_id, img_name, j,  img_scale)
            val_preds.extend(pred_content)
    else:
        print("current version only support tf1_mnn, pytorch, darknet, pure_tf1, i will finished code after sometimes")
        raise NameError
    # # calc mAP
    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
    gt_dict = parse_gt_rec(cfg.val_file, cfg.img_size, cfg.letterbox_resize)
    print("gt dict is", gt_dict)
    print("val_preds is", val_preds)
    info = ""
    for ii in range(cfg.class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=eval_threshold, use_07_metric=cfg.use_voc_07_metric)
        info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
    mAP = ap_total.average
    info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
    print(info)