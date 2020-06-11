# encoding=utf-8
import os
from utils.static_util import transform_data
from utils.standard_util import calculate_performance_index,calculate_performance_index_3label
from inference.inference import inference_write_results
import cfg
from tqdm import trange
from utils.static_util import get_txt_inference_single_img
from utils.eval_util import parse_gt_rec,  parse_gt_rec_ori_scale, voc_eval, get_preds_gpu_tf1, AverageMeter, get_preds_gpu_mnn, get_preds_gpu_darknet, get_preds_gpu_centernet

eval_threshold = cfg.score_th
if __name__ == "__main__":

    # inference_write_results(img_dirs=cfg.img_dir, model_dir=cfg.mnn_model_dir, INPUTSIZE=cfg.img_size[0], conf_th = cfg.score_th,
    #                         classes=cfg.names_class, via_flag = cfg.single_infere_img_via_flag, save_flag = cfg.single_infere_img_save_flag, save_dir=cfg.single_infere_img_save_dir)
    # transform_data(result_dir=cfg.single_img_result_dir, transform_result_dir=cfg.transform_result_dir, classes=cfg.names_class)
    # print(cfg.class_dict)
    # calculate_performance_index_3label(cfg.root_data_dir, transform_result_dir=cfg.transform_result_dir, dict_class=cfg.class_dict)

    assert cfg.inference_type in ["tf1_yolov3", "tf2_yolov3", "darknet_yolov3", "pytorch1_yolov3", "pytorch2_yolov3",
                                  "keras_yolov3", "tf1_yolov3_mnn", 'centernet', 'txt']
    cfg.inference_type = 'txt'
    val_img_cnt = len(open(cfg.val_file, 'r').readlines())
    img_id = []
    img_name = []
    img_scale = []
    img_sum = 0
    obj_img_sum = 0
    img_report_dict_gt = {}
    img_report_dict_algorithm = {}
    for line in open(cfg.val_file, 'r').readlines():
        # print(line)
        img_id.append(line.split(' ')[0])
        img_name.append(line.split(' ')[1])
        # print(line.split(' ')[1].split('/')[-1])
        img_scale.append([float(int(line.split(' ')[2]) / cfg.img_size[0]), float(int(line.split(' ')[3]) / cfg.img_size[0])])
        img_sum += 1
        img_report_dict_gt[line.split(' ')[0]] = 0
        if (len(line.split(' '))) > 4:
            obj_img_sum += 1
            img_report_dict_gt[line.split(' ')[0]] = 1

    print(img_sum)
    print(obj_img_sum)

    val_preds = []
    print(img_name)
    print(img_id)
    detections = 0
    if cfg.inference_type == 'txt':
        for j in trange(val_img_cnt):
            pred_content = get_txt_inference_single_img(img_id, cfg.txt_data_dir,img_scale,  img_name, j, classes=cfg.names_class, conf_th =cfg.score_th, use_orignal_scale=True)
            val_preds.extend(pred_content)
            if len(pred_content)>0:
                img_report_dict_algorithm[img_id[j]] = 1
            else:
                img_report_dict_algorithm[img_id[j]] = 0
            detections += len(pred_content)
        # # calc mAP
    rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
        # gt_dict = parse_gt_rec(cfg.val_file, cfg.img_size, cfg.letterbox_resize)
    gt_dict = parse_gt_rec_ori_scale(cfg.val_file, cfg.img_size, cfg.letterbox_resize)
    print("gt dict is", gt_dict)
    print("val_preds is", val_preds)
    info = ""
    for ii in range(cfg.class_num):
        npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=cfg.iou_thr,
                                           use_07_metric=cfg.use_voc_07_metric)
        info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
        rec_total.update(rec, npos)
        prec_total.update(prec, nd)
        ap_total.update(ap, 1)
    mAP = ap_total.average
    info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average,
                                                                           mAP)
    print(info)


    ### calc standdards
    # img_sums = len()
    print(img_report_dict_gt)
    print(img_report_dict_algorithm)
    # print(len(img_report_dict_gt))
    # print(len(img_report_dict_algorithm))

    true_report_imgs = 0
    false_report_imgs = 0
    missing_report_imgs = 0

    for i ,(txt_line_id, value) in enumerate(img_report_dict_gt.items()):
        # if img_report_dict_algorithm[txt_line_id] == img_report_dict_algorithm[txt_line_id] and img_report_dict_algorithm[txt_line_id] == 1:
        if img_report_dict_gt[txt_line_id] == 1 and img_report_dict_algorithm[txt_line_id] == 1:
            true_report_imgs += 1
        elif img_report_dict_gt[txt_line_id] == 0 and img_report_dict_algorithm[txt_line_id] == 1:
            false_report_imgs += 1
        elif img_report_dict_gt[txt_line_id] == 1 and img_report_dict_algorithm[txt_line_id] == 0:
            missing_report_imgs += 1
        elif img_report_dict_gt[txt_line_id] == 0 and img_report_dict_algorithm[txt_line_id] == 0:
            continue
    true_report_rate = float(true_report_imgs)/obj_img_sum
    false_report_rate = float(false_report_imgs)/(len(img_report_dict_gt) - obj_img_sum + 1)
    missing_report_rate = float(missing_report_imgs)/obj_img_sum

    print("all have obj img sum is: ", obj_img_sum)
    print("all img sum is: ", len(img_report_dict_gt))

    print("true_report_imgs is:", true_report_imgs)
    print("false_report_imgs is:",false_report_imgs)
    print("missing_report_imgs is:",missing_report_imgs)

    print("true_report_rate is: ",true_report_rate )
    print("false_report_rate is: ", false_report_rate)
    print("missing_report_rate is: ", missing_report_rate)

    print("detections sum is: ", detections)




