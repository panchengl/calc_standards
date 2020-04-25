import cv2
import os
import random
import colorsys
import numpy as np

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    # tl = 2
    # print(tl)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

def vis_all_results_img(txt_result_dir, ori_img_dir,  classes, via_flag = True, save_flag = False):
    '''
    txt_result_dir: inference result dir, [class conf xmin ymin xmax ymax]
    txt:  val0001 val0002 val003 ...... not  val0001.txt val00002.txt val0003.txt
    ori_img_dir: img dir
    classes: [classes1, classes2,...]
    vis_flag: sure visuliaze
    save flag: sure save img
    '''
    anno_file = sorted(os.listdir(txt_result_dir))
    img_file = sorted(os.listdir(ori_img_dir))
    color_table = get_color_table(len(classes))
    for id, anno in enumerate(anno_file):
        img = cv2.imread(os.path.join(img_dir, img_file[id]) )
        print(img_file[id])
        lines = open(os.path.join(anno_dir, anno)).readlines()
        for line in lines:
            result = line.split(' ')
            class_name, conf, xmin, ymin, xmax, ymax = result[0], result[1], result[2], result[3], result[4], result[5]
            plot_one_box(img, [xmin, ymin, xmax, ymax], label=class_name + ', {:.2f}%'.format(float(conf) * 100), color=color_table[classes.index(class_name)])
        if via_flag == True:
            cv2.namedWindow('result', 0)
            cv2.resizeWindow('result', 800, 800)
            cv2.imshow("result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_flag == True:
            cv2.imwrite(os.path.join(write_img_path, img_file[id]), img)

def draw_bbox(original_image, bboxes, classes):
    """
    :param original_image: 检测的原始图片，shape为(org_h, org_w, 3)
    :param bboxes: shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: None
    """
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    image_h, image_w, _ = original_image.shape
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(1.0 * (image_h + image_w) / 600)
        cv2.rectangle(original_image, (coor[0], coor[1]), (coor[2], coor[3]), bbox_color, bbox_thick)

        bbox_mess = '%s: %.3f' % (classes[class_ind], score)
        text_loc = (int(coor[0]), int(coor[1] + 5) if coor[1] < 20 else int(coor[1] - 5))
        cv2.putText(original_image, bbox_mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h, (255, 255, 255), bbox_thick // 3)
    return original_image

if __name__ == "__main__":
    anno_dir = "../data/txt_0319"
    img_dir = "../data/image"
    write_img_path = '/home/pcl/tf_work/map/data/visual_results_service'
    classes = ['DiaoChe', 'TaDiao', 'TuiTuJi', 'BengChe', 'WaJueJi', 'ChanChe']
    vis_all_results_img(anno_dir, img_dir, classes, True, True)


