# encoding=utf-8
import xml.etree.ElementTree as ET
import os
import cfg

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles

    for i in range(4):
        rec1[i] = float(rec1[i])
        rec2[i] = float(rec2[i])

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def get_obj_information(root_directory, dict_class):

    image_total_count = 0    #图片总数量
    image_with_danger_count = 0  #带有安全隐患的图片计数
    class_count = [0] * 6  #安全隐患的物体计数
    image_mask = []       #实际安全隐患和预测安全隐患

    total_obj = []  # 所有图片的所有物体信息列表
    files = os.listdir(root_directory)
    files.sort()
    for filename in files:
        tree = ET.parse(root_directory + '/' + filename)
        root = tree.getroot()
        image_total_count += 1
        with_danger = False  # 该张图片是否带有安全隐患
        objects = []  # 一张图片中的信息

        for obj in root.iter('object'):
            obj_struct = {}  # 一张图片中的一个物体的信息
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            obj_struct['used'] = 0  # 用于之后与预测物体进行配对
            objects.append(obj_struct)

            if obj_struct['name'] in dict_class:  #标记物体在安全隐患列表中
                class_count[dict_class[obj_struct['name']]] += 1
                with_danger = True

        if with_danger == False:  # 图片不带有安全隐患
            image_mask.append([0, 0])  #第一个数代表图片没有安全隐患
            #filelist.append(filename) 记录不带有安全隐患的文件名
        else:                # image_shandong with objects
            image_mask.append([1, 0])  #第一个数代表图片有安全隐患
            image_with_danger_count += 1
        total_obj.append(objects)
    return total_obj, class_count, image_total_count, image_with_danger_count, image_mask


def calculate_performance_index(root_directory, transform_result_dir, dict_class):
    total_obj, class_count, image_total_count, image_with_danger_count, image_mask = get_obj_information(root_directory + '/' + 'Annotations/', dict_class)

    true_identification = [0] * 6  #每类正确预测数量
    missing_report_count = 0
    false_report_count = 0
    class_false_report = [0] * 6   #每类预测错误的数量

    for filename in os.listdir(transform_result_dir):
        obj_class = filename[:-4]  #预测物体的类别，i为文件名
        for line in open(transform_result_dir + filename , "r"):
            line = line[:-1].split(' ')
            k = int(line[0][3:])  # 预测第k张图片有obj_class物体
            image_mask[k][1] = 1  # 预测第k张图片有安全隐患
            max_iou = cfg.standard_max_iou
            iter = -1  #最大iou下标
            for j in range(len(total_obj[k])):  #寻找与预测物体同类且有最大iou的标记，下标为iter
                if total_obj[k][j]['name'] == obj_class and total_obj[k][j]['used'] == 0:  #同类且标记没使用
                    iou = compute_iou(total_obj[k][j]['bbox'], line[2:])  #计算IOU
                    if iou > max_iou:
                        max_iou = iou
                        iter = j

            if iter != -1:  #如果找到符合条件的标记
                total_obj[k][iter]['used'] += 1  #第iter个标记已使用，不能再使用
                true_identification[dict_class[obj_class]] +=1  #该类预测正确的数量加一
            else:           #预测没有找到符合条件的标记
                class_false_report[dict_class[obj_class]] +=1   #该类预测错误的数量加一


    pre_image_with_danger = 0    #预测带有安全隐患的图片的数量

    for i in image_mask:
        if i[1] == 1:                     #图片预测带有安全隐患
            pre_image_with_danger += 1
            if i[0] == 0:                 #图片无安全隐患
                false_report_count += 1   #错报图片数量加一
        else:                             #图片预测没有安全隐患
            if i[0] == 1:                    #图片有安全隐患
                missing_report_count += 1     #漏报图片数量加一
    print('False report count: %d'%false_report_count)
    print('Missing report count: %d'%missing_report_count)

    true_identification_rate = [0] * 6   #每类预测准确率
    class_false_report__rate = [0] * 6   #每类误检率
    for i in range(6):
        true_identification_rate[i] = true_identification[i]/(class_count[i]) #计算每类预测准确率
        class_false_report__rate[i] = class_false_report[i]/(class_count[i])  #计算每类误报率
    false_report_rate = 1.0*false_report_count/image_with_danger_count            #计算误报率
    missing_report_rate = 1.0*missing_report_count/image_with_danger_count      #计算漏报率

    print('image_total_count:%d' %image_total_count)
    print('Image with danger count:%d'%image_with_danger_count)
    print('False report rate: %f' %false_report_rate)
    print('Missing report rate: %f' %missing_report_rate)
    print(dict_class)
    print('True identification:', true_identification)
    print('Class count:', class_count)
    print('True identification rate:', true_identification_rate)
    print('class_false_report__rate:', class_false_report__rate)

def get_obj_information_3label(root_directory, dict_class):
    class_sgjx = ["TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    class_cheliang = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    image_total_count = 0    #图片总数量
    image_with_danger_count = 0  #带有安全隐患的图片计数
    class_count = [0] * 3  #安全隐患的物体计数
    image_mask = []       #实际安全隐患和预测安全隐患

    total_obj = []  # 所有图片的所有物体信息列表
    files = os.listdir(root_directory)
    files.sort()
    for filename in files:
        tree = ET.parse(root_directory + '/' + filename)
        root = tree.getroot()
        image_total_count += 1
        with_danger = False  # 该张图片是否带有安全隐患
        objects = []  # 一张图片中的信息

        for obj in root.iter('object'):
            cla_name = obj.find('name').text
            if cla_name not in class_cheliang:
                continue
            obj_struct = {}  # 一张图片中的一个物体的信息
            obj_struct['name'] = "ShiGongJiXie" if cla_name in class_sgjx else cla_name
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            obj_struct['used'] = 0  # 用于之后与预测物体进行配对
            objects.append(obj_struct)
            # print("current name is", obj_struct['name'])
            if obj_struct['name'] in dict_class:  #标记物体在安全隐患列表中
                class_count[dict_class[obj_struct['name']]] += 1
                with_danger = True

        if with_danger == False:  # 图片不带有安全隐患
            image_mask.append([0, 0])  #第一个数代表图片没有安全隐患
            #filelist.append(filename) 记录不带有安全隐患的文件名
        else:                # image_shandong with objects
            image_mask.append([1, 0])  #第一个数代表图片有安全隐患
            image_with_danger_count += 1
        total_obj.append(objects)
    print("total obj is", total_obj)
    print("single obj class is", class_count)
    return total_obj, class_count, image_total_count, image_with_danger_count, image_mask

def get_obj_information_3label(root_directory, dict_class):
    class_sgjx = ["TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    class_cheliang = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
    image_total_count = 0    #图片总数量
    image_with_danger_count = 0  #带有安全隐患的图片计数
    class_count = [0] * 3  #安全隐患的物体计数
    image_mask = []       #实际安全隐患和预测安全隐患

    total_obj = []  # 所有图片的所有物体信息列表
    files = os.listdir(root_directory)
    files.sort()
    for filename in files:
        tree = ET.parse(root_directory + '/' + filename)
        root = tree.getroot()
        image_total_count += 1
        with_danger = False  # 该张图片是否带有安全隐患
        objects = []  # 一张图片中的信息

        for obj in root.iter('object'):
            cla_name = obj.find('name').text
            # if cla_name not in class_cheliang:
            #     continue
            obj_struct = {}  # 一张图片中的一个物体的信息
            # print(cla_name)
            obj_struct['name'] = "ShiGongJiXie" if cla_name in class_sgjx else cla_name
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                  int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text),
                                  int(bbox.find('ymax').text)]
            obj_struct['used'] = 0  # 用于之后与预测物体进行配对
            objects.append(obj_struct)
            # print("current name is", obj_struct['name'])
            if obj_struct['name'] in dict_class:  #标记物体在安全隐患列表中
                class_count[dict_class[obj_struct['name']]] += 1
                with_danger = True

        if with_danger == False:  # 图片不带有安全隐患
            image_mask.append([0, 0])  #第一个数代表图片没有安全隐患
            #filelist.append(filename) 记录不带有安全隐患的文件名
        else:                # image_shandong with objects
            image_mask.append([1, 0])  #第一个数代表图片有安全隐患
            image_with_danger_count += 1
        total_obj.append(objects)
    return total_obj, class_count, image_total_count, image_with_danger_count, image_mask

def calculate_performance_index_3label(root_directory, transform_result_dir, dict_class):
    total_obj, class_count, image_total_count, image_with_danger_count, image_mask = get_obj_information_3label(root_directory + '/' + 'Annotations/', dict_class)

    true_identification = [0] * len(dict_class)  #每类正确预测数量
    missing_report_count = 0
    false_report_count = 0
    class_false_report = [0] * len(dict_class)   #每类预测错误的数量

    for filename in os.listdir(transform_result_dir):
        print(filename)
        obj_class = filename[:-4]  #预测物体的类别，i为文件名
        print(obj_class)
        for line in open(transform_result_dir + filename , "r"):
            line = line[:-1].split(' ')
            k = int(line[0][3:])  # 预测第k张图片有obj_class物体
            image_mask[k][1] = 1  # 预测第k张图片有安全隐患
            max_iou = cfg.standard_max_iou
            iter = -1  #最大iou下标
            for j in range(len(total_obj[k])):  #寻找与预测物体同类且有最大iou的标记，下标为iter
                if total_obj[k][j]['name'] == obj_class and total_obj[k][j]['used'] == 0:  #同类且标记没使用
                    iou = compute_iou(total_obj[k][j]['bbox'], line[2:])  #计算IOU
                    if iou > max_iou:
                        max_iou = iou
                        iter = j

            if iter != -1:  #如果找到符合条件的标记
                total_obj[k][iter]['used'] += 1  #第iter个标记已使用，不能再使用
                true_identification[dict_class[obj_class]] +=1  #该类预测正确的数量加一
            else:           #预测没有找到符合条件的标记
                class_false_report[dict_class[obj_class]] +=1   #该类预测错误的数量加一


    pre_image_with_danger = 0    #预测带有安全隐患的图片的数量

    for i in image_mask:
        if i[1] == 1:                     #图片预测带有安全隐患
            pre_image_with_danger += 1
            if i[0] == 0:                 #图片无安全隐患
                false_report_count += 1   #错报图片数量加一
        else:                             #图片预测没有安全隐患
            if i[0] == 1:                    #图片有安全隐患
                missing_report_count += 1     #漏报图片数量加一
    print('False report count: %d'%false_report_count)
    print('Missing report count: %d'%missing_report_count)

    true_identification_rate = [0] * len(dict_class)   #每类预测准确率
    class_false_report__rate = [0] * len(dict_class)    #每类误检率
    for i in range(len(dict_class) ):
        true_identification_rate[i] = true_identification[i]/(class_count[i]) #计算每类预测准确率
        class_false_report__rate[i] = class_false_report[i]/(class_count[i])  #计算每类误报率
    false_report_rate = 1.0*false_report_count/image_with_danger_count            #计算误报率
    missing_report_rate = 1.0*missing_report_count/image_with_danger_count      #计算漏报率

    print('image_total_count:%d' %image_total_count)
    print('Image with danger count:%d'%image_with_danger_count)
    print('False report rate: %f' %false_report_rate)
    print('Missing report rate: %f' %missing_report_rate)
    print(dict_class)
    print('True identification:', true_identification)
    print('Class count:', class_count)
    print('True identification rate:', true_identification_rate)
    print('class_false_report__rate:', class_false_report__rate)


def calculate_performance_index_3label_pcl(root_directory, transform_result_dir, dict_class):
    total_obj, class_count, image_total_count, image_with_danger_count, image_mask = get_obj_information_3label(root_directory + '/' + 'Annotations/', dict_class)

    true_identification = [0] * len(dict_class)  #每类正确预测数量
    missing_report_count = 0
    false_report_count = 0
    class_false_report = [0] * len(dict_class)   #每类预测错误的数量

    for filename in os.listdir(transform_result_dir):
        print(filename)
        obj_class = filename[:-4]  #预测物体的类别，i为文件名
        print(obj_class)
        for line in open(transform_result_dir + filename , "r"):
            line = line[:-1].split(' ')
            k = int(line[0][3:])  # 预测第k张图片有obj_class物体
            image_mask[k][1] = 1  # 预测第k张图片有安全隐患
            max_iou = cfg.standard_max_iou
            iter = -1  #最大iou下标
            for j in range(len(total_obj[k])):  #寻找与预测物体同类且有最大iou的标记，下标为iter
                if total_obj[k][j]['name'] == obj_class and total_obj[k][j]['used'] == 0:  #同类且标记没使用
                    iou = compute_iou(total_obj[k][j]['bbox'], line[2:])  #计算IOU
                    if iou > max_iou:
                        max_iou = iou
                        iter = j

            if iter != -1:  #如果找到符合条件的标记
                total_obj[k][iter]['used'] += 1  #第iter个标记已使用，不能再使用
                true_identification[dict_class[obj_class]] +=1  #该类预测正确的数量加一
            else:           #预测没有找到符合条件的标记
                class_false_report[dict_class[obj_class]] +=1   #该类预测错误的数量加一


    pre_image_with_danger = 0    #预测带有安全隐患的图片的数量

    for i in image_mask:
        if i[1] == 1:                     #图片预测带有安全隐患
            pre_image_with_danger += 1
            if i[0] == 0:                 #图片无安全隐患
                false_report_count += 1   #错报图片数量加一
        else:                             #图片预测没有安全隐患
            if i[0] == 1:                    #图片有安全隐患
                missing_report_count += 1     #漏报图片数量加一
    print('False report count: %d'%false_report_count)
    print('Missing report count: %d'%missing_report_count)

    true_identification_rate = [0] * len(dict_class)   #每类预测准确率
    class_false_report__rate = [0] * len(dict_class)    #每类误检率
    for i in range(len(dict_class) ):
        true_identification_rate[i] = true_identification[i]/(class_count[i]) #计算每类预测准确率
        class_false_report__rate[i] = class_false_report[i]/(class_count[i])  #计算每类误报率
    false_report_rate = 1.0*false_report_count/image_with_danger_count            #计算误报率
    missing_report_rate = 1.0*missing_report_count/image_with_danger_count      #计算漏报率

    print('image_total_count:%d' %image_total_count)
    print('Image with danger count:%d'%image_with_danger_count)
    print('False report rate: %f' %false_report_rate)
    print('Missing report rate: %f' %missing_report_rate)
    print(dict_class)
    print('True identification:', true_identification)
    print('Class count:', class_count)
    print('True identification rate:', true_identification_rate)
    print('class_false_report__rate:', class_false_report__rate)

if __name__ == "__main__":
    calculate_performance_index_3label_pcl(cfg.root_data_dir, transform_result_dir=cfg.transform_result_dir, dict_class=cfg.class_dict)