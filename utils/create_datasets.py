import xml.etree.ElementTree as ET
from os import getcwd
import os
sets=[('santachi_3label', 'val')]

# orignal_classes = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "SuLiaoBu", "FengZheng", "Niao", "NiaoWo", "ShanHuo", "YanWu", "JianGeBang", "JueYuanZi", "FangZhenChui"]
orignal_classes = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
classes = ["DiaoChe", "TaDiao", "ShiGongJiXie"]
# img_dir = "/home/pcl/data/VOC2007/JPEGImages"
# anno_dir = "/home/pcl/data/VOC2007/Annotations"

img_dir = "/home/pcl/tf_work/map/data/image_santachi"
anno_dir = "/home/pcl/tf_work/map/data/image_santachi"
result_dir = "/home/pcl/tf_work/map/data/result_ssd3x3"

def convert_annotation(year, in_file, list_file):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    # in_file = open(anno_dir+ '/%s.xml'% image_id.split('.')[0])
    tree=ET.parse(in_file)
    root = tree.getroot()
    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)
    list_file.write(" " + str(width) + " " + str(height))
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if cls not in orignal_classes or int(difficult)==1:
            continue
        try:
            cls_id = orignal_classes.index(cls)
            if cls_id in [2, 3, 4, 5]:
                cls_id = 2
        except:
            continue
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + str(cls_id) + " " + " ".join([str(a) for a in b]))

wd = getcwd()

def process(img_dir):
    num = 0
    # filelist = os.listdir(img_dir)
    for root, dirs, files in os.walk(img_dir):
        if len(files) != 0:
            for file in files:
                num = num + 1
                Olddir = os.path.join(root, file)
                print('old dir is ', Olddir)
                # if os.path.isdir(Olddir):
                filename = os.path.splitext(file)[0]
                filetype = os.path.splitext(file)[1]
                Newdir = os.path.join(root, filename.replace(' ', '').replace(',', '').replace('(','').replace(')','').replace(',','') + filetype);
                os.rename(Olddir, Newdir)
                print('new dir is ', Newdir)
    print(num)

for year, image_set in sets:
    process(result_dir)
    file_list = []
    for root, dirs, files in os.walk(img_dir):
        if len(files) != 0:
            for i in files:
                file_type = i.split('.')[-1]
                if file_type == 'JPG' or file_type == 'jpeg' or file_type == 'jpg':
                    file_list.append(os.path.join(root, i))
    # print(file_list)
    print("[INFO] the current img nums is %d" %len(file_list))
    num = 0
    list_file = open('../data/%s_%s.txt'%(year, image_set), 'w')
    for image_id in file_list:
        print(image_id)
        num = num + 1
        if(num %1000 == 0):
            print("current deal img_num is %d"%num)
        if image_id.split('.')[-1] != 'jpg':
            continue
        anno_file = image_id.replace('jpg', 'xml').replace('JPG', 'xml').replace('jpeg', 'xml').split('/')[-2:]
        print(anno_file)
        try:
            in_file = open(os.path.join(anno_dir, anno_file[0], anno_file[1]))
        except:
            continue
        # list_file.write(str(num) + ' ' + img_dir + '/%s' %(image_id))
        list_file.write(str(num) + ' ' + '%s' %(image_id))
        convert_annotation(year, in_file, list_file)
        # num = num + 1
        list_file.write('\n')
    list_file.close()
