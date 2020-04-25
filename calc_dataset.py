from utils.static_util import static_class_num,check_img_anno
import os
import xml.etree.ElementTree as ET
import pickle
class_id_dict = {"DiaoChe": 0, "TaDiao": 0, "TuiTuJi": 0, "BengChe": 0, "WaJueJi": 0, "ChanChe": 0, "YanWu":0, "ShanHuo":0, "SuLiaoBu":0, "DaoXianYiWu":0}
anno_dir = "/home/pcl/data/dianli_orignal/"
anno_dir = "/home/pcl/data/dianli_zhongkeyuan/"
result = static_class_num(anno_dir, class_id_dict)
print(result)
ShiGongJiXie = 0
ShiGongJiXie = class_id_dict['TuiTuJi'] + class_id_dict['BengChe'] + class_id_dict["WaJueJi"] + class_id_dict["ChanChe"]
print(ShiGongJiXie)
img_dir = "/home/pcl/data/dianli_zhongkeyuan/"
anno_dir = "/home/pcl/data/dianli_zhongkeyuan/"
check_img_anno(img_dir, anno_dir)
