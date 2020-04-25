import xml.etree.ElementTree as ET
import pickle
import os
from utils.common_util import create_directory

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
		filename = os.path.join(transform_result_dir , name + ".txt")
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

if __name__ == "__main__":
	class_id_dict = {"DiaoChe": 0, "TaDiao": 0, "TuiTuJi": 0, "BengChe": 0, "WaJueJi": 0, "ChanChe": 0, "ShanHuo":0, "YanWu":0, "SuLiaoBu": 0}
	names = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
	# anno_dir = "/home/pcl/tf_work/map/data/Annotations"
	# anno_dir = "/home/pcl/data/dianli_zhongkeyuan/"
	anno_dir = "/home/pcl/data/VOC2007/Annotations"
	all_dict = static_class_num(anno_dir, class_id_dict)
	print(all_dict)
	# transform_data(result_dir="/home/pcl/tf_work/map/data/mnn_txt", transform_result_dir='/home/pcl/tf_work/map/data/mnn_class', classes=names)