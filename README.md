20200611 updates:

    1. add txt calc map and product standards, one img must have one txt file, for example, a img have 2 objs, txt file just like:

        class1 score xmin ymin xmax ymax

        class2 scaoe xmin ymin xmax ymax

    2. when u use txt method, u need do this follows:

        first: create txt file, one img one txt file

        second: modify the params in cfg.py ,  txt_data_dir , val_file, inference_type

        last: python calc_standards

    additions:

        if u do not how to create txt files, u can use utils/tf_utils.py/tf1_inference_write_results()

20200426 updates:

    1. add pytorch_centernet inference: https://github.com/xingyizhou/CenterNet

    2. fixes some bugs when calc map, code not deal with images with no object, so, if fixed, decreased precision, unchanged recall rate

    3. add code for drawimg P-R curve of different class


The function of this project is as follows:

    1. Calculate the map value of different yolov3 algorithm versions

    2. Calculate the product-level standards of different yolov3 algorithm versions

    3. Count the number of different kinds of targets in the own datasets

    4. The program can be expanded to use new algorithms for calculating voc map values because newer algorithms use coco standards insteda of voc standards

The project support following yolov3 verisons(tf-version only support GPU, other version also support CPU):

    1. official darknet_yolov3:  must use cfg file and weights file

    2. pytorch version: https://github.com/ultralytics/yolov3

    3. tensorflow verison: https://github.com/wizyoung/YOLOv3_TensorFlow

    4. tensorflow2mnn version: https://github.com/ShuiXianhua/YOLO_V3

   U can use my project in your own datasets

next stage: support more algorithms, just like SOTA object detection algorithms

Requirements:

    tensorflow>=1.12, pytoch>=1.0, torchvision, tqdm, opencv, pillow, matplotlib, numpy.


files stage:

      my_code/data/Annotations: xml files

      my_code/data/image:       imgs

      my_code/data/*.names:     your labels file

      my_code/weights/:         your model_dir

      then edit params in cfg.py

How To Use:

    1. create txt datasets(val file): put your datasets become like this:  img_id img_dir width height label x0 y0 x1 y1 label x0 y0 x1 y1  ...

    just like:

                0 aaa.jpg 1200 900 1 355 252 481 295

                1 bbb.jpg 1200 900 4 805 478 901 558
                ...

    2. edit your file dir in cfg.py according your algorithm version

    just like: i use pytorch version, i will edit pytorch params in cfg.py  ,  included params : inference_type model_dir cfg_file, img_size, single_img_inference_save_dir.....

    3.calc map

    just like:  python calc_map.py   , after inference finished, program will print map values

    4. calc product-level standards

    just like: python calc_standard.py , after inference finished, program will print  values

# calc_standards
