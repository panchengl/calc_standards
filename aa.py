# # dict_a = {"a":1}
# # print(len(dict_a))
# import torch
# a = torch.randn(1,3, 11, 11)
# # print(a)
# conv2 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
# b = conv2(a)
# print(a)
#
# import torch.nn as nn
# import torch
#
# im = torch.randn(1, 1, 5, 5)
# c = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=1)
# output = c(im)
#
# print(im)
# print(output)
#
#

import os
print(len(os.listdir("/home/pcl/data/VOC2007/JPEGImages")))
