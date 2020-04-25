# coding: utf-8

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim


def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    # inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    inputs = tf.image.resize_bilinear(inputs, size=[new_height, new_width], name='upsampled')
    return inputs


def yolo_block_pecentage(inputs, filters, pecentage, prune_cnt=1):
    import numpy as np
    true_filters_1 = filters
    true_filters_2 = filters * 2
    for i in range(prune_cnt):
        true_filters_1 = np.floor(true_filters_1 * pecentage)
        true_filters_2 = np.floor(true_filters_2 * pecentage)
    net = conv2d(inputs, true_filters_1, 1)
    net = conv2d(net, true_filters_2, 3)
    net = conv2d(net, true_filters_1, 1)
    net = conv2d(net, true_filters_2, 3)
    net = conv2d(net, true_filters_1, 1)
    route = net
    net = conv2d(net, true_filters_2, 3)

    return route, net