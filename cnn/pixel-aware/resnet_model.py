#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    AvgPooling, Conv2D, GlobalAvgPooling, FullyConnected,
    LinearWrap, BNReLU, BatchNorm)

from tensorpack.models.common import layer_register

import cv2

import numpy as np

import random
import math

from tensorpack.tfutils.tower import get_current_tower_context


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)

def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l_3x3 = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=tf.identity)

    shape = l_3x3.get_shape().as_list()
    l_gap = GlobalAvgPooling('gap', l)
    l_gap = FullyConnected('fc1', l_gap, ch_out, activation=tf.nn.relu)
    l_gap = FullyConnected('fc2', l_gap, ch_out, activation=tf.identity)
    l_gap = tf.reshape(l_gap, [-1, ch_out, 1, 1])
    l_gap = tf.tile(l_gap, [1, 1, shape[2], shape[3]])
    
    l_concat = tf.concat([l_3x3, l_gap], axis = 1)
    l_concat = Conv2D('conv_c1', l_concat, ch_out, 1, strides=1, activation=tf.nn.relu)
    l_concat = Conv2D('conv_c2', l_concat, ch_out, 1, strides=1, activation=tf.identity)
    l_concat = tf.sigmoid(l_concat)

    l = l_3x3 + l_gap * l_concat
    l = BNReLU('conv2',l)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))


def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))



def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
                # l = BNReLU('block{}'.format(i), l)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope([Conv2D], use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        logits = (LinearWrap(image).Conv2D('conv0', 64, 7, strides=2, activation=BNReLU)())
        logits = (LinearWrap(logits).MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func, 'group2', block_func, 256, num_blocks[2], 2)
                  .apply(group_func, 'group3', block_func, 512, num_blocks[3], 2)())
        logits = (LinearWrap(logits).GlobalAvgPooling('gap')())
        logits = (LinearWrap(logits).FullyConnected('linear', 1000)())
    return logits
