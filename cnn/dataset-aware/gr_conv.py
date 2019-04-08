#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: p_norm.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from tensorpack.utils import logger
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack import *
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format

from tensorpack.models import (BNReLU)


import numpy as np

__all__ = ['Grconv']

def fc(x, out_shape,padding='same',
    data_format='channels_first',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    split=1,): 
    x = GlobalAvgPooling('gap', x)
    x = FullyConnected('fc1', x, out_shape[1], activation=tf.nn.relu)
    x = FullyConnected('fc3', x, out_shape[1])
    x = tf.reshape(x, [-1, out_shape[1], 1, 1])
    return x


@layer_register()
def Grconv(x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='same',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    split=1,
    glocal = False):
    z3 = Conv2D('z3', x, filters=filters, kernel_size=3, strides=strides, 
        padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=tf.identity, use_bias=use_bias, 
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, split=1)
    out_shape = z3.get_shape().as_list()

    z1 = Conv2D('z1', x, filters=filters, kernel_size=1, strides=strides, 
        padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=tf.identity, use_bias=use_bias, 
        kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, 
        bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, split=1)
    zf = fc(x, out_shape,padding=padding,data_format=data_format,dilation_rate=dilation_rate,activation=tf.identity,
        use_bias=use_bias,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer,kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,activity_regularizer=activity_regularizer)

    p = tf.get_variable('PPP', [3, 1, out_shape[2], out_shape[3]], initializer=tf.ones_initializer(), trainable = True)
    p = tf.nn.softmax(p, 0)

    z = p[0:1,:,:,:] * z3 + p[1:2,:,:,:] * z1 + p[2:3,:,:,:] * zf
    z = BNReLU('sum', z)
    return z