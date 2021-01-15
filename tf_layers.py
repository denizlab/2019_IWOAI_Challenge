# ==============================================================================
# Copyright (C) 2020 Vladimir Juras, Ravinder Regatte and Cem M. Deniz
#
# This file is part of 2019_IWOAI_Challenge
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

import tensorflow as tf


def _variable_on_cpu(name, shape, initializer):
    '''Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    '''
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, scale, initializer=tf.contrib.layers.variance_scaling_initializer()):
    #Helper to create an initialized Variable with weight decay.
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer,
                              regularizer=tf.contrib.layers.l2_regularizer(scale))
    return var


def conv3d(name, bottom, num_output, kernel_size=[3,3,3], reg_constant=0.0, strides=[1,1,1,1,1],
            padding='SAME', initializer=tf.contrib.layers.variance_scaling_initializer(), bias=True):
    bottom_shape = bottom.get_shape()
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[kernel_size[0], kernel_size[1], kernel_size[2], bottom_shape[4], num_output],
                                             scale=reg_constant, initializer=initializer)
        conv = tf.nn.conv3d(bottom, kernel, strides, padding=padding)
        if bias:
            biases = _variable_on_cpu('biases', [num_output], initializer=tf.constant_initializer(0.0))
            top = tf.nn.bias_add(conv, biases, name=scope.name)
        else:
            top = conv
    print (name,padding, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top), max_outputs=4)
    return top


def downconv3d(name, bottom, num_output, kernel_size=[2,2,2], reg_constant=0.0, strides=[1,2,2,2,1],
            padding='SAME', initializer=tf.contrib.layers.variance_scaling_initializer()):
    bottom_shape = bottom.get_shape()
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[kernel_size[0], kernel_size[1], kernel_size[2], bottom_shape[4], num_output],
                                             scale=reg_constant, initializer=initializer)
        conv = tf.nn.conv3d(bottom, kernel, strides, padding=padding)
        biases = _variable_on_cpu('biases', [num_output], initializer=tf.constant_initializer(0.0))
        top = tf.nn.bias_add(conv, biases, name=scope.name)
    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def max_pool(name, bottom):
    with tf.variable_scope(name) as scope:
        top = tf.nn.max_pool3d(input=bottom, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', name=name)
    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def upconv3d(name, bottom, num_output, out_value_shape, kernel_size=[2,2,2], reg_constant=0.0, strides=[1, 2, 2, 2, 1],
              padding='SAME', initializer=tf.contrib.layers.variance_scaling_initializer()):
    batch_size = tf.shape(bottom)[0]
    bottom_shape = bottom.get_shape()
    output_shape = tf.stack([batch_size, out_value_shape[0], out_value_shape[1], out_value_shape[2], num_output])
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[kernel_size[0], kernel_size[1], kernel_size[2], num_output, bottom_shape[4]],
                                             scale=reg_constant, initializer=initializer)
        conv = tf.nn.conv3d_transpose(bottom, kernel, output_shape, strides, padding=padding)
        biases = _variable_on_cpu('biases', [num_output], initializer=tf.constant_initializer(0.0))
        top = tf.nn.bias_add(conv, biases, name=scope.name)
    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def atrousconv3d(name, bottom, num_output, kernel_size=[3,3,3], reg_constant=0.0, strides=[1, 1, 1], dilation_rate=[2,2,1], 
              padding='SAME', initializer=tf.contrib.layers.variance_scaling_initializer()):
    batch_size = tf.shape(bottom)[0]
    bottom_shape = bottom.get_shape()
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[kernel_size[0], kernel_size[1], kernel_size[2], num_output, bottom_shape[4]],
                                             scale=reg_constant, initializer=initializer)
        conv = tf.nn.convolution(bottom, kernel, padding=padding, strides=strides, dilation_rate=dilation_rate)
        biases = _variable_on_cpu('biases', [num_output], initializer=tf.constant_initializer(0.0))
        top = tf.nn.bias_add(conv, biases, name=scope.name)
    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def relu(name, bottom):
    with tf.variable_scope(name) as scope:
        top = tf.nn.relu(bottom, name=name)
    #tf.summary.histogram(name + '/activations', top)
    #tf.summary.image(name, _get_image_summary(top), max_outputs=4)
    return top


def add_res(name, bottom, res, conv=True, reg_constant=0.0, skip=False):
    if skip:
        return bottom
    
    if conv:
        bottom_shape = bottom.get_shape()
        res_shape = res.get_shape()
        with tf.variable_scope(name) as scope:
            kernel = _variable_with_weight_decay('weights', shape=[1, 1, 1, res_shape[4], bottom_shape[4]],
                                            scale=reg_constant)
            conv = tf.nn.conv3d(res, kernel, strides=[1,1,1,1,1], padding='SAME')
            top = tf.add(bottom, conv, name=name)
    else:
        with tf.variable_scope(name) as scope:
            top = tf.add(bottom, res, name=name)
    #tf.summary.image(name, _get_image_summary(top))
    return top


def concat(name, x1, x2):
    with tf.variable_scope(name) as scope:
        top = tf.concat([x1, x2], 4, name=name)

    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def multiconcat(name, xs):
    with tf.variable_scope(name) as scope:
        top = tf.concat(xs, 4, name=name)

    print (name, top.get_shape())
    #tf.summary.image(name, _get_image_summary(top))
    return top


def dropout(name, bottom, keep_prob=0.5):
    top = tf.nn.dropout(bottom, keep_prob, name=name)
    return top


def _get_image_summary(img, idx=0):
    '''
    Make an image summary for 5d tensor image with index idx
    '''
    img_z = tf.shape(img)[3]
    V = tf.slice(img, (0, 0, 0, img_z//2-1, idx), (1, -1, -1, 3, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, -1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))

    return V