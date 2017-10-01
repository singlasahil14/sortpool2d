# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'pool_type, relu_leakiness, optimizer')


class ResNet(object):
  """ResNet model."""

  def __init__(self, hps, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    """Build a whole graph for the model."""
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    self._build_model()
    if self.mode == 'train':
      self._build_train_op()
    self.summaries = tf.summary.merge_all()

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._images
      x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 4

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in six.moves.range(1, self.hps.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      embeddings = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(embeddings, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    with tf.variable_scope('costs'):
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      self.xent_cost = tf.reduce_mean(xent, name='xent')
      self.decay_cost = self._decay()
      self.cost = self.xent_cost + tf.multiply(self.hps.weight_decay_rate, self.decay_cost)

      tf.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

    apply_op = optimizer.apply_gradients(
        zip(grads, trainable_variables),
        global_step=self.global_step, name='train_step')

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.mode == 'train':
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _weighted_avg_pool(self, x, ksize, strides, padding='VALID'):
    num_channels = x.get_shape().as_list()[-1]
    weights = tf.get_variable('pool_weights', ksize[1:] + [1], tf.float32, 
           initializer=tf.random_normal_initializer(stddev=0.05))
    weights = tf.tile(weights, [1, 1, num_channels, 1])
    x = tf.nn.depthwise_conv2d(x, weights, strides=strides, padding=padding)
    return x

  def _sort_pool2d(self, x, k=1, padding='SAME'):
    assert k in [1,2,3,4]
    if k==1:
      return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], padding=padding)
    batch_size, height, width, num_channels = x.get_shape().as_list()
    pad_bottom = height%2
    pad_right = width%2
    height_div2 = height + pad_bottom
    width_div2 = width + pad_right
    if(padding=='SAME'):
      x = tf.pad(x, [[0, 0], [0, pad_bottom], [0, pad_right], [0, 0]], "CONSTANT")

    _, height, width, _ = x.get_shape().as_list()
    offsets_y = tf.range(0, height, 2)
    offsets_x = tf.range(0, width, 2)

    sub_y0 = tf.gather(x, offsets_y, axis=1)
    sub_y1 = tf.gather(x, offsets_y + 1, axis=1)

    sub_00 = tf.gather(sub_y0, offsets_x, axis=2)
    sub_00 = tf.reshape(sub_00, [-1])
    sub_01 = tf.gather(sub_y0, offsets_x + 1, axis=2)
    sub_01 = tf.reshape(sub_01, [-1])

    sub_10 = tf.gather(sub_y1, offsets_x, axis=2)
    sub_10 = tf.reshape(sub_10, [-1])
    sub_11 = tf.gather(sub_y1, offsets_x + 1, axis=2)
    sub_11 = tf.reshape(sub_11, [-1])

    sub0 = tf.where(tf.less(sub_00, sub_01), tf.stack([sub_00, sub_01], axis=1), 
                    tf.stack([sub_01, sub_00], axis=1))
    sub1 = tf.where(tf.less(sub_10, sub_11), tf.stack([sub_10, sub_11], axis=1), 
                    tf.stack([sub_11, sub_10], axis=1))

    sub00 = tf.squeeze(tf.slice(sub0, [0, 0], [-1, 1]))
    sub01 = tf.squeeze(tf.slice(sub0, [0, 1], [-1, 1]))

    sub10 = tf.squeeze(tf.slice(sub1, [0, 0], [-1, 1]))
    sub11 = tf.squeeze(tf.slice(sub1, [0, 1], [-1, 1]))

    # assume elem1 <= elem3
    def sort_elems(elem1, elem2, elem3, elem4):
      elem2_less_than_elem3 = tf.stack([elem1, elem2, elem3, elem4], axis=1)
      elem2_greater_equal_elem3_and_elem2_less_than_elem4 = tf.stack([elem1, elem3, elem2, elem4], axis=1)
      elem2_greater_equal_elem3_and_elem2_greater_equal_elem4 = tf.stack([elem1, elem3, elem4, elem2], axis=1)
      elem2_greater_equal_elem3 = tf.where(tf.less(elem2, elem4), 
        elem2_greater_equal_elem3_and_elem2_less_than_elem4, elem2_greater_equal_elem3_and_elem2_greater_equal_elem4)
      return tf.where(tf.less(elem2, elem3), elem2_less_than_elem3, elem2_greater_equal_elem3)

    sub00_less_sub10 = sort_elems(sub00, sub01, sub10, sub11)
    sub00_greater_equal_sub10 = sort_elems(sub10, sub11, sub00, sub01)

    sorted_sub_flat = tf.where(tf.less(sub00, sub10), sub00_less_sub10, sub00_greater_equal_sub10)
    sorted_sub = tf.slice(sorted_sub_flat, [0, 4-k], [-1, k])
    sorted_sub = tf.reshape(sorted_sub, [-1, int(height/2), int(width/2), num_channels, k])

    pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k], 
                tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
    pool_weights = tf.nn.softmax(pool_weights)

    weighted_subsets = pool_weights*sorted_sub
    x = tf.reduce_sum(weighted_subsets, 4)
    return x

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        if (self.hps.pool_type==0) or not(stride==[1,2,2,1]):
          orig_x = self._weighted_avg_pool(orig_x, stride, stride, 'VALID')
        else:
          orig_x = self._sort_pool2d(orig_x, self.hps.pool_type, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                           activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    tf.logging.info('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))
        # tf.summary.histogram(var.op.name, var)

    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      conv_filter = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=tf.sqrt(2.0/n)))
      return tf.nn.conv2d(x, conv_filter, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    x = tf.reshape(x, [self.hps.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
