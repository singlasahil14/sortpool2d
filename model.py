from collections import namedtuple

import tensorflow as tf
import six

from tensorflow.python.training import moving_averages

class Network(object):
  """General layers in a neural network."""

  def __init__(self):
    """Network constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self._nonlin_dict = {'relu': self._relu, 'selu': self._selu, 
      'maxout': self._maxout, 'identity': self._identity}

  def _get_shape(self, x):
    """Get shape of tensor as a tuple."""
    shp = tf.shape(x)
    return shp[0], shp[1], shp[2], shp[3]

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, x, name='batch_norm'):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))
      ndims = len(x.get_shape().as_list())
      if ndims==2:
          axes = [0]
      elif ndims==4:
          axes = [0, 1, 2]
      else:
          raise ValueError('Undefined axes for ndims %d' % ndims)
      mean, variance = tf.nn.moments(x, axes, name='moments')
      # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.0001)
      y.set_shape(x.get_shape())
      return y

  def _conv(self, filter_size, out_filters, name='conv'):
    """Convolution."""
    def conv_fn(inp, stride, padding):
      with tf.variable_scope(name):
        in_filters = int(inp.get_shape()[-1])
        n = filter_size * filter_size * out_filters
        conv_filter = tf.get_variable(
            'DW', [filter_size, filter_size, in_filters, out_filters],
            tf.float32, initializer=tf.random_normal_initializer(
                stddev=tf.sqrt(2.0/n)))
        return tf.nn.conv2d(inp, conv_filter, [1, stride, stride, 1], padding)
    return conv_fn

  def _dense(self, out_dim, name='dense'):
    """Dense layer for final output."""
    def dense_fn(x):
      with tf.variable_scope(name):
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
              initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
              initializer=tf.constant_initializer(0.0))
      return tf.nn.xw_plus_b(x, w, b)
    return dense_fn

  def _flatten(self, x):
    x = tf.contrib.layers.flatten(x)
    return x

  def _max_pool2d(self, inp, ksize, stride, padding='SAME', name='max_pool2d'):
    return tf.nn.max_pool(inp,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)

  def _sort_pool2d(self, x, padding='SAME', k=1, name='sort_pool2d'):
    if k==1:
        return self._max_pool2d(x, 2, 2, padding=padding, name=name)
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
    sorted_sub = sorted_sub

    with tf.variable_scope(name):
      pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k], 
          tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
    pool_weights = tf.nn.softmax(pool_weights)

    weighted_subsets = pool_weights*sorted_sub
    x = tf.reduce_sum(weighted_subsets, 4)
    return x

  def _identity(self, x, lin_fn, name='identity'):
    with tf.variable_scope(name):
      x = lin_fn(x)
    return tf.identity(x, name='identity')

  def _relu(self, x, lin_fn=tf.identity, name='relu'):
    """Relu, with optional leaky support."""
    x = lin_fn(x)
    return tf.where(tf.less(x, 0.), tf.zeros(tf.shape(x)), x, name='relu')

  def _selu(self, x, lin_fn=tf.identity, name='selu'):
    """Selu, a normalizing activation function."""
    x = lin_fn(x)
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return tf.identity(scale*tf.where(tf.less(x, 0.0), alpha * tf.elu(x), x), name='selu')

  def _maxout(self, x, lin_fn, name='maxout'):
    with tf.variable_scope('max1'):
      out1 = lin_fn(x)
    with tf.variable_scope('max2'):
      out2 = lin_fn(x)
    return tf.maximum(out1, out2, name=name)

  def _tanh(self, x, lin_fn=tf.identity, name='tanh'):
    """tanh activation function."""
    x = lin_fn(x)
    return tf.nn.tanh(x, name='tanh')

  def _softmax(self, x):
    x = tf.nn.softmax(x)
    return x
