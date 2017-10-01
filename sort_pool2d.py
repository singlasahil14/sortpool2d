import tensorflow as tf

def sort_pool2d(x, k=1, padding='SAME', name='pool'):
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

  with tf.variable_scope(name):
    pool_weights = tf.get_variable('pool_weights', [1,1,1,1,k], 
                tf.float32, initializer=tf.random_normal_initializer(stddev=0.1))
    pool_weights = tf.nn.softmax(pool_weights)

  weighted_subsets = pool_weights*sorted_sub
  x = tf.reduce_sum(weighted_subsets, 4)
  return x
