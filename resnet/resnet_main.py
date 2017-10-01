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

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import os
import json

import cifar_input
import numpy as np, pandas as pd
from collections import OrderedDict, defaultdict
import resnet_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_integer('eval_batch_count', 40,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_string('result_path', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_float('weight_decay', 0.0002,
                          'weight decay rate for network weights')
tf.app.flags.DEFINE_integer('pool_type', 1, 'pooling type for network')
tf.set_random_seed(99)

def train(hps):
  """Training loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.data_path, hps.batch_size, 'train')
  model = resnet_model.ResNet(hps, images, labels, 'train')
  model.build_graph()

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=os.path.join(FLAGS.result_path, 'summary'),
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors=OrderedDict(zip(['iterations', 'total loss', 'cross entropy', 'weight decay', 
                               'accuracy'],
                      [model.global_step, model.cost, model.xent_cost, 
                       model.decay_cost, precision])),
      every_n_iter=100)

  images_eval, labels_eval = cifar_input.build_input(
    FLAGS.dataset, FLAGS.data_path,
    hps.batch_size, 'eval')
  eval_model = resnet_model.ResNet(hps, images_eval,
                                   labels_eval, 'eval')
  tf.get_variable_scope().reuse_variables()
  eval_model.build_graph()

  train_metrics_dict = defaultdict(list)
  eval_metrics_dict = defaultdict(list)
  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          [model.global_step,  # Asks for global step value.
          model.cost, precision, model.xent_cost,
          model.decay_cost],
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results[0]
      train_loss = run_values.results[1]
      train_precision = run_values.results[2]
      xent_cost = run_values.results[3]
      decay_cost = run_values.results[4]
      train_metrics_dict['iterations'].append(train_step)
      train_metrics_dict['total_loss'].append(train_loss)
      train_metrics_dict['accuracy'].append(train_precision)
      train_metrics_dict['cross_entropy'].append(xent_cost)
      train_metrics_dict['decay_loss'].append(decay_cost)

      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 60000:
        self._lrn_rate = 0.01
      elif train_step < 80000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001
      if((train_step%1000)==0):
        total_prediction, correct_prediction = 0, 0
        total_loss = 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
          (loss, predictions, truth) = run_context.session.run(
              [eval_model.xent_cost, eval_model.predictions,
               eval_model.labels])
          truth = np.argmax(truth, axis=1)
          predictions = np.argmax(predictions, axis=1)
          correct_prediction += np.sum(truth == predictions)
          total_prediction += predictions.shape[0]
          total_loss += loss
        eval_precision = correct_prediction / float(total_prediction)
        eval_loss = total_loss / float(FLAGS.eval_batch_count)
        print('eval_precision: %f, eval_loss: %f', eval_precision, eval_loss)

        eval_metrics_dict['iterations'].append(train_step)
        eval_metrics_dict['cross_entropy'].append(eval_loss)
        eval_metrics_dict['accuracy'].append(eval_precision)

        trainmetrics_df = pd.DataFrame(train_metrics_dict)
        trainmetrics_df.to_csv(os.path.join(FLAGS.result_path, 'train_metrics.csv'))

        evalmetrics_df = pd.DataFrame(eval_metrics_dict)
        evalmetrics_df.to_csv(os.path.join(FLAGS.result_path, 'eval_metrics.csv'))

  config_proto = tf.ConfigProto(allow_soft_placement=True)
  config_proto.gpu_options.allow_growth = True
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.result_path,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=config_proto) as mon_sess:
    for i in range(100000):
      if mon_sess.should_stop():
        break
      mon_sess.run(model.train_op)

def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  batch_size = 128

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  weight_decay_rate = FLAGS.weight_decay
  pool_type = FLAGS.pool_type
  hps = resnet_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=weight_decay_rate,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             pool_type=pool_type)
  if not os.path.exists(FLAGS.result_path):
    os.makedirs(FLAGS.result_path)
  config_str = json.dumps(hps._asdict())
  config_file = os.path.join(FLAGS.result_path, 'config')
  config_file_object = open(config_file, 'w')
  config_file_object.write(config_str)
  config_file_object.close()

  with tf.device(dev):
    train(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
