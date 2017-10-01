import tensorflow as tf
import time
import sys, os
from one_shot_learning_network import MatchingNetwork
from collections import defaultdict
import pandas as pd

class ExperimentBuilder:

    def __init__(self, data, logs_path):
        """
        Initializes an ExperimentBuilder object. The ExperimentBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data
        self.logs_path = logs_path
        self.train_metrics = defaultdict(list)
        self.eval_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, fce, pool_range):

        """

        :param batch_size: The experiment batch size
        :param classes_per_set: An integer indicating the number of classes per support set
        :param samples_per_class: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :param pool_range: Range of pooling in sortpool_2d
        :return: a matching_network object, along with the losses, the training ops and the init op
        """
        height, width, channels = self.data.x.shape[2], self.data.x.shape[3], self.data.x.shape[4]
        self.support_set_images = tf.placeholder(tf.float32, [batch_size, classes_per_set, samples_per_class, height, width,
                                                              channels], 'support_set_images')
        self.support_set_labels = tf.placeholder(tf.int32, [batch_size, classes_per_set, samples_per_class], 'support_set_labels')
        self.target_image = tf.placeholder(tf.float32, [batch_size, height, width, channels], 'target_image')
        self.target_label = tf.placeholder(tf.int32, [batch_size], 'target_label')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.rotate_flag = tf.placeholder(tf.bool, name='rotate-flag')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout-prob')
        self.current_learning_rate = 1e-03
        self.learning_rate = tf.placeholder(tf.float32, name='learning-rate-set')
        self.one_shot_omniglot = MatchingNetwork(batch_size=batch_size, support_set_images=self.support_set_images,
                                            support_set_labels=self.support_set_labels,
                                            target_image=self.target_image, target_label=self.target_label,
                                            keep_prob=self.keep_prob, num_channels=channels,
                                            is_training=self.training_phase, fce=fce, rotate_flag=self.rotate_flag,
                                            num_classes_per_set=classes_per_set,
                                            num_samples_per_class=samples_per_class, learning_rate=self.learning_rate, pool_range=pool_range)

        summary, self.losses, self.c_error_opt_op = self.one_shot_omniglot.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        return self.one_shot_omniglot, self.losses, self.c_error_opt_op, init

    def run_training_epoch(self, total_train_batches, sess, epoch_id):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        start_time = time.time()
        for i in range(total_train_batches):  # train epoch
            x_support_set, y_support_set, x_target, y_target = self.data.get_train_batch(augment=True)
            _, c_loss_value, acc = sess.run(
                [self.c_error_opt_op, self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                           self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                           self.training_phase: True, self.rotate_flag: False, self.learning_rate: self.current_learning_rate})

            if i % 100 == 0:
                iter_out = "epoch: {}, iter: {}, train_loss: {:.4f}, train_accuracy: {:.3f}, time_per_iter: {:.3f}".format(epoch_id, 
                           i, float(c_loss_value), float(acc), (time.time()-start_time)/(i+1))
                print(iter_out)

            total_c_loss += c_loss_value
            total_accuracy += acc
            self.total_train_iter += 1
            if self.total_train_iter % 2000 == 0:
                self.current_learning_rate /= 2
                print("change learning rate", self.current_learning_rate)

        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches

        self.train_metrics['loss'].append(total_c_loss)
        self.train_metrics['accuracy'].append(total_accuracy)
        pd_train_metrics = pd.DataFrame(self.train_metrics)
        pd_train_metrics.to_csv(os.path.join(self.logs_path, 'train_metrics.csv'))
        return total_c_loss, total_accuracy

    def run_validation_epoch(self, total_val_batches, sess):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.

        for i in range(total_val_batches):  # validation epoch
            x_support_set, y_support_set, x_target, y_target = self.data.get_val_batch(augment=True)
            c_loss_value, acc = sess.run(
                [self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                           self.support_set_labels: y_support_set, self.target_image: x_target, self.target_label: y_target,
                           self.training_phase: False, self.rotate_flag: False})

            total_val_c_loss += c_loss_value
            total_val_accuracy += acc

        total_val_c_loss = total_val_c_loss / total_val_batches
        total_val_accuracy = total_val_accuracy / total_val_batches

        self.eval_metrics['loss'].append(total_val_c_loss)
        self.eval_metrics['accuracy'].append(total_val_accuracy)
        pd_eval_metrics = pd.DataFrame(self.eval_metrics)
        pd_eval_metrics.to_csv(os.path.join(self.logs_path, 'eval_metrics.csv'))
        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self, total_test_batches, sess):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        for i in range(total_test_batches):
            x_support_set, y_support_set, x_target, y_target = self.data.get_test_batch(augment=True)
            c_loss_value, acc = sess.run(
                [self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                           self.support_set_labels: y_support_set, self.target_image: x_target,
                           self.target_label: y_target,
                           self.training_phase: False, self.rotate_flag: False})

            total_test_c_loss += c_loss_value
            total_test_accuracy += acc


        total_test_c_loss = total_test_c_loss / total_test_batches
        total_test_accuracy = total_test_accuracy / total_test_batches

        self.test_metrics['loss'].append(total_test_c_loss)
        self.test_metrics['accuracy'].append(total_test_accuracy)
        pd_test_metrics = pd.DataFrame(self.test_metrics)
        pd_test_metrics.to_csv(os.path.join(self.logs_path, 'test_metrics.csv'))
        return total_test_c_loss, total_test_accuracy
