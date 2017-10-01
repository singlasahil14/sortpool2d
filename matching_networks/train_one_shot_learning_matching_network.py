from one_shot_learning_network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import data as dataset
import argparse, json, os
from storage import *

tf.reset_default_graph()
tf.set_random_seed(99)
def add_arguments(parser):
  parser.add_argument('--epochs', default=30, type=int, help='number of epochs to run (default %(default)s)')
  parser.add_argument('--batch-size', default=32, type=int, help='batch size (default %(default)s)')
  parser.add_argument('--logs-path', default='result', type=str, help='Directory for storing training and eval logs')
  parser.add_argument('--fce', default=False, help='use full context embeddings', action='store_true')
  parser.add_argument('--classes', default=20, type=int, help='5 way or 20 way (default: %(default)s)', choices=[5, 20])
  parser.add_argument('--shots', default=1, type=int, help='1 shot or 5 shot (default: %(default)s)', choices=[1, 5])
  parser.add_argument('--train-batches', default=1000, type=int, help='number of training batches (default %(default)s)')
  parser.add_argument('--val-batches', default=250, type=int, help='number of validation batches (default %(default)s)')
  parser.add_argument('--test-batches', default=250, type=int, help='number of test batches (default %(default)s)')
  parser.add_argument('--pool-range', default=1, type=int, help='pool range for sort pooling (default %(default)s)', choices=[1,2,3,4])
  
def check_arguments(options):
  assert options.epochs > 0
  assert options.batch_size > 0
  assert not(os.path.exists(options.logs_path)), "result dir already exists!"
  assert options.train_batches > 0
  assert options.val_batches > 0
  assert options.test_batches > 0

# Experiment Setup
parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
check_arguments(args)

total_epochs = args.epochs
batch_size = args.batch_size
logs_path = args.logs_path
fce = args.fce
classes_per_set = args.classes
samples_per_class = args.shots
total_train_batches = args.train_batches
total_val_batches = args.val_batches
total_test_batches = args.test_batches
pool_range = args.pool_range

# Save experiment configuration
os.makedirs(logs_path)
config_str = json.dumps(vars(args))
config_file = os.path.join(logs_path, 'config')
config_file_object = open(config_file, 'w')
config_file_object.write(config_str)

# Experiment builder
data = dataset.OmniglotNShotDataset(batch_size=batch_size,
                                    classes_per_set=classes_per_set, samples_per_class=samples_per_class)
experiment = ExperimentBuilder(data, logs_path)
one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(batch_size,
                                                                              classes_per_set,
                                                                              samples_per_class, fce, 
                                                                              pool_range)

# Experiment initialization and running
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    best_val = float('inf')
    for e in range(total_epochs):
        total_c_loss, total_accuracy = experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                     sess=sess, epoch_id=e)
        print("Epoch {}: train_loss: {:.4f}, train_accuracy: {:.3f}".format(e, total_c_loss, total_accuracy))

        total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(
                                                          total_val_batches=total_val_batches,
                                                          sess=sess)
        print("Epoch {}: val_loss: {:.4f}, val_accuracy: {:.3f}".format(e, total_val_c_loss, total_val_accuracy))

        if total_val_c_loss <= best_val: #if new best val accuracy -> produce test statistics
            best_val = total_val_c_loss
            total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                                                                total_test_batches=total_test_batches, sess=sess)
            print("Epoch {}: test_loss: {:.4f}, test_accuracy: {:.3f}".format(e, total_test_c_loss, total_test_accuracy))
        else:
            total_test_c_loss = -1
            total_test_accuracy = -1

