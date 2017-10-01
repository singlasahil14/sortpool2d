import tensorflow as tf
import os, gzip, numpy as np

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.expand_dims(np.frombuffer(lbpath.read(), 
                               dtype=np.uint8, offset=8), axis=1)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)
    return images, labels

def load_data(dataset):
  if(dataset=='fashion-mnist'):
    x_train, y_train = load_mnist('data/fashion', kind='train')
    x_test, y_test = load_mnist('data/fashion', kind='t10k')

    y_train, y_test = dense_to_one_hot(y_train, n_classes=10), dense_to_one_hot(y_test, n_classes=10)
  elif(dataset=='cluttered-mnist'):
    mnist_cluttered = np.load('data/mnist_sequence1_sample_5distortions5x5.npz')
    x_train, y_train = mnist_cluttered['X_train'], mnist_cluttered['y_train']
    x_valid, y_valid = mnist_cluttered['X_valid'], mnist_cluttered['y_valid']
    x_test, y_test = mnist_cluttered['X_test'], mnist_cluttered['y_test']
    
    x_train = np.concatenate((x_train, x_valid))
    y_train = np.concatenate((y_train, y_valid))

    x_train = x_train.reshape(-1, 40, 40, 1)
    x_test = x_test.reshape(-1, 40, 40, 1)

    y_train, y_test = dense_to_one_hot(y_train, n_classes=10), dense_to_one_hot(y_test, n_classes=10)
  elif(dataset=='cifar10'):
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

    y_train, y_test = dense_to_one_hot(y_train, n_classes=10), dense_to_one_hot(y_test, n_classes=10)
  elif(dataset=='cifar100'):
    (x_train, y_train), (x_test, y_test) = tf.contrib.keras.datasets.cifar100.load_data()

    y_train, y_test = dense_to_one_hot(y_train, n_classes=100), dense_to_one_hot(y_test, n_classes=100)
  else:
    raise ValueError('dataset not found')
  return (x_train, y_train), (x_test, y_test)

def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels = np.array(labels)
    n_labels = labels.shape[0]
    index_offset = np.arange(n_labels) * n_classes
    labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot
