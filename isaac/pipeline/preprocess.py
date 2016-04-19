import numpy as np


__all__ = [
    "mean_normalizer",
    "get_XY_from_frame"
]


def mean_normalizer(arr):
    '''
    Update the column in-place using mean normalize.
    '''
    for i in range(arr.shape[1]): # column by column
        column = arr[:, i]
        mean_f, max_f = np.mean(column), np.max(column)
        arr[:, i] = (column - mean_f) / max_f # mean normalization

def get_XY_from_frame(arr, columns, outcome=-1,
        normalize='mean', add_bias=True):
    '''
    arr
        np.narray, the training data
    
    columns
        list, list of indexes of input
    
    outcome
        int, the outcome column, defaults to the last column
    
    return
        tuple, (X, Y)
        
    '''
    len_data = arr.shape[0]
    Y = arr[:, outcome]
    X = arr[:, columns]
    normalizer = normalize + "_normalizer"
    globals()[normalizer](X)
    if add_bias:
        X = np.hstack((np.ones((len(Y), 1)), X)) # adding one bias column
    return (X, Y)

def load_training_mnist(path_mnist):
    from mnist import MNIST
    mnist = MNIST(path_mnist)
    images, labels = mnist.load_training()
    return (images, labels)

def load_testing_mnist(path_mnist):
    from mnist import MNIST
    mnist = MNIST(path_mnist)
    images, labels = mnist.load_testing()
    return (images, labels)

def preprocess_mnist(images, labels):
    Xs = np.array(images, dtype='float64')
    scaling = 1/255 # so that color intensities range from 0 to 1
    Xs *= scaling
    Ys = np.zeros((len(labels), 10), dtype='uint8')
    Ys[range(len(labels)),labels] = 1
    return (Xs, Ys)

def training_mnist_preprocessed(path_mnist):
    images, labels = load_training_mnist(path_mnist)
    return preprocess_mnist(images, labels)

def testing_mnist_preprocessed(path_mnist):
    images, labels = load_testing_mnist(path_mnist)
    return preprocess_mnist(images, labels)

def n_2_bitmap(n):
    '''
    Represents a label in MNIST dataset with
    a bitmap. e.g., 5 -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    return
        np.array
    '''
    bits = np.zeros(10)
    if n != 0:
        bits[n] = 1
    return bits
