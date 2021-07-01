import pickle, gzip, numpy as np
import math


import pandas as pd


    
def read_pickle_data(file_name): 
    """Unzips and reads the pkl dataset."""
    f = gzip.open(file_name, 'rb')
    data = pickle.load(f, encoding='latin1')
    f.close()
    return data

def load_train_and_test_pickle(file_name):
    """Unpacks the pkl datasets."""
    train_x, train_y, test_x, test_y = read_pickle_data(file_name)
    return train_x, train_y, test_x, test_y

def get_mnist_data(train_size = 1000, test_size = 1000):
    """
    Transforms the pkl dataset into four np.ndarrays: features and labels for a training and testing set.
    Additionally, gives the option to reduce the set sizes by deleting a portion of it.
    """
    train_set, valid_set, test_set = read_pickle_data('./mnist.pkl.gz')
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    train_x = np.vstack((train_x, valid_x))
    train_y = np.append(train_y, valid_y)
    test_x, test_y = test_set

    #Deletes part of the dataset if smaller one is needed.
    train_x = np.delete(train_x,np.s_[train_size:60000],0)
    train_y = np.delete(train_y,np.s_[train_size:60000])
    test_x  = np.delete(test_x,np.s_[test_size:10000],0)
    test_y  = np.delete(test_y,np.s_[test_size:10000])

    return (train_x, train_y, test_x, test_y)

def one_hot_encode(labels):
    """
    One_hot_encodes a np.ndarray of labels for multi-class ml.
    """
    num_classes = len(np.unique(labels))
    encoded_labels = np.zeros((labels.shape[0],num_classes))
    for i in range(labels.shape[0]):
        encoded_labels[i][labels[i]] = 1
    return encoded_labels

def relu(x):
        return max(x,0)

def relu_deriv(x):
    if x <= 0:
        return 0
    else:
        return 1

def leaky_relu(x,alpha = 0.01):
    if x < 0:
        return alpha*x
    else:
        return x

def leaky_relu_deriv(x,alpha = 0.01):
    if x < 0:
        return alpha
    else:
        return 1

def hyperbolic_tangent(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)) 

def hyperbolic_tangent_deriv(x):
    return 1 - hyperbolic_tangent(x)**2

def elu(x,alpha = 0.01):
    if x <= 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x

def elu_deriv(x,alpha = 0.01):
    if x <= 0:
        return alpha * math.exp(x)
    else:
        return 1

def exponantial(x):
    return math.exp(x)