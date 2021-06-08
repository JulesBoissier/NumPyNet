import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import numpy as np
import pygame as pg

from neural_net import NeuralNetwork
from mnist_ui import Board, run_mnist_app
from mnist_utils import get_mnist_data, one_hot_encode


def prepare_mnist_dataset(
    train_size : int = 60000,
    test_size : int = 10000
    ) -> list:
    """
    Splits the dataset into a training set and a test set.
    Additionally, one hot encodes the training labels to be used as input for the neural net.

    Parameters
    ----------
    train_size: number of elements in the training set. (Max 60 000)
    test_size: number of elements in the testing set. (Max 10 000)

    Returns
    ----------
    train_x: A (train_size by 784) np.ndarray containing the features of the training set.
    train_y: A (train_size by 10) np.ndarray containing the one hot encoded labels for the training set.
    test_x: A (test_size by 784) np.ndarray containing the features of the test set.
    test_y: A (test_size by 1) np.ndarray containing the labels, not one hot encoded, of the test set.
    """

    train_x, train_y, test_x, test_y = get_mnist_data(train_size, test_size)
    train_y = one_hot_encode(train_y)

    return train_x, train_y, test_x, test_y

def train_and_evaluate_new_network(
    train_x : np.ndarray,
    train_y : np.ndarray, 
    test_x : np.ndarray,
    test_y : np.ndarray,
    architecture : list = [784,128,10],
    epochs: int = 10,
    learning_rate: float = 0.015,
    validation_ratio: float = 0.1,
    plot_loss : bool = True,
    name :str = 'new_model',
    ):
    """
    Builds a new fully connected neural network and trains it on the given dataset.
    Plots the training and validation at every epoch of the training.
    Evaluates the accuracy of the model on the test set, as well as the recall, precision and F1 score for each class.
    Saves the weights and biases of the model in a pkl file.

    Parameters
    ----------
    train_x: A np.ndarray containing the features of the training set.
    train_y: A np.ndarray containing the one hot encoded labels for the training set.
    test_x: A np.ndarray containing the features of the test set.
    test_y: A np.ndarray containing the labels, not one hot encoded, of the test set.
    architecture: A list of integers defining the number of nodes at each layer. (Input and output should fit dataset shape.)
    epochs: An integer denoting the number of training cycles over the dataset.
    learning_rate: A float denoting the step size for iterations of the gradient descent during network training.
    validation_ratio: A float denoting the ratio of the split of the training set to create a validation set.
    plot_loss: A bool deciding wether or not to plot the loss during training.
    name: A string denoting the name of the pkl file for the saved model, which is later used to load a model.
    """
    Net = NeuralNetwork(architecture)
    Net.train(train_x,train_y,epochs,learning_rate,validation_ratio,plot_loss)
    Net.evaluate(test_x,test_y)
    Net.save_model(name)
        
def load_and_evaluate_model(
    test_x : np.ndarray,
    test_y : np.ndarray,
    name :str = 'new_model',
    evaluate : bool = True, 
    ):
    """
    Creates a new neural net and loads weights and biases from a previously trained model.
    Evaluates the accuracy of the model on the test set, as well as the recall, precision and F1 score for each class.

    Parameters
    ----------
    test_x: A np.ndarray containing the features of the test set.
    test_y: A np.ndarray containing the labels, not one hot encoded, of the test set.
    name: A string denoting the name of the pkl file to be loaded. (Must be saved in the same folder as code.)
    evaluate: A bool denoting wether to evaluate the loaded model or not.
    """

    Net = NeuralNetwork()
    Net.load_model(name)

    if evaluate:
        Net.evaluate(test_x,test_y)

if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_mnist_dataset(train_size = 500, test_size = 1000)
    train_and_evaluate_new_network(train_x,train_y,test_x,test_y)

    # load_and_evaluate_model(test_x,test_y,name = 'ui_classifier')

