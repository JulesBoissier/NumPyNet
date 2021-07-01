import os

import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame as pg
import copy
from typing import List

from neural_net import NeuralNetwork
from tuner import *
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
    activation_function : str ='relu',
    epochs: int = 10,
    learning_rate: float = 0.015,
    validation_ratio: float = 0.1,
    evaluate : bool = True,
    plot_loss : bool = True,
    name :str = 'new_model',
    ) -> None:
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
    evaluate : A bool denoting wether to evaluate the model's performance on a test set.
    plot_loss: A bool deciding wether or not to plot the loss during training.
    name: A string denoting the name of the pkl file for the saved model, which is later used to load a model.
    """
    Net = NeuralNetwork(architecture, activation_function)
    Net.train(train_x,train_y,epochs,learning_rate,validation_ratio,plot_loss)
    Net.save_model(name)

    if evaluate:
        Net.evaluate(test_x,test_y)
            
def load_and_evaluate_model(
    test_x : np.ndarray,
    test_y : np.ndarray,
    name :str = 'new_model',
    evaluate : bool = True,
    ) -> None:
    """
    Creates a new neural net and loads weights and biases from a previously trained model.
    Evaluates the accuracy of the model on the test set, as well as the recall, precision and F1 score for each class.

    Parameters
    ----------
    test_x: A np.ndarray containing the features of the test set.   
    test_y: A np.ndarray containing the labels, not one hot encoded, of the test set.
    name: A string denoting the name of the pkl file to be loaded. (Must be saved in the same folder as code.)
    evaluate: A bool denoting wether to evaluate the model's performance on a test set.
    """

    Net = NeuralNetwork()
    Net.load_model(name)

    if evaluate:
        Net.evaluate(test_x,test_y)

def hyper_parameter_tuner(
    train_x : np.ndarray,
    train_y : np.ndarray, 
    test_x : np.ndarray,
    test_y : np.ndarray,
    architectures : List[List[int]] = [[128],[128,256]],
    activation_functions : List[str] = ['leaky_relu'],
    epochs_list: List[int] = [10],
    learning_rates: List[float] = [0.015],
    validation_ratio: float = 0.1,
    plot_loss : bool = False,
    display_table : bool = False,
    save_best_model : bool = False,
    name : str = 'best_model',
    ) -> NeuralNetwork :
    """
    Creates, trains and evaluates models from a list of hyperparameters to find which combination yields the best
    accuracy.

    Parameters
    ----------
    train_x: A np.ndarray containing the features of the training set.
    train_y: A np.ndarray containing the one hot encoded labels for the training set.
    test_x: A np.ndarray containing the features of the test set.
    test_y: A np.ndarray containing the labels, not one hot encoded, of the test set.
    architectures: A list of network architectures, only for the hidden layers.
    activation_functions: A list of activation function names. 
                          Valid options : 'relu', 'leaky_relu','hyperbolic_tangent','elu'
    epochs_list: A list of epochs to define the number of cycles of the training for each model.
    learning_rates: A list of learning rates to define the step sizes for the gradient descent.
    validation_ratio: A float denoting the ratio of the split of the training set to create a validation set.
    evaluate : A bool denoting wether to evaluate the model's performance on a test set.
    plot_loss: A bool deciding wether or not to plot the loss during training.
    save_best_model: A bool to define wether to save the best model or not.
    name: A string denoting the name of the pkl file for the saved model, which is later used to load a model.

    Returns
    ----------
    best_model: The instance of the NeuralNetwork which yielded the best accuracy.
    """
    model_list = build_models_for_tuner(train_x,train_y,architectures,activation_functions)

    best_model, accuracy, learning_rate, epoch = grid_train_models(
        train_x,train_y,test_x,test_y,model_list,epochs_list,learning_rates,validation_ratio,plot_loss,display_table
        )

    best_model.evaluate(test_x,test_y, display_table = True)

    print((f'The best model obtained an accuracy {100 * accuracy}%, with an architecture of {best_model.architecture},' 
           f'a {best_model.activation_name} activation function, and by being trained for {epoch} epochs with a '
           f'learning rate of {learning_rate}.'))

    

    if save_best_model:
        best_model.save_model(name)

    return best_model

if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_mnist_dataset(train_size = 10000, test_size = 10000)
    train_and_evaluate_new_network(train_x,train_y,test_x,test_y,activation_function = 'elu')
    #hyper_parameter_tuner(train_x, train_y, test_x, test_y)
    #load_and_evaluate_model(test_x,test_y,name = 'ui_classifier')

