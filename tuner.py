from typing import List
import numpy as np
import copy

from neural_net import NeuralNetwork

def build_models_for_tuner(
    train_x : np.ndarray,
    train_y : np.ndarray, 
    architectures : List[List[int]],
    activation_functions : List[str],
    ) -> List[NeuralNetwork]:
    """
    Given lists of architectures and activation_functions, creates a list of instances of the NeuralNetwork class,
    for all possible combinations.
    The input and output layers are not included in the architecture input and are instead added in the function,
    to fit the training set.

    Parameters
    ----------
    train_x: A np.ndarray containing the features of the training set.
    train_y: A np.ndarray containing the one hot encoded labels for the training set.
    architectures: A list of network architectures, only for the hidden layers.
    activation_functions: A list of activation function names. 
                          Valid options : 'relu', 'leaky_relu','hyperbolic_tangent','elu'

    Returns
    ----------
    model_list: A list of intances of the NeuralNetwork class.
    """

    input_layer = train_x.shape[1]
    output_layer = train_y.shape[1]

    for architecture in architectures:
        architecture.insert(0,input_layer)
        architecture.append(output_layer)

    model_list= []

    for activation_function in activation_functions:
        model_sub_list = [NeuralNetwork(architecture, activation_function) for architecture in architectures]
        model_list += model_sub_list

    return model_list

def grid_train_models(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    model_list: List[NeuralNetwork],
    epochs_list: List[int],
    learning_rates: List[float],
    validation_ratio: float,
    plot_loss: bool,
    display_table: bool
    ) -> list:
    """
    Given lists of architectures and activation_functions, creates a list of instances of the NeuralNetwork class,
    for all possible combinations.
    The input and output layers are not included in the architecture input and are instead added in the function,
    to fit the training set.

    Parameters
    ----------
    train_x: A np.ndarray containing the features of the training set.
    train_y: A np.ndarray containing the one hot encoded labels for the training set.
    test_x: A np.ndarray containing the features of the test set.
    test_y: A np.ndarray containing the labels, not one hot encoded, of the test set.
    model_list: A list of intances of the NeuralNetwork class.
    epochs_list: A list of epochs to define the number of cycles of the training for each model.
    learning_rates: A list of learning rates to define the step sizes for the gradient descent.
    validation_ratio: A float denoting the ratio of the split of the training set to create a validation set.
    plot_loss: A bool deciding wether or not to plot the loss during training.
    display_table: A bool deciding wether to plot the analysis table for each trained model.

    Returns
    ----------
    best_model: An instance of the NeuralNetwork class which obtained the highest accuracy on the test set.
    best_accuracy: The highest accuracy obtained by a model during the tuning phase.
    best_learning_rate: The learning rate used to train the best model.
    best_epoch_length: The number of training cycles used for the best model.
    """


    best_accuracy = 0

    for epochs in epochs_list:
        for learning_rate in learning_rates:
            current_model_list =  copy.deepcopy(model_list)
            
            for model in current_model_list:
                
                model.train(train_x,train_y,epochs,learning_rate,validation_ratio,plot_loss)
                accuracy = model.evaluate(test_x,test_y, display_table)

                if accuracy > best_accuracy:
                    best_model = model
                    best_accuracy, best_learning_rate,best_epoch_length = accuracy, learning_rate, epochs
    
    return best_model, best_accuracy, best_learning_rate, best_epoch_length