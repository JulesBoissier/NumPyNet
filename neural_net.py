import math
import pickle
import copy

import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
from mnist_utils import *



class NeuralNetwork:

    """ Builds, trains and uses a custom fully connected neural network."""

    def __init__(
        self,
        architecture : list = [784,128,10],
        activation_function : str = 'relu'
        ):

        if isinstance(architecture,list) == False or len(architecture) < 2 or 0 in architecture:     
            raise TypeError("The first input for the neural network must be a list of length superior to two. Each entry representing the number of nodes on that layer.")
        self.architecture = architecture 
        self.weights = []
        self.bias = []
        self.layers = len(self.architecture)
        self.temperature = 1
        for i in range(self.layers-1):
            #He Weight Initialization
            self.weights.append(np.random.normal(0,np.sqrt(2/self.architecture[i]),(self.architecture[i+1],self.architecture[i])))
            self.bias.append(np.zeros((self.architecture[i+1],1)))

        self._assign_actfunc(activation_function)
        

    def _assign_actfunc(
        self,
        name : str
        ) -> None:
        """
        Assigns a specific activation function to be used by the neural network.

        Parameters
        ----------
        name: A string denoting the activation function to be used.
              Valid options : 'relu', 'leaky_relu','hyperbolic_tangent','elu'
        """
        self.activation_name = name
        if name == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_deriv
        elif name == 'leaky_relu':
            self.activation_function = leaky_relu
            self.activation_derivative = leaky_relu_deriv
        elif name == 'hyperbolic_tangent':
            self.activation_function = hyperbolic_tangent
            self.activation_derivative = hyperbolic_tangent_deriv
        elif name == 'elu':
            self.activation_function = elu
            self.activation_derivative = elu_deriv

    def _forward_prop(
        self,
        element : np.ndarray,
        ) -> list:
        """
        Propagates an input array forwards through the neural network, yielding a softmax classification.

        Parameters
        ----------
        element: Horizontal array describing the input element. (Width should match input layer height)

        Returns 
        ----------
        softmax_output: Vertical array containing the softmax classification output by the network. 
        z_list: List of vertical arrays describing neuron states for each layer.
        a_list: List of vertical arrays describing neuron's activated states for each layer.
        """

        act_function = np.vectorize(self.activation_function)
        expon = np.vectorize(exponantial)
        z_list = []
        a_list = []
        a = element.reshape(element.shape[0],-1)
        a_list.append(a)

        for k in range(self.layers-1):
            
            z = np.dot(self.weights[k],a) + self.bias[k]
            a = act_function(z)
            z_list.append(z)
            a_list.append(a)

        output = expon(a)
        softmax_output = output/sum(output)
        
        
        return softmax_output, z_list, a_list

    def _backward_prop(
        self,
        label : np.ndarray,
        softmax_prediction : np.ndarray,
        z_list : list,
        ) -> list:
        """
        Backpropagates the error through the network to obtain the error associated to each neuron.

        Parameters
        ----------
        label: Ground truth value one hot encoded in a vertical array.
        softmax_prediction: Vertical array containing the network prediction in the form of softmax prediction. 
        z_list: List of vertical arrays describing neuron states for each layer.

        Returns 
        ----------
        delta_list: List of vertical arrays containing the partial contribution to the error for each layer.
        softmax_loss: A float denoting the softmax loss for this element's prediction.
        """

        act_derivative = np.vectorize(self.activation_derivative)
        label = label.reshape(label.shape[0],-1)
        delta_list = []
        delta = (softmax_prediction - label)
        softmax_loss = - np.sum(np.multiply(label,np.log(softmax_prediction)))
        delta_list.append(delta)

        for k in reversed(range(1,self.layers-1)):
            derivative = act_derivative(z_list[k-1])
            matmul = np.matmul((np.transpose(self.weights[k])),delta_list[0])
            delta = np.multiply(matmul,derivative) 
            delta_list.insert(0,delta) 
        return delta_list, softmax_loss

    def _update_weights(
        self,
        delta_list : list,
        a_list : list, 
        learning_rate : float = 0.15,
        ) -> None:
        """
        Applies gradient descent to the weights and biases of the neural network.

        Parameters
        ----------
        delta_list: List of vertical arrays containing the partial contribution to the error for each layer.
        a_list: List of vertical arrays describing neuron's activated states for each layer.    
        """
        #maybe add momentum to grad descent
        for k in range(self.layers-1):
                    grad = np.matmul(delta_list[k],np.transpose(a_list[k]))
                    self.weights[k] -= learning_rate * grad
                    self.bias[k] -= learning_rate * delta_list[k]

    def _train_validation_split(
        self,
        train_x : np.ndarray,
        train_y : np.ndarray,
        validation_ratio: float,
        ) -> list:
        """
        Splits the training set into a training set and a validation set.

        Parameters
        ----------
        train_x: An array of element features.
        train_y: An array of onehot encoded labels corresponding to the elements in the train_x array.
        validation_ratio: A float denoting the ratio in size of the validation set compared to the train_x set.

        Returns
        ----------
        train_x, train_y : Two arrays containing the training set.
        valid_x, valid_y : Two arrays containing the validation set.
        """
        set_size = len(train_x)
        train_size = int(set_size * (1 - validation_ratio))
        shuffler = np.random.permutation(set_size)

        shuffled_x = train_x[shuffler]
        shuffled_y = train_y[shuffler]

        train_x, valid_x = shuffled_x[:train_size:], shuffled_x[train_size:]
        train_y, valid_y = shuffled_y[:train_size:], shuffled_y[train_size:]

        return train_x, train_y, valid_x, valid_y

    def _compute_loss(
        self,
        element : np.ndarray,
        label : np.ndarray,
        ) -> float:
        """
        Computes the cross entropy loss for a single prediction/label pair.
        
        Parameters
        ----------
        element: An array containing the features of an element
        label: An array containing a onehot encoded label

        Returns
        ----------
        loss: A float denoting the cross entropy loss.
        """

        label = label.reshape(label.shape[0],-1)
        softmax_prediction = self._forward_prop(element)[0]
        loss = - np.sum(np.multiply(label,np.log(softmax_prediction)))
        return loss

    def _compute_validation_loss(
        self,
        valid_x : np.ndarray,
        valid_y : np.ndarray,
        ) -> float:

        validation_epoch_loss = 0
        for i in range(valid_x.shape[0]):
            validation_loss = self._compute_loss(valid_x[i],valid_y[i])
            validation_epoch_loss += validation_loss / valid_x.shape[0]
        
        return validation_epoch_loss

    def _plot_loss(
        self,
        training_loss : list,
        validation_loss : list,
        ) -> None:
        """
        Plots a scatter plot of the training loss as a function of the number of epochs. Also plots the validation loss if the 
        validation set is not empty.

        Parameters
        ----------
        training_loss: A list of floats denoting the loss for each epoch.
        validation_loss: A list of floats denoting the loss for each epoch.
        """
        plt.plot(training_loss,label = 'training loss')
        if validation_loss != []:
            plt.plot(validation_loss, label = 'validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Cross Entropy Loss per Epoch')
        plt.xticks([i for i in range(len(training_loss))])
        plt.legend()
        plt.show()
   
    def _shape_verification(
        self,
        train_x : np.ndarray,
        train_y : np.ndarray,
        ) -> None:
        """
        Checks that the input layer matches the number of features and the output layers matches number of classes.
        If not, reshapes the input and/or output layers to match the training data.
        """
        if train_x.shape[1] != self.architecture[0] or train_y.shape[1] != self.architecture[-1]:

            self.architecture[0] = train_x.shape[1]
            self.architecture[-1] = train_y.shape[1]

            #He Weight reinitialization for first weight and bias matrices
            self.weights[0] = (np.random.normal(0,np.sqrt(2/self.architecture[0]),(self.architecture[1],self.architecture[0])))
            self.bias[0] = (np.zeros((self.architecture[1],1)))

            #He Weight reinitialization for last weight and bias matrices
            self.weights[-1] = (np.random.normal(0,np.sqrt(2/self.architecture[-2]),(self.architecture[-1],self.architecture[-2])))
            self.bias[-1] = (np.zeros((self.architecture[-1],1)))

            print('Network architecture was reshaped to shape', self.architecture, 'to fit training data.') 

    def train(
        self,
        train_x : np.ndarray,
        train_y : np.ndarray, 
        epochs: int = 10,
        learning_rate: float = 0.015,
        validation_ratio: float = 0,
        plot_loss : bool = False,
        ) -> None:   
        """
        Trains the neural network on the given dataset (i.e. updates the weights and biases to reduce error).
        Additionally, reshapes the network architecture if it doesn't match dataset's input or output dimensions. 

        Parameters
        ----------
        train_x: Array containing training elements. 
        train_y: Array containing one hot encoded labels matching training elements.
        epochs: Number of training cycles, integer.
        learning_rate: Learning rate for the gradient descent algorithm.
        validation_ratio: A float denoting the ratio in size of the validation set compared to the train_x set.
        plot_loss: A bool defining wether the loss during training is plotted or not.
        """
        train_x, train_y, valid_x, valid_y = self._train_validation_split(train_x,train_y,validation_ratio)
        self._shape_verification(train_x,train_y)
        training_loss_list = [] 
        validation_loss_list = []

        for i in range(epochs):
            training_epoch_loss = 0
            for j in range(train_x.shape[0]):
                softmax_output, z_list, a_list = self._forward_prop(train_x[j])
                delta_list, training_loss = self._backward_prop(train_y[j], softmax_output, z_list)
                self._update_weights(delta_list, a_list, learning_rate) #need this to only be done for every minibatch
                
                training_epoch_loss += training_loss / train_x.shape[0]
            
            training_loss_list.append(training_epoch_loss)

            if validation_ratio > 0:
                validation_epoch_loss = self._compute_validation_loss(valid_x,valid_y)
                validation_loss_list.append(validation_epoch_loss)
        
        self.temperature = self._confidence_calibration(valid_x, valid_y)

        if plot_loss:
            self._plot_loss(training_loss_list, validation_loss_list)

    def _negative_log_likelihood_gradient(
        self,
        logit_vector : np.ndarray,
        class_value : int,
        number_of_classes : int,
        temperature : float,
        ):
        """
        Computes the gradient of the NLL (cross entropy loss) function for the softmax of (z/temperature) in which z is the 
        logit vector. Gradient will then be used in function _confidence_calibration for gradient descent to optimize w.r.t Z.

        Parameters
        ----------
        logit_vector: An array, containing the output of the neural network before the softmax function is applied.
        class_value: An integer, denoting the number associated to the class for the ground truth associated to this specific prediction.
        number_of_classes: An integer, denoting the total number of classes.
        temperature: A float representing a metric for the degree of confidence in predictions.

        Returns
        ----------
        gradient: A float denoting the gradient of the cross entropy loss with respect to the temperature.
        """
        numerator = 0
        denominator = 0
        for i in range(10):
            if i != class_value:
                numerator += logit_vector[class_value] * math.exp(logit_vector[i]/temperature) - logit_vector[i] * math.exp(logit_vector[i]/temperature)
        
        for i in range(10):
            denominator += math.exp(logit_vector[i]/temperature)
        denominator *= temperature**2

        gradient = numerator/denominator

        return gradient

    def _confidence_calibration(
        self, 
        valid_x : np.ndarray,
        valid_y : np.ndarray,
        epochs : int = 20, 
        learning_rate : float = 0.5
        ) -> float:
        """
        Determines a degree of calibration for the prediction confidence probability through a gradient descent algorithm.
        Without this, network is overconfident in predictions, see 'On Calibration of Modern Neural Networks',
        Chuan Guo et al. 2017.

        Parameters
        ----------
        valid_x: Array containing features of elements, should use same samples as validation set.
        valid_y: Array containing corresponding labels, onehot encoded, to the valid_x array.
        epochs: Number of epochs for the optimzation of the confidence parameter.
        learning_rate: Learning rate for the gradient descent algorithm.

        Returns
        ----------
        temperature: A float representing a metric for the degree of confidence in predictions.
        """
        number_of_classes = valid_y.shape[1]
        temperature = 5

        for epoch in range(epochs):
            for i in range(len(valid_x)):

                class_value = np.argmax(valid_y[i])
                softmax_output, z_list, a_list = self._forward_prop(valid_x[i])
                logit_vector= a_list[-1] 
                gradient = self._negative_log_likelihood_gradient(logit_vector,class_value,number_of_classes,temperature)
                temperature -= gradient * learning_rate
        
        return max(float(temperature),0)

    def predict(
        self,
        test_x : np.ndarray,
        ) -> list:
        """
        Feeds forwards elements through the network and predicts the class.
        Recomputes the calibrated confidence probability for the softmax output

        Parameters
        ----------
        train_x: Array containing elements for which to predict the class.

        Returns:
        ----------
        predictions: Array containing predicted classes for all elements collapsed back into single column representation. 
        probabilities: Array containing estimations for degree of confidence for the predictions
        """
        predictions = np.zeros((test_x.shape[0],1)) 
        probabilities = np.zeros((test_x.shape[0],1))
        expon = np.vectorize(exponantial)
        for i in range(test_x.shape[0]):
                output, z_list, a_list = self._forward_prop(test_x[i])        
                predictions[i] = np.argmax(np.transpose(output))  


                new_a = (a_list[-1]/self.temperature)
                output = expon(new_a)
                probabilities= max(output/sum(output))


        return predictions, probabilities
    
    def evaluate(
        self,
        test_x : np.ndarray,
        test_y : np.ndarray,
        display_table : bool = True,
        ) -> float: 
        """
        Evaluates the accuracy of the neural network on a batch of test elements.
        Yields several metrics to evaluate the performance, including the overall accuracy, TP, TN, FP, FN for each class
        and the precision and recall for each class.

        Parameters
        ----------
        test_x: Array containing test elements. 
        test_y: Array containing one hot encoded labels matching test elements.

        Returns
        ----------
        The overall accuracy of the neural network on the test set
        """
        predictions, probabilities = self.predict(test_x)
        count_errors = np.count_nonzero(np.transpose(predictions)-test_y)
        

        if display_table:

            print('The  overall accuracy is', round(1 - count_errors/test_y.shape[0],3),'.')

            unique_classes = np.unique(test_y)
            test_y = test_y.reshape(test_y.shape[0],-1)
            zipped_arrays = np.concatenate((test_y,predictions),axis = 1)

            analysis_table = [[unique_class] for unique_class in unique_classes]

            #Analysing each class separatly:
            for idx, unique_class in enumerate(unique_classes):
                class_list = []
                non_class_list = []
                for i in range(zipped_arrays.shape[0]):
                    if zipped_arrays[i][0] == unique_class:
                        class_list.append(zipped_arrays[i][1])
                    else:
                        non_class_list.append(zipped_arrays[i][1])

                tp = np.count_nonzero(class_list == unique_class)
                fn = len(class_list) - tp
                fp = np.count_nonzero(non_class_list == unique_class)
                tn = len(non_class_list) - fn
                
                try:
                    precision = round(tp/(tp + fp),3)
                    recall = round(tp/(tp + fn),3)
                    f1_score = round(2 * (precision * recall) / (precision + recall),3)
                except ZeroDivisionError:
                    precision = 'N/A'
                    recall = 'N/A'
                    f1_score = 'N/A'

                analysis_table[idx].extend([precision, recall, f1_score])

            print(tabulate(analysis_table, headers=['Class','Precision', 'Recall','F1 Score']))
        
        return 1 - count_errors/test_y.shape[0]
    
    def save_model(
        self,
        name :str = 'new_model',
        ) -> None:
        """
        Saves a dictionary of the weights and biases of the model in a .pkl file.
        """
        model_dict = {}
        model_dict['Temperature'] = self.temperature
        model_dict['Activation'] = self.activation_name
        for idx, element in enumerate(self.bias):
            model_dict[f'Bias{idx}'] = pd.DataFrame(data=element)
        for idx, element in enumerate(self.weights):
            model_dict[f'Weight{idx}'] = pd.DataFrame(data=element)
        with open(f'{name}.pkl', 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _reset_network_parameters(self) -> None:
        """ 
        Resets several network parameters, to be later reinitialized in the load_model function.
        """
        self.architecture = []
        self.weights = []
        self.bias = []

    def load_model(
        self,
        name : str = 'new_model',
        ) -> None:
        """
        Loads the model from the .pkl document created by the self.save_model function.
        """
        self._reset_network_parameters()

        with open(f'{name}.pkl', 'rb') as handle:
            model = pickle.load(handle)

        for layer_name, parameter in model.items():
            
            if layer_name.startswith('Bias'):
                value = parameter.to_numpy()
                self.bias.append(value)
            elif layer_name.startswith('Weight'):
                value = parameter.to_numpy()
                self.weights.append(value)
            elif layer_name ==  'Temperature':
                self.temperature = parameter
            elif layer_name == 'Activation':
                activation_function_name = parameter

        for weight in self.weights:
            self.architecture.append(weight.shape[1])
        self.architecture.append(self.weights[-1].shape[0])

        self._assign_actfunc(activation_function_name)

