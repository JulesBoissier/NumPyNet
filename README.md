# NumPyNet

A fully-connected neural network built mainly in NumPy complemented by a digit classifying app.

---

### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [Basic Underlying Theory](#basic-underlying-theory)
- [References](#references)

---

## Description  

### Fully-Connected Neural Network: [(code)](neural_net.py)

Using only the numpy package as a backbone, this project aims to create an easily useable neural network with a wide range of features. As a preface, it should be noted that if this was the sole aim, it would've been a colossal waste of time, as this can be achieved in four to five lines of code using a package such as tensorflow. In actual fact, the underlying goal consisted in challenging myself to create a clean and functional code, as well as verifying that I was familiar enough with the theory required to build such a network.

The NeuralNetwork class comes with the necessary functions to train, predict and evaluate its performance on a test dataset. Additionally, it gives the user to save and load a model's weights and biases.

This project comes with a MNIST dataset as well as a previously trained model which obtained an accuracy of 97%. It can however be used with any other dataset that would be suitable for a fully-connected neural network.

### Digit Classifying App:  [(code)](mnist_ui.py)

As a means to showcase the ability of the aforementioned, using a pre-trained model on the MNIST dataset, this pygame app allows the user to draw a digit and have the app attempt to classify it. The results are fairly satisfactory, helped by the fact that the script attempts to preprocess the hand-drawn inputs in much the same way as the original MNIST dataset was. Namely, the input data is cropped, centered and grey-scaled and the padded by a box size of four.

Additionally, the app outputs a confidence probability in its prediction, for which the value is calibrated by the softmax temperature scaling for which the computation is  detailed below.

---

## How To Use


The neural network class [(code)](neural_net.py) built contains five out of class functions (train, predict, evaluate, save_model and load_model), for which the names appear to be self-explanatory. It should be noted that the evaluate function prints a table (using the tabulate package) displaying the precision, recall and F1 score for each class.

The utils file [(code)](mnist_utils.py) contains a few functions which are used to extract the dataset from the pkl file, and one hot encode the labels.

Three  main functions [(code)](mnist_main.py) are defined to prepare the MNIST dataset and use the main features of the neural network:

> prepare_mnist_dataset 

Extracts the dataset from the mnist.pkl.gz file, splits it into a training set and test set of desired sizes and one hot encodes the training labels. The output of this function are made to be used as inputs for the NeuralNetwork class.

> train_and_evaluate_new_network

Creates a new custom fully connected network, trains and evaluates it on the input data and saves the model's weights and biases in a .pkl file in the folder.
Requires training labels to be one hot encoded but not the test labels.
It should be noted that the network's input and output layer will be reshaped if they do not match the training data's format.

> load_and_evaluate_model

Creates a new fully-connected neural network and loads its architecture, weights and biases from a saved model in the project folder. Also gives the option to evaluate the networks performance on a test set.
 



## Basic Underlying Theory

As mentionned in the project description, no groundbreaking theory will be found in this project. I would however like to detail a few blips in the theory that I had not heard of before researching for this particular project.

In asking my neural networks to print the probability of their predictions as an output to the digit classifying app when the user submits a hand-drawn digit I came to the realization that these models were atrociously overconfident. This was surprising as I was under the impression that the softmax would do a good job at estimating a degree of confidence, and seemed to do so on the MNIST dataset.

However, thanks to the app, I was able to do more test with non-MNIST inputs, and while the predictions for digits were mostly correct, it would also classifiy non-digits inputs, such as almost empty or completely black images, with a very high degree of confidence (+ 99%).
  
Thankfully, a paper was written on this very subject [1], giving several strategies for calibrating neural networks. The _confidence_calibration function in the NeuralNetwork class therefore optimizes temperature coefficient through gradient descent on the cross-entropy loss of the validation set. Afterwards, the multi-class output is computed as the softmax of the network output over the temperature, rather than just the softmax of the network output. This effectively dampens the softmax function and skews the confidence of the network down.


## References
[1] Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger, 2017, 'On Calibration of Modern Neural Networks'.
      
---

[Back To The Top](#numpynet)
