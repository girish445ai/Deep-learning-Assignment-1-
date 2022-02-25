# Deep-learning-Assignment-1-
CS6910 Assignment
## Authors: R.S.V.GIRISH (EE21S115) , R.JYOTHIRADITYA (CH21SO23)
-----------------------------------------------------------------------------------------------------------
### Problem Statement
In this assignment we need to implement a feedforward neural network and write the backpropagation code for training the network. This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

### Explanation of the Project :
The github repository contains the code for assignmnent_1 of CS6910 which is to build a Feed Forward Neural Network to train and classify the images with respect to their labels in Fashion_Mnist dataset.

The entire code can be divided into major parts which are the functions *ForwardPropagation , Backpropagation , Train* which use some other functions like optimizers ,accuracy etc. which are definded in the code.

The variables used in our code are these : 

**VARIABLES** :

inputnode : The number of neurons in the input layer. 

hiddennodes : The number of neurons in each hidden layer.

outputnode : The number of neurons in the output layer.

hiddenlayers : The number of hidden layers.

num_epochs : The number of epochs to run this code by.

learning_rate : The learning rate you want to use.

batch_size : The size of each mini batch you want to use.

train_set : The number of images from the training set you want to use for training.
    In this case, we are using 90% of the training data,
    so train_set = 90% of 60,000 = 54,000.

activation_func : The activation function for training.
         Can be
        'sigmoid' for sigmoid activation function
        'tanh' for tanh activation function.
        'relu' for relu activation function.

loss_func : The loss function for training.
         Can be
        'squared_error' for Mean squared error calculation
        'cross_entropy' for Cross entropy error calculation.

W : Dictionary to store the weights.\
b : Dictionary to store the biases.\
a : Dictionary to store the pre-activation.\
h : Dictionary to store the activation.

**Initializers**:
init(W, b, inputnode, hiddenlayers, hiddennodes, outputnode, initializer) :- The initialize function, which will initialize all the
    weights randomly according to a distribution. All biases are initialized as 0 in our code.
    Takes 7 arguments, all of them as their name suggests.
We have defined two types of initializers for initialising the weights and biases.
1. random initialisation
2. Xavier initialisation

**FUNCTIONS** : 

sigmoid(x) :- The sigmoid function, returns the the value of sigmoid(x).

d_sigmoid(x) :- Returns the derivative of the sigmoid function. 

d_tanh(x) :- Derivative of the tanh function.

relu(x) :- Returns the max(0,x).

drelu(x):- Returns the value of the derivative of the relu function.

g(ai, hiddennodes, activation_func) :- The activation function (ex. sigmoid, tanh,relu),
    Takes 3 arguments - the pre-activation input, length of the layer, and the activation function.

g1(ai, hiddennodes, activation_func) :- The derivate of the activation function (ex. sigmoid, tanh),
    Takes 3 arguments - the pre-activation input, length of the layer, and the activation function.

o(al, outputnode) :- The output function. Takes 2 arguments - the output pre-activation and the length of the output layer. 

e(i, len) :- one hot vector .

**forward_propagation(x_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, L, activation_func)** :-
    Runs the forward propogation for the given index L of the training data x_train. 
    Takes 11 arguments, all of them as their name suggests. Returns the final probability distribution for each class label.

**back_prop(y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, VW, Vb, L, yhat, activation_func, loss_func)** :-
    Runs the back propogation algorithm for the given Lth index of training data. y_train is the actual output 
    of the training data. y_hat is the final probability distribution we get from forward propogation. Returns 
    the derivate of the loss function w.r.t each of the weights and biases. 

**accuracy(x_test, y_test, x_train, y_train, W, b, hiddenlayers, outputnode, hiddennodes, activation_func)** :- 
    Calculate the accuracy for the current weights and biases. x_test and y_test are the testing data.
    Takes 10 arguments, all of them as their name suggests. Prints the accuracy.
    
**Optimizers**:-
sgd(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func): *Stochastic gradient descent .*

mbgd(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func): *momentum based gradient descent.*

nagd(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func):*nesterov accelerated gradient descent.*

rmsprop(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func):*root mean square propagation*.

adam(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func):*Adam Optimizer*.

nadam(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, train_set, activation_func, loss_func):*Nesterov accelerated adam optimizer*.


**train(x_train, y_train, hiddenlayers, hiddennodes, inputnode, outputnode, W, b, a, h, learning_rate, num_epochs, batch_size, opt, train_set, activation_func, loss_func, initializer):**
Final function which trains the given data according to all the given parameters.
    Takes 18 arguments, all of them as their name suggests.


To run the code, simply set the values to each of the variables inside sweep_config and below that. Then run colab notebook Assignment1_Neural_networks.ipynb to execute the code.
An example configuration is given below : -

sweep_config = {
    'name'  : "Jyothiraditya", 
    'method': 'random', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'hiddenlayers': {
            'values': [3, 5]
        },
        'num_epochs': {
            'values': [10, 15]
        },
        'hiddennodes': {
            'values': [32, 64]
        },
        'learning_rate': {
            'values': [1e-2,5e-3,1e-3]
        },
        'initializer': {
            'values': ["random","Xavier"]
        },
        'batch_size': {
            'values': [32,64]
        },
        'opt': {
            'values': ["sgd","mbgd","nagd","rmsprop","adam","nadam"]
        },
        'activation_func': {
            'values': ["tanh","sigmoid","relu"]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project = "dl_assignment-jyothir")

### Initialize nodes and training set
outputnode = 10
inputnode = 784
train_set = 54000

W = {}

b = {}

a = {}

h = {}

loss_func = "cross_entropy"

### execute the training and validation functions
Function execute():
Trains the model and plots the outputs for all the parameters in wandb that are mentioned in the sweep config .

*To run for multiple values of any of the hyper parameters in the sweep_config, simply append those values to the values in each of the items in parameters inside sweep_config and run wandb.agent(sweep_id, execute)*
