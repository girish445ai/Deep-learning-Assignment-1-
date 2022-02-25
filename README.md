# Deep-learning-Assignment-1-
CS6910 Assignment
## Authors: R.S.V.GIRISH (EE21S115) , R.JYOTHIRADITYA (CH21SO23)
-----------------------------------------------------------------------------------------------------------
### Link to the project wandb report:


### Explanation of the Project :
The github repository contains the code for assignmnent_1 of cs6910 which is to build 


The functions and various variables used in our code are these : 

VARIABLES :

num_input : The number of neurons in the input layer. Type : INT.

num_neurons : The number of neurons in each hidden layer. Type : INT.

num_output : The number of neurons in the output layer. Type : INT.

num_hlayers : The number of hidden layers. Type : INT.

num_epochs : The number of epochs to run this code by. Type : INT.

learning_rate : The learning rate you want to use. Type : INT.

batch_size : The size of each mini batch you want to use. Type : INT.

train_set : The number of images from the training set you want to use for training.
    In this case, we are using 90% of the training data,
    so train_set = 90% of 60,000 = 54,000. Type : INT.

opt : The type of optimization you want to use for training. 
    Can be 
        'sgd' for stocastic gradient descent
        'mbgd' for momentum based gradient descent
        'nagd' for nesterov accelerated gradient descent
        'rmsprop' for room mean square propogation
        'adam'  for adam optimizer
        'nadam' for nadam optimizer. 
    Type : STRING.

activation_func : The activation function you want to use for training.
    Can be
        'sigmoid' for sigmoid activation function
        'tanh' for tanh activation function.
    Type : STRING.

loss_func : The loss function you want to use for training.
    Can be
        'squared_error' for Mean squared error calculation
        'cross_entropy' for Cross entropy error calculation.
    Type : STRING.

W : Dictionary to store the weights. Type : Dictionary.
B : Dictionary to store the biases. Type : Dictionary.
A : Dictionary to store the pre-activation. Type : Dictionary.
H : Dictionary to store the activation. Type : Dictionary.

FUNCTIONS : 

sigmoid(x) :- The sigmoid function, takes one argument - the val.

d_sigmoid(x) :- Derivative of the sigmoid function, takes one argument - the val.

d_tanh(x) :- Derivative of the tanh function, takes one argument - the val.

g(ai, num_neurons, activation_func) :- The activation function you want to use(ex. sigmoid, tanh),
    Takes 3 arguments - the pre-activation input, length of the layer, and the activation function.

g1(ai, num_neurons, activation_func) :- The derivate of the activation function you want to use(ex. sigmoid, tanh),
    Takes 3 arguments - the pre-activation input, length of the layer, and the activation function.

o(al, num_output) :- The output function. Takes 2 arguments - the output pre-activation and the length of the output layer.

e(i, len) :- Return an array of length len of 0's with 1 at index i. Takes 2 arguments - the index, and the length.

init(W, B, num_input, num_hlayers, num_neurons, num_output) :- The initialize function, which will initialize all the
    weights randomly according to a distribution. All biases are initialized as 0 in our code.
    Takes 6 arguments, all of them as their name suggests.

forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func) :-
    Runs the forward propogation for the given index ii of the training data x_train. 
    Takes 11 arguments, all of them as their name suggests. Returns the final probability distribution for each class label.

back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func) :-
    Runs the back propogation algorithm for the given iith index of training data. y_train is the actual output 
    of the training data. fin_ans is the final probability distribution we get from forward propogation. Returns 
    the derivate of the loss function w.r.t each of the weights and biases. Takes 15 arguments, all of them as their name suggests.

accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func) :- 
    Calculate the accuracy for the current weights and biases. x_test and y_test are the testing data.
    Takes 10 arguments, all of them as their name suggests. Prints the accuracy.

train(x_train, y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, learning_rate, num_epochs, batch_size, opt, train_set, activation_func, loss_func) :-
    Final function which trains the given data according to all the given parameters.
    Takes 17 arguments, all of them as their name suggests.


To run the code, simply set the values to each of the variables inside sweep_config and below that. Then run python feedforward_neuralnetwork.py to execute the code.
An example configuration is given below : -

sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'random', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'num_hlayers': {
            'values': [3]
        },
        'num_epochs': {
            'values': [10]
        },
        'num_neurons': {
            'values': [32]
        },
        'learning_rate': {
            'values': [1e-2]
        },
        'batch_size': {
            'values': [32]
        },
        'opt': {
            'values': ["sgd"]
        },
        'activation_func': {
            'values': ["tanh"]
        }
    }
}
num_output = 10
num_input = 784
train_set = 54000
loss_func = "cross_entropy"

The above sweep_config uses 3 hidden layers, each having 32 neurons, with input layer having 784 neurons and output layer having 10 neurons, running the code
for 10 epochs, with a learning rate of 0.01, mini batch size as 32 and using sgd optimizer, tanh activation function, and cross_entropy loss.

Goal is to maximize the val_acc, so set that in sweep_config. To run for multiple values of any of the hyper parameters in the sweep_config, simply
append those values to the values in each of the items in parameters inside sweep_config.
