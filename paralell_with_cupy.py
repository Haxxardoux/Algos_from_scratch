# For GPU use 
from numba import autojit, jit, cuda, float32
from timeit import default_timer as timer
import cupy as cp

# The rest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
import math

# Importing data
test = loadmat('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/ex3data1.mat')
input_array = cp.array(test['X'])
output_array = test['y']

# Cleaning output array
transformed_y = []
for i in [i[0] for i in test['y']]:
    foo = [0,0,0,0,0,0,0,0,0,0].copy()
    foo[i-1]=True
    transformed_y.append(foo)
output_array = cp.array(transformed_y)


def neural_net(input_array, output_array, n_iterations, layer_n_list, gamma=0, visualize_error=0):
    layer_N_list = [input_array.shape[1]] + layer_n_list

    # Procedurally initiate matrix of weights
    weights_matrix = initialize_random_weights(layer_N_list)

    # Gradient descent iteration
    error = []
    for i in np.arange(n_iterations):

        # Forward propagation
        node_array = propagate(input_array,weights_matrix,layer_n_list)

        # Backpropagation
        grad_adjustment_vector, iter_error = backpropagate(node_array, output_array, weights_matrix, layer_n_list)

        for i in cp.arange(len(layer_n_list)):
            weights_matrix[int(i)] -= (grad_adjustment_vector[int(i)] + gamma*weights_matrix[int(i)]) / 5000
        error.append(iter_error.sum())

    final_error = (output_array - node_array[-1].round()).mean()
    
    if visualize_error == True:
        plt.plot(error)
        plt.show()

    return final_error

# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ 
## EXPERIMENTATION SECTION ##
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ 

# layer_n_list = [25,10]
# layer_N_list = [400, 25, 10]

# list_of_structures = [[100, 50, 30, 10], [25, 10],[50, 25, 10],[50, 40, 30, 10],[800, 100, 10],[100, 50, 30, 10],[25, 10],[50, 25, 10],[50, 40, 30, 10],[800, 100, 10],[100, 50, 30, 10],[25, 10],[50, 25, 10],[50, 40, 30, 10],[800, 100, 10]]

# # Start timer
# #start = timer()

# # Processes
# List_of_structures = []
# for i in list_of_structures:
#     item = [input_array.shape[1]] + i
#     List_of_structures.append(item)

# weights_tensor = []
# for i in List_of_structures:
#     weights_tensor.append(initialize_random_weights(i))

# We find that paralell iteration of gradient descent algorithm is not faster than successive. :/. this is probably not the case when implemented correct.
# 3x faster than non-gpu

run_training_code = False

if run_training_code == True:
    for j in cp.arange(50):
        node_tensor = []
        for i, structure in enumerate(List_of_structures):
            # Forward propagation
            node_tensor.append(propagate(input_array, weights_tensor[i], structure[1:]))

            # Backpropagation
            grad_adjustment_vector, iter_error = backpropagate(node_tensor[i], output_array, weights_tensor[i], structure[1:])

            for k in np.arange(len(structure[1:])):
                weights_tensor[i][k] -= (grad_adjustment_vector[k] / 5000) #+ gamma*weights_matrix[k]) / 5000

    for i, structure in enumerate(List_of_structures):
        final_error = (output_array - node_tensor[i][-1].round()).mean()


    # Stop timer
    time = timer()-start
    print('total training time', time)

# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
## END EXPERIMENTATION SECTION
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 




# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ 
### COMBINING INTO CLASS OBJECT ###
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ 

# Helper functions
# These are the functions required for the class to operate correctly, despite not being class functions
def train_cv_split(x_array, y_array, cv_pct):
    '''
    Inputs - Dataset in the form of np array, percentage of dataset you would like to select as cross validation set \n
    There are two "dataset" arguments, data and data2. This is in case you have separate arrays for x and y. They will be sampled with the same indices.
    Outputs - Training set, cross validation set, output format is np array
    '''
    
    idx = cp.random.randint(x_array.shape[0], size= int(cp.around(x_array.shape[0]*cv_pct)) )
    
    mask = cp.ones(x_array.shape[0],dtype=bool) 
    mask[idx] = 0

    x_array_test = x_array[idx, :]
    x_array_train = x_array[mask, :]
    y_array_test = y_array[idx, :]
    y_array_train = y_array[mask, :]

    return x_array_train, x_array_test, y_array_train, y_array_test

def backpropagate(node_array, output_array, weights_matrix, layer_n_list):
    ''' 
    My implementation of a backpropagation algorithm. 
    Inputs - \n
    node_array - list of np arrays, each array (each list element) corresponds to the nodes in each layer, for each training observation. The first element is the input layer, last element is output layer. \n
    Output_aray - a numpy array corresponding to the y values of each observation \n
    weights_matrix - A list where the elements are two dimensional numpy arrays of float64s. \n
    layer_n_list - the vector corresponding to the neural network structure.
    Example- [10,5,3] corresponds to some unknown amount of inputs, 10 nodes in layer 1, 5 in layer 2, and 3 output nodes. \n \n

    Outputs - \n
    grad_adjustment_vector - list of numpy arrays containing the amounts that the weights should be adjusted to reduce error in a given iteration of the backprop algorithm \n
    Output err-r also used in adjusting weights in backprop algorithm
    '''
 
    output_error = node_array[-1] - output_array
    d_vector = [output_error]
    grad_adjustmet_vector = [cp.matmul(output_error.T, node_array[-2])]
    for i in cp.arange(len(layer_n_list)-1):
        output_error = cp.matmul(output_error, weights_matrix[-int(i+1)][:,1:]) * (node_array[-int(i+2)][:,1:]*(1-node_array[-int(i+2)][:,1:]))
        grad_adjustment = cp.matmul(output_error.T, node_array[-int(i+3)])
        d_vector.append(output_error)
        grad_adjustmet_vector.append(grad_adjustment)
    
    grad_adjustmet_vector.reverse()

    return grad_adjustmet_vector, d_vector[0]

def initialize_random_weights(layer_n_list):
    '''
    Input- the vector corresponding to the neural network structure.
    Example- [10,5,3] corresponds to some unknown amount of inputs, 10 nodes in layer 1, 5 in layer 2, and 3 output nodes. \n
    The last element is always the number of output nodes. \n \n
    Output- A list where the elements are two dimensional numpy arrays of float64s. 
    '''
    random_choice_params = []
    for i in np.arange(len(layer_n_list)-1):
        adjacent_lengths = layer_n_list[i] + layer_n_list[i+1]
        random_choice_params.append(math.sqrt(6)/(math.sqrt(adjacent_lengths)))
    
    weights_matrix = []
    for n, k in enumerate(random_choice_params):
        weights_array = []
        for i in np.arange(layer_n_list[n+1]):
            x_feature_list = []
            for j in np.arange(layer_n_list[n]):
                x_feature_list.append(random.uniform(-k, k))
            weights_array.append(x_feature_list)
        weights_matrix.append(np.array(weights_array))

    for i in np.arange(len(layer_n_list)-2):
        weights_matrix[i+1] = np.concatenate((np.array([[1]]*weights_matrix[i+1].shape[0]), weights_matrix[i+1]), axis=1)
    
    weights_matrix = [cp.asarray(i) for i in weights_matrix]

    return weights_matrix

def sigmoid(x):
    """
    Input- one dimensional numpy array
    Output- array of the same shape
    """
    return 1/(1+cp.exp(-x))

def propagate(input_array, weights_matrix, layer_n_list):
    '''
    forward propagation algorithm. \n
    Inputs - \n
    input_array - two dimensional numpy array corresponding to x data \n 
    weights_matrix - A list where the elements are two dimensional numpy arrays of float64s. \n
    layer_n_list- the vector corresponding to the neural network structure.
    Example- [10,5,3] corresponds to some unknown amount of inputs, 10 nodes in layer 1, 5 in layer 2, and 3 output nodes. \n
    Output - \n
    node_array - list of np arrays, each array (each list element) corresponds to the nodes in each layer, for each training observation. The first element is the input layer, last element is output layer. \n
    '''
    node_array = [input_array]
    for i in cp.arange(len(layer_n_list)-1):
        foo = cp.matmul(node_array[int(i)], weights_matrix[int(i)].T)
        foo = sigmoid(foo)

        foo = cp.concatenate((cp.array([[1]]*foo.shape[0]), foo), axis=1)

        node_array.append(foo)

    foo = cp.matmul(node_array[-1], weights_matrix[-1].T)
    foo = sigmoid(foo)
    node_array.append(foo)

    return node_array


class neural_network():
    def load_data(self, x_array, y_array):
        x_array = cp.concatenate((cp.array([[1]]*x_array.shape[0]), x_array), axis=1)
        random.Random(4).shuffle(x_array)
        random.Random(4).shuffle(y_array)
        self.n_observations = x_array.shape[1]
        self.x_array = x_array
        self.y_array = y_array

    def train_cv_split(self, cv_pct=0.2):
        '''
        Inputs - Dataset in the form of np array, percentage of dataset you would like to select as cross validation set \n
        There are two "dataset" arguments, data and data2. This is in case you have separate arrays for x and y. They will be sampled with the same indices.
        Outputs - Training set, cross validation set, output format is np array
        '''
        
        # Create indices masks for test and training set
        idx = cp.random.randint(self.x_array.shape[0], size=int(round(self.x_array.shape[0]*cv_pct)))
        mask = cp.ones(self.x_array.shape[0], dtype=bool) 
        mask[idx] = False

        # Split train and test set into folds, since backpropagation efficiency goes as n^5. Optimal size of folds depends on ability to compute in paralell.
        self.train_folds = math.floor(sum(mask)/2500)
        self.train_cutoffs = cp.linspace(0, sum(mask), self.train_folds+1)
        self.test_folds = math.floor(len(idx)/2500)
        self.test_cutoffs = cp.linspace(0, len(idx), self.test_folds+1)

        x_train_list = []
        y_train_list = []
        for i in cp.arange(self.train_folds):
            x_train_list.append(self.x_array[mask, :][int(self.train_cutoffs[i]):int(self.train_cutoffs[i+1]), :])
            y_train_list.append(self.y_array[mask, :][int(self.train_cutoffs[i]):int(self.train_cutoffs[i+1]), :])

        y_test_list = []
        x_test_list = []
        for i in cp.arange(self.test_folds):
            x_test_list.append(self.y_array[idx, :][self.test_cutoffs[i]:self.test_cutoffs[i+1], :])
            y_test_list.append(self.y_array[idx, :][self.test_cutoffs[i]:self.test_cutoffs[i+1], :])

        self.x_array_test = x_test_list
        self.x_array_train = x_train_list
        self.y_array_test = y_test_list
        self.y_array_train = y_train_list


    def train_net(self, layer_n_list, n_iterations, gamma=0):
        self.n_iterations = n_iterations
        self.layer_n_list = layer_n_list
        start = timer()
        try:
            self.x_array_train
        except AttributeError:
            self.train_cv_split()

        layer_N_list = [self.n_observations] + self.layer_n_list

        # Procedurally initiate matrix of weights
        weights_matrix = initialize_random_weights(layer_N_list)

        # Gradient descent iteration
        for n in cp.arange(self.n_iterations):   
            error = []
            for j in cp.arange(self.train_folds):
                # Forward propagation
                node_array = propagate(self.x_array_train[int(j)], weights_matrix, self.layer_n_list)

                # Backpropagation
                grad_adjustment_vector, iter_error = backpropagate(node_array, self.y_array_train[int(j)], weights_matrix, self.layer_n_list)

                for i in cp.arange(len(self.layer_n_list)):
                    weights_matrix[int(i)] -= (grad_adjustment_vector[int(i)] + gamma*weights_matrix[int(i)]) / 5000
                error.append(iter_error.sum())

            self.node_array = node_array
            self.weights_matrix = weights_matrix
            self.train_error = abs((self.y_array_train[int(j)] - self.node_array[-1].round()))

        print(self.train_error.mean())
        print('time',timer()-start)
    
    def test_net(self):
        x_array_test =  [i[0] for i in self.x_array_test]
        y_array_test =  [i[0] for i in self.y_array_test]

        node_array = propagate(x_array_test, self.weights_matrix, self.layer_n_list)
        self.test_error = abs((y_array_test - node_array[-1].round())).mean()


    def graph_learning_curve(self, layer_n_list, n_samples = 5, n_iterations = 50, cv_size_min = 0.1, cv_size_max = 0.5, timeit = False):
        '''
        Does not require any prior neural net training or validation, or train/test split
        '''
        if timeit == True:
            start = timer()

        cv_sizes = cp.linspace(cv_size_min, cv_size_max, int((cv_size_max - cv_size_min)*20))
        train_error_vector = []
        test_error_vector = []
        for cv_size in cv_sizes:
            train_error = []
            test_error = []
            for i in cp.arange(n_samples):
                self.x_array_train, self.x_array_test, self.y_array_train, self.y_array_test = train_cv_split(self.x_array, self.y_array, cv_pct = cv_size)
                self.train_net(layer_n_list, n_iterations)
                train_error.append(self.train_error)
                self.test_net()
                test_error.append(self.test_error)
            train_error_vector.append(np.mean(train_error))
            test_error_vector.append(np.mean(test_error))

        train_error_vector = np.array(train_error_vector)
        test_error_vector = np.array(test_error_vector)
        cv_sizes = np.linspace( cv_size_min, cv_size_max, int((cv_size_max - cv_size_min)*20) )

        if timeit == True:
            print('Time to do {} iterations:'.format(int((cv_size_max - cv_size_min)*20)*n_samples), round(timer()-start, 3))

        # eventually replace with legend because i am not a fucking moron
        print('The blue line is training error, orange line is test error')

        plt.plot(cv_sizes, train_error_vector)
        plt.plot(cv_sizes, test_error_vector)
        plt.show()            

net = neural_network()
net.load_data(input_array, output_array)
net.train_net([25,10],50)

#net.graph_learning_curve([25, 10], timeit=True)
        



