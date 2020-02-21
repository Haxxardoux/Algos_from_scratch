# For GPU use 
from numba import autojit, jit, cuda 
from timeit import default_timer as timer   

# The rest
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from scipy.io import loadmat
import random 
import math

# Neural net from scratch

# Load input array 
#weights = loadmat('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/ex3weights.mat')
test = loadmat('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/ex3data1.mat')
input_array = np.array(test['X'])
output_array = test['y']

# Temp
transformed_y = []
for i in [i[0] for i in test['y']]:
    foo = [0,0,0,0,0,0,0,0,0,0].copy()
    foo[i-1]=True
    transformed_y.append(foo)
output_array = np.array(transformed_y)

# shuffle the input and output arrays in the same order (seed 4)
random.Random(4).shuffle(input_array)
random.Random(4).shuffle(output_array)

@cuda.jit('void(float64[:])', device=True)
def sigmoid(x):
    """
    Input- one dimensional numpy array
    Output- array of the same shape
    """

    return 1/(1+np.exp(-x))

@cuda.jit('void(int64[:])', device=True)
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

    return weights_matrix

@cuda.jit('void(float64[:,:], int64[:,:], float64[:,:], int64[:])', device=True)
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
    grad_adjustmet_vector = [np.matmul(output_error.T, node_array[-2])]
    for i in np.arange(len(layer_n_list)-1):
        output_error = np.matmul(output_error, weights_matrix[-(i+1)][:,1:]) * (node_array[-(i+2)][:,1:]*(1-node_array[-(i+2)][:,1:]))
        grad_adjustment = np.matmul(output_error.T, node_array[-(i+3)])
        d_vector.append(output_error)
        grad_adjustmet_vector.append(grad_adjustment)
    
    grad_adjustmet_vector.reverse()

    return grad_adjustmet_vector, d_vector[0]

@cuda.jit('void(int64[:,:], float64[:,:], int64[:])', device=True)
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
    for i in np.arange(len(layer_n_list)-1):
        foo = np.matmul(node_array[i], weights_matrix[i].T)
        foo = sigmoid(foo)
        foo = np.concatenate((np.array([[1]]*foo.shape[0]), foo), axis=1)
        node_array.append(foo)

    foo = np.matmul(node_array[-1], weights_matrix[-1].T)
    foo = sigmoid(foo)
    node_array.append(foo)

    return node_array


@cuda.jit('float64(int32[:,:], int32[:,:], int64[:], int32, float64, int32)')
def neural_net(input_array, output_array, n_iterations, layer_n_list, gamma=0, visualize_error=0):
    '''
    Inputs - \n
    input_array - two dimensional numpy array corresponding to x data \n 
    Output_aray - a numpy array corresponding to the y values of each observation \n
    n_iterations - iterations in gradient descent algorithm \n
    layer_n_list - the vector corresponding to the neural network structure.
    Example- [10,5,3] corresponds to some unknown amount of inputs, 10 nodes in layer 1, 5 in layer 2, and 3 output nodes. \n
    gamma - ignore for now \n
    visualize_error - ignore for now  \n \n

    Outputs - training error
    '''
    layer_N_list = [input_array.shape[1]] + layer_n_list
    n_iterations = 10
    # Procedurally initiate matrix of weights
    weights_matrix = initialize_random_weights(layer_N_list)

    # Gradient descent iteration
    error = []
    for i in np.arange(n_iterations):

        # Forward propagation
        node_array = propagate(input_array,weights_matrix,layer_n_list)

        # Backpropagation
        grad_adjustment_vector, iter_error = backpropagate(node_array, output_array, weights_matrix, layer_n_list)

        for i in np.arange(len(layer_n_list)):
            weights_matrix[i] -= (grad_adjustment_vector[i] + gamma*weights_matrix[i]) / 5000
        error.append(iter_error.sum())

    final_error = (output_array - node_array[-1].round()).mean()
    
    if visualize_error == True:
        plt.plot(error)
        plt.show()

    return final_error#, weights_matrix

start = timer()
griddim = (10,10)
blockdim = (10,10)
#neural_net[griddim, blockdim](input_array, output_array, [100, 90, 10], 0, 0)
neural_net(input_array, output_array, [100, 90, 10], 0, 0)
print('time:', timer()-start)
