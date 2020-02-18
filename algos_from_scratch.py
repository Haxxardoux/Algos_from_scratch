# Logistic regression from scratch

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from scipy.io import loadmat
import random 
import math

x = np.arange(100)
y = np.array([0]*53 + [1]*47)
x = (x-min(x))/(max(x)-min(x))


# Linear regression from scratch
# Not done, obvious errors but too lazy to fix
# Still works though

# x = np.arange(10000)
# y = (np.arange(10000)*2)

# def hypothesis(x, y, beta0, beta1, learning_rate = 0.00000001):
#     param_vector = [beta0, beta1]

#     x_matrix = []
#     for i in x:
#         x_matrix.append([1,i])
#     error_matrix = []
#     for i in np.arange(15):
#         hypothesis_evaluation = np.matmul(x_matrix, [beta0, beta1])
#         error = np.array(hypothesis_evaluation) - np.array(y)
#         error_matrix.append(np.sum(error))
#         beta0 = beta0 - (learning_rate/len(x)) * np.sum(error)
#         beta1 = beta1 - (learning_rate/len(x)) * np.sum(error*np.array(x))
#         # plt.plot(x,y)
#         # plt.plot(x,hypothesis_evaluation.tolist())
#         # plt.show()

#     print('beta 0:',beta0, 'beta 1:',beta1)

# from sklearn.linear_model import LinearRegression
# import time

# t0 = time.time()
# hypothesis(x,y,2,1)
# t1 = time.time()
# print('my time',t0-t1)

# t2 = time.time()
# reg = LinearRegression().fit(x.reshape(1, -1), y.reshape(1, -1))

# t3 = time.time()
# print('their time',t2-t3)


# ==============================================================
# Neural net from scratch
# Not done lol

# Load input array 
weights = loadmat('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/ex3weights.mat')
test = loadmat('C:/Users/turbo/Python projects/Algos from scratch/Algos_from_scratch/ex3data1.mat')
input_array = test['X']
output_array = test['y']
input_array = np.array(input_array)
# Weights for first layer
weights_1 = weights['Theta1']

# Weights for second layer
weights_2 = weights['Theta2']

def sigmoid(x):
    """
    Computes estimated y values given theta parameters and x values. 
    Returns: List with same number of elements as x matrix, each element is estimated y value
    """
    #foo = np.matmul(x, theta_vector)
    return 1/(1+np.exp(-x))

def gradient(x, theta_vector):
    """
    Computes the gradient given sigmoid, specifically for neural net backpropagation
    """
    return sigmoid(x, theta_vector)*(1-sigmoid(x, theta_vector))

def initialize_random_weights(layer_n_list):
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

def backpropagate(transformed_y, nodes_matrix, weights_matrix, n_minus_1_list):
    nodes_matrix = nodes_matrix[::-1]
    weights_matrix = weights_matrix[::-1]
    output_error = np.array(nodes_matrix[0])-np.array(transformed_y)
    error_matrix = [output_error.tolist()]
    delta = output_error
    for i, k in enumerate(n_minus_1_list):
        gz = np.array(nodes_matrix[i+1])*(1-np.array(nodes_matrix[i+1]))
        delta = np.dot(np.array(weights_matrix[i]).T,delta) * gz
        error_matrix.append(delta.tolist())
    #print(np.array(error_matrix[0]))
    #print(np.array(nodes_matrix[1]))
    return error_matrix 

def propagate(input_array, weights_matrix, layer_n_list):
    node_array = [input_array]
    for i in np.arange(len(layer_n_list)-1):
        foo = np.matmul(node_array[0], weights_matrix[0].T)
        foo = sigmoid(foo)
        foo = np.concatenate((np.array([[1]]*foo.shape[0]), foo), axis=1)
        node_array.append(foo)

    foo = np.matmul(node_array[-1], weights_matrix[-1].T)
    foo = sigmoid(foo)
    node_array.append(foo)

    return node_array

def neural_net(input_array, layer_n_list, gamma, n_iterations):
    # Add column of 1s at the beginning
    input_array = np.concatenate((np.array([[1]]*input_array.shape[0]),input_array), axis=1)
    layer_N_list = [input_array.shape[1]] + layer_n_list
    temp = input_array

    # Procedurally initiate matrix of weights
    weights_matrix = initialize_random_weights(layer_N_list)

    # Gradient descent iteration
    error = []
    for i in np.arange(n_iterations):
        input_array = temp
        node_array = propagate(input_array,weights_matrix,layer_n_list)

        # add backpropogate function
        d3 = node_array[2] - output_array
        d2 = np.matmul(d3, weights_matrix[1][:,1:]) * (node_array[1][:,1:]*(1-node_array[1][:,1:]))
        Delta2 = np.matmul(d3.T, node_array[1])
        Delta1 = np.matmul(d2.T, node_array[0])
        weights_matrix[1] -= (Delta2 + gamma*weights_matrix[1]) / (5000)
        weights_matrix[0] -= (Delta1 + gamma*weights_matrix[0]) / (5000)
        error.append(d3.sum())

    print((output_array - node_array[2].round()).mean())
    plt.plot(error)
    plt.show()

    return 


print(neural_net(input_array, [25,10], 0.097,300))


