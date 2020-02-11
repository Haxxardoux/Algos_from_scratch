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

# Weights for first layer
weights_1 = weights['Theta1']

# Weights for second layer
weights_2 = weights['Theta2']

def sigmoid(x, theta_vector):
    """
    Computes estimated y values given theta parameters and x values. 
    Returns: List with same number of elements as x matrix, each element is estimated y value
    """
    foo = np.matmul(x, theta_vector)
    return 1/(1+np.exp(-foo))

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
        weights_matrix.append(weights_array)

    return weights_matrix

# Temp
test_y = [i[0] for i in test['y']==10]
test_x = test['X'][test_y]

def propagate(observation, weights_matrix, n_minus_1_list):
    node_matrix = [observation.tolist()]
    for n in np.arange(len(n_minus_1_list)):
        node_array = []
        for i in np.arange(n_minus_1_list[n]):
            node_array.append(np.dot(weights_matrix[n][i], node_matrix[n]))
        node_matrix.append(node_array)
    return node_matrix

def neural_net(input_array, layer_n_list):
    # Add column of 1s at the beginning
    input_array = np.concatenate((np.array([[1]]*input_array.shape[0]),input_array), axis=1)
    layer_N_list = [input_array.shape[1]] + layer_n_list
    
    weights_matrix = initialize_random_weights(layer_N_list)

    test = input_array[0]

    node_matrix = propagate(test, weights_matrix, layer_n_list)

    print(node_matrix[2][0])

    # propagate(debugging_array[0], weights_array)
    # for i, n in enumerate(layer_n_list[:-1]):
    #     print('array', debugging_array.shape)
    #     print('weights', np.array(weights_matrix[i]).shape)
    #     debugging_array = propagate(debugging_array[0], weights_matrix[i], n)
    return debugging_array

#print(len(initialize_random_weights([401, 25, 10])[0]))
neural_net(input_array, [25,10])

