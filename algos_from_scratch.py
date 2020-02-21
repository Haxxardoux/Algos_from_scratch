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

@cuda.autojit
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

@cuda.autojit
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

@cuda.jit('void(float64[:,:], int64[:,:], float64[:,:], int64[:])', device=True)
def backpropagate(node_array, output_array, weights_matrix, layer_n_list):
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
def neural_net(input_array, output_array, layer_n_list, gamma=0, visualize_error=0):
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
neural_net[griddim, blockdim](input_array, output_array, [100, 90, 10], 0, 0)
print('time:', timer()-start)

# Diagnosing bias vs variance
# ====================================================================================================

def train_cv_split(data, data2=None, cv_pct=0.2):
    '''
    Inputs - Dataset in the form of np array, percentage of dataset you would like to select as cross validation set \n
    There are two "dataset" arguments, data and data2. This is in case you have separate arrays for x and y. They will be sampled with the same indices.
    Outputs - Training set, cross validation set, output format is np array
    '''
    idx = np.random.randint(data.shape[0], size=int(round(data.shape[0]*cv_pct)))
    cv = data[idx, :]
    training = data[~idx, :]
    if data2 is not None:
        cv_2 = data2[idx, :]
        training_2 = data[~idx, :]
        return training, cv, training_2, cv_2
    
    return training, cv

#train, test = train_cv_split(input_array)

train_error_vector = []
test_error_vector = []


# ======================================================================================================
### COMBINING INTO CLASS OBJECT ###
# ======================================================================================================


class neural_network():
    def load_data(self, x_array, y_array):
        x_array = np.concatenate((np.array([[1]]*x_array.shape[0]), x_array), axis=1)
        random.Random(4).shuffle(x_array)
        random.Random(4).shuffle(y_array)
        self.x_array = x_array
        self.y_array = y_array

    def train_cv_split(self, cv_pct=0.2):
        '''
        Inputs - Dataset in the form of np array, percentage of dataset you would like to select as cross validation set \n
        There are two "dataset" arguments, data and data2. This is in case you have separate arrays for x and y. They will be sampled with the same indices.
        Outputs - Training set, cross validation set, output format is np array
        '''
        
        idx = np.random.randint(self.x_array.shape[0], size=int(round(self.x_array.shape[0]*cv_pct)))
        
        self.x_array_test = self.x_array[idx, :]
        self.x_array_train = self.x_array[~idx, :]
        self.y_array_test = self.y_array[idx, :]
        self.y_array_train = self.y_array[~idx, :]

    def train_net(self, layer_n_list, n_iterations, gamma=0):
        self.n_iterations = n_iterations
        self.layer_n_list = layer_n_list

        try:
            self.x_array_train
        except AttributeError:
            self.train_cv_split()

        layer_N_list = [self.x_array_train.shape[1]] + self.layer_n_list

        # Procedurally initiate matrix of weights
        weights_matrix = initialize_random_weights(layer_N_list)

        # Gradient descent iteration
        error = []
        for i in np.arange(self.n_iterations):

            # Forward propagation
            node_array = propagate(self.x_array_train, weights_matrix, self.layer_n_list)

            # Backpropagation
            grad_adjustment_vector, iter_error = backpropagate(node_array, self.y_array_train, weights_matrix, self.layer_n_list)

            for i in np.arange(len(self.layer_n_list)):
                weights_matrix[i] -= (grad_adjustment_vector[i] + gamma*weights_matrix[i]) / 5000
            error.append(iter_error.sum())

        self.node_array = node_array
        self.weights_matrix = weights_matrix
        self.train_error = abs((self.y_array_train - self.node_array[-1].round())).mean()

    def test_net(self):
        node_array = propagate(self.x_array_test, self.weights_matrix, self.layer_n_list)
        self.test_error = abs((self.y_array_test - node_array[-1].round())).mean()

# net = neural_network()

# net.load_data(input_array, output_array)

# start = timer()
# x_list = np.linspace(0.1,0.5,5)
# test = []
# train = [] 
# for i in x_list:
#     test_error_vector = []
#     train_error_vector = []
#     for j in np.arange(10):
#         net.train_cv_split(cv_pct = i)
#         net.train_net([100,10], 100)
#         net.test_net()
#         test_error_vector.append(net.test_error)
#         train_error_vector.append(net.train_error)
#     test.append(np.mean(test_error_vector))
#     train.append(np.mean(train_error_vector))
# print('without gpu:', timer()-start)


#plt.plot(x_list, np.array(test))
#plt.plot(x_list, np.array(train))
#plt.show()