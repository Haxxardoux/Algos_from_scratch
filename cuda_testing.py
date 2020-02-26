# For GPU use 
from numba import autojit, jit, cuda, float32, vectorize, guvectorize
from timeit import default_timer as timer

# The rest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
import math

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
    
    weights_matrix = [cp.asarray(i) for i in weights_matrix]

    return weights_matrix

def sigmoid(x):
    return 1/(1+cp.exp(-x))

def propagate(input_array, weights_matrix, layer_n_list):
    node_array = [input_array]
    for i in cp.arange(len(layer_n_list)-1):
        foo = cp.matmul(node_array[int(i)], weights_matrix[int(i)].T)
        foo = sigmoid(foo)
        foo = cp.concatenate((cp.array([[1]]*foo.shape[0]), foo), axis=1)
        print(foo.shape)
        node_array.append(foo)

    foo = cp.matmul(node_array[-1], weights_matrix[-1].T)
    foo = sigmoid(foo)
    node_array.append(foo)

    return node_array

def backpropagate(node_array, output_array, weights_matrix, layer_n_list):
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

layer_n_list = [25,10]
layer_N_list = [400, 25, 10]

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



List = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
List2 = List
List2.reverse()
test_array = np.outer(np.ones(16),List)
test_array2 = np.outer(np.ones(16),List2)
empty_array = np.zeros_like(test_array)

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

#fast_matmul(test_array, test_array2, empty_array)
#print(empty_array)




@vectorize(['float64(float64)'], target='cuda')
def sigmoid(array_element):
    return math.exp(array_element)

@vectorize(['float32(float32, float32)'], target='cuda')
def add(x, y):
    return x + y

@guvectorize(['void(float64[:,:], float64[:,:], float64[:,:])'], '(m,n),(n,p)->(m,p)')
def matmul(A, B, C):
    m, n = A.shape
    n, p = B.shape
    for i in range(m):
        for j in range(p):
            C[i, j] = 0
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

List = np.linspace(200,500,400)
Array = np.outer(np.ones(5000),List)

List2 = np.linspace(100,150,25)
Array2 = np.outer(np.ones(400), List2)

gpu_output = np.zeros(shape=(Array.shape[0],Array2.shape[1]), dtype=np.float32)

normalized_gpu = cuda.device_array(shape=(Array.shape[0],Array2.shape[1]), dtype=np.float32)


# Timing 
# ======================================
start = timer()
for i in range(10):
    matmul(Array,Array2,gpu_output, out=normalized_gpu)
print('gpu time',start-timer())

start = timer()
for i in range(10):
    np.matmul(Array,Array2)
print('cpu time', start-timer())















# test_structure = [400, 100, 50, 30, 10]
# propagate(input_array, initialize_random_weights(test_structure), test_structure[1:])

# n_observations = len(input_array)
# test_structure[1:-1] = np.array(test_structure[1:-1])+1

# array_dimensions = list(zip([len(input_array)]*len(test_structure), test_structure))

# empty = np.array([ [print(shape) for shape in array_dimensions] ])
# print(empty)