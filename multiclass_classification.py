from scipy.io import loadmat
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

# OneVsMany classification from scratch - classifying handwritten numbers between 0 and 9 (labels 1-10)
# Also logistic regression
# Done!

# Load data
test = loadmat('ex3data1.mat')
input_array = test['X']

# 5000 examples of 20x20 numbers, 5000x400 matrix, add column of 1s at beginning. each pixel will be a feature, and we need a column of ones to act as intercept parameter
input_array_one = np.concatenate((np.array([[1]]*input_array.shape[0]),input_array), axis=1)

# using gradient descent, so important to normalize 
def normalize(array):
    """
    Return array with sum 0 
    """
    return array/np.sum(array)

# Sigmoid is general term for function, computes the estimated y values given the parameters of logistic function and inputs. 
def sigmoid(x, theta_vector):
    """
    Computes estimated y values given theta parameters and x values. 
    Returns: List with same number of elements as x matrix, each element is estimated y value
    """
    foo = np.matmul(x, theta_vector)

    return 1/(1+np.exp(-foo))

def gradient(predicted_points, y, x):
    """
    Computes the "gradient" part of gradient descent. Returns a list with length equal to number of parameters 
    """
    error = predicted_points - y
    error = np.matmul(error, x)

    return error/401 

def logistic_regression(x, y, learning_rate = 0.5, debugging_mode = False):
    """
    Standard logistic regression function, but parameters minimized with gradient descent. Learning rate will almost certainly require optimization. To do so, you can set debugging_mode to true to get graphs of the error
    Returns: guesses for y, and vector of parameters. 
    """
    x = np.concatenate((np.array([[1]]*x.shape[0]),x), axis=1)
    error_matrix = []
    param_vector = np.array([1]*x.shape[1], dtype=np.float64)
    param_vector = normalize(param_vector)

    for i in np.arange(200):
        hypothesis_evaluation = sigmoid(x, param_vector)        
        gradients = gradient(hypothesis_evaluation,y,x)
        error_matrix.append(np.sum(gradients)*x.shape[1])
        adjustment = learning_rate*gradients
        param_vector -= learning_rate*adjustment
    training = sigmoid(x,param_vector)>.5
    guesses = [i[0] for i in test['y'][training]]

    if debugging_mode == True:
        plt.plot(error_matrix)
        plt.ylim(-100,100)
        plt.show()

    return guesses, param_vector

# Actual multiclass classification, train classifier for each class. The error metric used only accounts for instances where the classifier misclassified the target class, does not include instances when the classifier neglects to classify the target class
for i in np.arange(1,11):
    print(i)
    bools = test['y']==i
    bools = [j[0] for j in bools.tolist()]
    guesses, param_vector = logistic_regression(input_array, bools)
    print(len(guesses))
    print(1-np.mean(np.array(guesses) != i))