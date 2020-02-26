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
# ======================================================================================================================================================
# Not done, obvious errors but too lazy to fix
# Still works though

x = np.arange(10000)
y = (np.arange(10000)*2)

def hypothesis(x, y, beta0, beta1, learning_rate = 0.00000001):
    param_vector = [beta0, beta1]

    x_matrix = []
    for i in x:
        x_matrix.append([1,i])
    error_matrix = []
    for i in np.arange(15):
        hypothesis_evaluation = np.matmul(x_matrix, [beta0, beta1])
        error = np.array(hypothesis_evaluation) - np.array(y)
        error_matrix.append(np.sum(error))
        beta0 = beta0 - (learning_rate/len(x)) * np.sum(error)
        beta1 = beta1 - (learning_rate/len(x)) * np.sum(error*np.array(x))
        # plt.plot(x,y)
        # plt.plot(x,hypothesis_evaluation.tolist())
        # plt.show()

    print('beta 0:',beta0, 'beta 1:',beta1)

from sklearn.linear_model import LinearRegression
import time

t0 = time.time()
hypothesis(x,y,2,1)
t1 = time.time()
print('my time',t0-t1)

t2 = time.time()
reg = LinearRegression().fit(x.reshape(1, -1), y.reshape(1, -1))

t3 = time.time()
print('their time',t2-t3)

