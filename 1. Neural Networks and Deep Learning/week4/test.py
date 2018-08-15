# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 19:43:10 2018

@author: zhangyw49
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def linear_forward(A_pre, W, b):
    Z = W.dot(A_pre) + b
    cache = A_pre, W, b
    return Z, cache

def sigmoid_forward(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache

def relu_forward(Z):
    A = np.maximum(0, Z)
    cache =Z
    return A, cache

def linear_backward(dZ, cache):
    A_pre, W, b = cache
    m = W.shape[1]
    dW = dZ.dot(A_pre.T) / m
    db = dZ.sum(axis=1, keepdims=True) / m
    dA_pre = W.T.dot(dZ)
    return dA_pre, dW, db

def sigmoid_backward(dA, cache):
    Z = cache
    A = 1 / (1 + np.exp(-Z))
    dZ = dA * A * (1-A)
    return dZ

def relu_backward(dA, cache):
    Z = cache
    dZ = dA.copy()
    dZ[Z<=0] = 0
    return dZ


###############################################
def linear_active_forward(A_pre, W, b, active):
    Z, linear_cache = linear_forward(A_pre, W, b)
    if active == "sigmoid":
        A, active_cache = sigmoid_forward(Z)
    elif active == "relu":
        A, active_cache = relu_forward(Z)
    cache = (linear_cache, active_cache)
    return A, cache

def linear_active_backward(dA, cache, active):
    linear_cache, active_cache = cache
    if active == "sigmoid":
        dZ = sigmoid_backward(dA, active_cache)
    elif active == "relu":
        dZ = relu_backward(dA, active_cache)
    dA_pre, dW, db = linear_backward(dZ, linear_cache)
    return dA_pre, dW, db
################################################

def dnn_forward(X, parameters):
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):
        A, cache = linear_active_forward(A, parameters["W"+str(l)], parameters["b"+str(l)], active="relu")
        caches.append(cache)
    AL, cache = linear_active_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], active="sigmoid")
    caches.append(cache)
    return AL, caches

def dnn_backward(Y, AL, caches, parameters):
    L = len(parameters) // 2
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    dA, dW, db = linear_active_backward(dAL, caches[-1], active="sigmoid")
    grads = {"dW"+str(L): dW, "db"+str(L): db}
    
    for l in reversed(range(1, L)):
        dA , dW, db = linear_active_backward(dA, caches[l-1], active="relu")
        grads["dW"+str(l)], grads["db"+str(l)] = dW, db
    return grads
####################################################

def parameters_init(layer_dims, random_state=None):
    np.random.seed = random_state
    L = len(layer_dims)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(Y.dot(np.log(AL).T) + (1-Y).dot(np.log(1-AL.T))) / m
    return np.squeeze(cost)

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L):
        parameters["W"+str(l)] -= grads["dW"+str(l)] * learning_rate
        parameters["b"+str(l)] -= grads["db"+str(l)] * learning_rate
    return parameters
########################################################
def DNN_model(X, Y, layer_dims, learning_rate=0.075, num_iter=2500, random_state=0, print_cost=True):
    parameters = parameters_init(layer_dims, random_state)
    costs = []
    for i in range(num_iter):
        AL, caches = dnn_forward(X, parameters)
        grads = dnn_backward(Y, AL, caches, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            cost = compute_cost(AL, Y)
            costs.append(cost)
            print ("Cost after iteration %i: %f" %(i, float(cost)))
            # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters

def DNN_predict(X, parameters):
    AL, _ = dnn_forward(X, parameters)
    Yhat = np.round(AL)
    return Yhat
#########################################################
layer_dims = [12288,8, 2, 1]
parameters  = DNN_model(train_set_x, train_set_y, layer_dims)



## START CODE HERE ##
my_image = "cat.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)
## END CODE HERE ##

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")



##############################################################
