
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def Initialize_parameters_deep(layers_dim):
    np.random.seed(3)
    L = len(layers_dim)
    parameters = {}
    for l in range(1,L):
        parameters["W" + str(l)] = np.random.randn(layers_dim[l],layers_dim[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dim[l],1))

        assert(parameters["W" + str(l)].shape == (layers_dim[l],layers_dim[l-1]))
        assert(parameters["b" + str(l)].shape == (layers_dim[l],1))

    return parameters
  
def linear_forward(A,W,b):

    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)

    return Z, cache

def sigmoid(Z):
    a = 1/(1+np.exp(-Z))
    cache = (Z,)
    return a, cache

def relu(Z):
    a = Z * (Z > 0)
    cache = (Z,)
    return a, cache

def linear_activation_forward(A_prev,W,b,activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)

    return A, cache

def L_model_forward(X,parameters):

    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1,L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]

        A, cache = linear_activation_forward(A_prev,W,b,activation="relu")
        caches.append(cache)

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]

    AL, cache = linear_activation_forward(A,W,b,activation="sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL,caches

def compute_cost(AL,y):
    m = Y.shape[1]

    cost = -(1/m)*np.sum(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T))
    cost = np.sqeeze(cost)
    assert(cost.shape == ())
    return cost

  
