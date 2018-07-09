
# Building deep neural network with numpy

import numpy as np
import matplotlib.pyplot as plt
import h5py

def weights_initialization(layer_dims):
    np.random.seed(3)
    weights = {}
    L = len(layer_dims)
    
    for l in range(1,L):
        print("sqe:",l)
        print(layer_dims[l])
        print(layer_dims[l-1])
        weights['W'+str(l)] = np.random.randn(layer_dims[1]*layer_dims[l-1])*0.01
        weights['b'+str(l)] = np.zeros((layer_dims[1],1))
    
    return weights
