
# Building deep neural network with numpy

import numpy as np
import matplotlib.pyplot as plt
import h5py

def weights_initialization(layer_dims):
    
    weights = {}
    L = len(layer_dims)
    
    for l in range(1,L):

        weights['W'+str(l)] = np.random.randn(layer_dims[1]*layer_dims[l-1])*0.01
        weights['b'+str(l)] = np.zeros((layer_dims[1],1))
    
    return weights
