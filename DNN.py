
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
  
  
