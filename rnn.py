
# Feedforward for a sequence of length 1 

import numpy as np

np.random.seed(1)
X1 = np.array([1,0,0,0]).reshape(4,1)
y = np.array([0,1,0,0]).reshape(4,1)
# X1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# X2 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
# y = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])

n_x = X1.shape[0]
n_a = 6 # no of hidden units
n_y = y.shape[0]

Wax = np.random.randn(n_a,n_x)
Waa = np.random.randn(n_a,n_a)
Wya = np.random.randn(n_y,n_a)
a_prev = np.zeros((n_a,1))
ba = np.zeros((n_a,1))
by = np.zeros((n_y,1))

a_curr = np.tanh(np.dot(Waa,a_prev) + np.dot(Wax,X1) + ba)
y_hat = softmax(np.dot(Wya,a_curr) + by)


