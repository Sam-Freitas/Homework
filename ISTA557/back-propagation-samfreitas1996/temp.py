import math

import numpy as np
from scipy.special import expit, logit
import pytest

import nn

def s_matrix(x):
    # s_func = lambda z: 1 / (1 + math.exp(-z))
    # sigmoid_matrix = np.vectorize(s_func)
    # # return 1 / (1 + math.exp(-x))
    # return sigmoid_matrix(x)
    y = expit(x)
    return y

input_matrix = np.random.uniform(size=(100, 1))
output_matrix = (input_matrix > 0.5).astype(int)

net = nn.SimpleNetwork.of(1, 5, 5, 1)

test_inputs = np.array([[0.0], [0.1], [0.2], [0.3], [0.4],
                        [0.6], [0.7], [0.8], [0.9], [1.0]])
test_outputs = np.array([[0], [0], [0], [0], [0],
                            [1], [1], [1], [1], [1]])

a = []
h = []
h.append(test_inputs)
i=0
# iterate through the weights to perform feed forward
# where :
# a - activation layer (weighted sum?)
# h - application of non-linearity 
# s_matrix is just a sigmoid funtion that runs on a matrix
for thisWeight in net.weights:
    a.append(h[i].dot(thisWeight))
    h.append(s_matrix(a[i]))
    i +=1

print(h[-1], 'h last')
prediction_matrix_binary = np.where(h[-1] < 0.5, 0,1)


print('end')