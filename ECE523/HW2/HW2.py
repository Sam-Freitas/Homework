# HW2
# Samuel Freitas
# 2/10/21

import numpy
from numpy.random import multivariate_normal as N
import numpy as np
import matplotlib.pyplot as plt



# 2 

# Find the result of minimize the loss
# of sum of the squared errors; however, add in a penalty for an L2 
# penalty on the weights

# argmin{  sum (wTxi - yi)^2 + λ||w||^2.2}

# How does this change the solution to the original linear regression solution? 
# What is the impact of adding in this penalty?
# Write your own implementation of logistic regression and 
# implement your model on either real-world 
# (see Github data sets:https://github.com/gditzler/UA-ECE-523-Sp2018/tree/master/data), 
# or synthetic data. If you simply use Scikit-learn’s implementation of the logistic 
# regression classifier, then you’ll receive zero points. 
# A full 10/10 will be awarded to those that implement logistic 
# regression using the optimization of cross-entropy using stochastic gradient descent

# creating my own synthetic data 
u1 = np.array([0, 2])
u2 = np.array([2, 0])
E = np.array([[1, 0], [0, 1]])
num_samples = 100

data1 = N(u1,E,size= num_samples)
data2 = N(u2,E,size= num_samples)

data_full = np.append(data1,data2,axis=0)
labels_full = np.append(np.zeros(len(data1)),np.ones(len(data2)),axis=0)


def _sigmoid(x):
    return 1/(1 + np.exp(-x))

epochs = 500
lr = 0.1
X = np.asarray(data_full).T
y = np.asarray(labels_full).T
X = np.concatenate([X,np.ones([1,X.shape[-1]])],axis=0)
dims, n_data_points = X.shape
    
W = np.random.randn(1,dims)
loss = []

# train
for i in range(epochs):
    X_hat = np.matmul(W,X)
    y_hat = _sigmoid(X_hat)

    
    
    cost = -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    loss.append(cost)
    
    dc_dw = -np.sum((y-y_hat)*X,axis=-1)[np.newaxis,:]
    W = W - dc_dw * lr - cost
    
def plot_loss(loss):
    plt.scatter(list(range(len(loss))),loss)
    
# predict

# for count, value in enumerate(data_full):
Z = np.asarray(data_full).T
X = np.concatenate([Z,np.ones([1,Z.shape[-1]])],axis=0)
X_hat = np.matmul(W,X)
y_hat = _sigmoid(X_hat)
    

plt.ion()
# creates a figure
f1 = plt.figure(1)
# plots the data 
p1 = plt.plot(data1[:,0], data1[:,1], 'o', c='r')
p2 = plt.plot(data2[:,0], data2[:,1], 'o', c='g')
# plotting options
plt.axis('equal')
plt.title('Training data w/ labels')
plt.legend(['d1','d2'])
f1.show()


plt.ion()
f2 = plt.figure(2)
# plots the data 
for i in range(len(y_hat[0])):
    if np.round(y_hat[0][i]) > 0:
        plt.plot(data_full[i,0], data_full[i,1], 'o', c='r')
    else:
        plt.plot(data_full[i,0], data_full[i,1], 'o', c='g')
plt.axis('equal')
plt.title('tested w/ labels')
plt.legend(['d1','d2'])
f2.show()

plt.ioff()

plt.show()

# 3

# The ECE523 Lecture notes has a function for generating a checkerboard data set. 
# Generate checker-board data from two classes and use any density estimate 
# technique we discussed to classify newdata using
# pY|X(y|x) =̂ pX|Y(x|y)̂pY(y)̂/pX(x)
# where pY|X(y|x) is your estimate of the posterior given you estimates of
# pX|Y(x|y) using a density estimator and̂ pY(y) using a maximum likelihood estimator. 
# You should plot̂ pX|Y(x|y) using apseudo color plot (seehttps://goo.gl/2SDJPL). 
# Note that you must model̂ pX(x),̂pY(y), and̂ pX|Y(x|y). 
# Note that̂ pX(x)can be calculated using the Law of Total Probability.arizona.edu4February 17, 2021
