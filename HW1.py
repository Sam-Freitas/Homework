# HW1
# Samuel Freitas
# 1/24/2021



### 2 

import numpy
from numpy.random import multivariate_normal as N
from numpy.linalg import inv,det
import numpy as np

import sklearn
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


## a 
# write a general function to generate random samples from N(u,E) in d-dimensions 

# in this homework u is the mean of a nd dataset and E is the cov matrix of 1 a nd dataset 

# dimetionaity is determined by the length of the mean and cov (u,E) vectors 
u = np.array([0, 0])
E = np.array([[1, 0], [0, 1]])
num_samples = 100

data = N(u,E,size= num_samples)


## b
# write a procedure of the discriminant:

def g(x,u,E):

    d = len(x)

    a = (-1/2)*np.matmul(np.matmul((x-u).T,inv(E)),(x-u))
    b = (-1*d/2)*np.log(2*np.pi)
    c = (-1/2)*np.log(det(E)) + np.log(1/len(x))

    return (a+b+c)

# generate a single point 
temp = g(np.array([0,0]),np.mean(data,axis=0),np.cov(data.T))

## c
# Generate a 2D dataset with three classes and use the quadratic classifier 
# above to learn the parameters and make predictions. 
# As an example, you should generate training data shown
# below to estimate the the parameters of the classifier in (1) 
# and you should test_data the classifieron another randomly generated dataset. 
# It is also sufficient to show the dataset used to train
# your classifier and the decision boundary it produces

# create training set
# means and covariances
u1 = [-3,0]
E1 = E
u2 = [0,3]
E2 = E
u3 = [3,0]
E3 = E

# training data 
train_1 = N(u1,E1,size= num_samples)
train_2 = N(u2,E2,size= num_samples)
train_3 = N(u3,E3,size= num_samples)

# appended training data, means, and covariance matricies for custom g(x)
u_train = []
u_train.append(np.mean(train_1,axis=0))
u_train.append(np.mean(train_2,axis=0))
u_train.append(np.mean(train_3,axis=0))
E_train = []
E_train.append(np.cov(train_1.T))
E_train.append(np.cov(train_2.T))
E_train.append(np.cov(train_3.T))

# general format of the following plotting 
# in order to keep the plots open
plt.ion()
# creates a figure
f1 = plt.figure(1)
# plots the data 
p1 = plt.plot(train_1[:,0], train_1[:,1], 'o', c='r')
p2 = plt.plot(train_2[:,0], train_2[:,1], 'o', c='g')
p3 = plt.plot(train_3[:,0], train_3[:,1], 'o', c='b')
# plotting options
plt.axis('equal')
plt.title('Training data w/ labels')
plt.legend(['w1','w2','w3'])
f1.show()

# generate new dataset (300 samples long matrix)
test_data = N(u1,E1,size= num_samples)
test_data = np.append(test_data,N(u2,E2,size= num_samples),axis=0)
test_data = np.append(test_data,N(u3,E3,size= num_samples),axis=0)


# every other classification figure follows this same process
# make a new figure
f2 = plt.figure(2)
# enumerate through each point in the test data set
for i, each_point in enumerate(test_data):
    # create a data set
    this_prediction = np.zeros((3,))
    # for each "model" calculate g(x) of for that point for each training set
    for j in range(3):
        this_prediction[j] = g(each_point,u_train[j],E_train[j])
    # find which condition it applies to 
    condition = np.argmax(this_prediction)
    # plot that point with the asociated condition 
    if condition == 0:
        plt.plot(each_point[0],each_point[1],'o',c='r')
    elif condition == 1:
        plt.plot(each_point[0],each_point[1],'o',c='g')
    elif condition == 2:
        plt.plot(each_point[0],each_point[1],'o',c='b')
# plot options 
plt.axis('equal')
pop_a = mpatches.Patch(color='r', label='Population A')
pop_b = mpatches.Patch(color='g', label='Population B')
pop_c = mpatches.Patch(color='b', label='Population C')
plt.title('Training data sorted with given quadratic estimator')
plt.legend(handles=[pop_a,pop_b,pop_c])
f2.show()


# use the boundary conditions to create a grid alignment to show the 
# quadratic esimators points without having to draw complicated lines 
minX, maxX, minY, maxY = -6., 6., -6., 6.
x = np.linspace(int(minX), int(maxX), 20)
y = np.linspace(int(minX), int(maxX), 20)
# create the mesh based on these arrays
X, Y = np.meshgrid(x, y)
grid_test_data = np.vstack((X.flatten(),Y.flatten())).T

# plot the meshed grid that has been sorted
f3 = plt.figure(3)
for i, each_point in enumerate(grid_test_data):
    this_prediction = np.zeros((3,))
    for j in range(3):
        this_prediction[j] = g(each_point,u_train[j],E_train[j])
    condition = np.argmax(this_prediction)
    if condition == 0:
        plt.plot(each_point[0],each_point[1],'o',c='r')
    elif condition == 1:
        plt.plot(each_point[0],each_point[1],'o',c='g')
    elif condition == 2:
        plt.plot(each_point[0],each_point[1],'o',c='b')
plt.axis('equal')
pop_a = mpatches.Patch(color='r', label='Population A')
pop_b = mpatches.Patch(color='g', label='Population B')
pop_c = mpatches.Patch(color='b', label='Population C')
plt.title('Training data sorted with given quadratic estimator showing ')
plt.legend(handles=[pop_a,pop_b,pop_c])
f3.show()

## d
# write a procedure for computing the mahalanobis distance between a point x
# and some mean vector u, given a covariance matrix E

# mean and dataset creations 
u_mahal = [0,0]
d4 = N(u_mahal,E,size= num_samples)

def d_Mahal(x,u,E):

    dist = np.matmul(np.matmul((x-u).T,inv(E)),(x-u))

    return(dist)

# create an example 
d_Mahal_example = d_Mahal(d4[0],u_mahal,E)

## e 
# Implement the naive Bayes classifier, and compare results to pythons built in
# use different means,cov,prior probabilities to demonstrate that your implementation
# is correct

# using the numpy gaussian naive bayes 
# P(xi|y) = 1/(sqrt(2*pi*sigma_y^2)) * exp(-(xi-u_y)^2/(2*sigma_y^2))
# is equal to 
# P(xi|y) = 1/(sqrt(2*pi*det(E))) * exp((1/2)*(xi-u_y).T*inv(E)*(xi-u_y))

def naive_bayes_custom(x,u,E):

    a = np.array(1/(np.sqrt(2*np.pi*det(E))))
    b = d_Mahal(x,u,E)
    p_x = a*b

    return(p_x)

# create 2 arrays
# all the training data as a single 2d array
training_data = np.append(np.append(train_1,train_2,axis=0),train_3,axis=0)
# all the labels for the training data (0,1,2)
training_labels = np.append(np.append(np.zeros((100,)),(np.ones((100,))),axis=0),(np.ones((100,))+1),axis=0)

# use the scipi GaussianNB() function to generate predictions on the test_data
naive_bayes_builtin_predict = GaussianNB().fit(training_data,training_labels).predict(test_data)

# implement the custom naive bayes model
f4 = plt.figure(4)
for i, each_point in enumerate(test_data):
    this_prediction = np.zeros((3,))
    for j in range(3):
        this_prediction[j] = naive_bayes_custom(each_point,u_train[j],E_train[j])
    condition = np.argmin(this_prediction)
    if condition == 0:
        plt.plot(each_point[0],each_point[1],'o',c='r')
    elif condition == 1:
        plt.plot(each_point[0],each_point[1],'o',c='g')
    elif condition == 2:
        plt.plot(each_point[0],each_point[1],'o',c='b')
plt.axis('equal')
pop_a = mpatches.Patch(color='r', label='Population A')
pop_b = mpatches.Patch(color='g', label='Population B')
pop_c = mpatches.Patch(color='b', label='Population C')
plt.title('Training data sorted with custom Naive Bayes')
plt.legend(handles=[pop_a,pop_b,pop_c])
f4.show()

# plot the built in naive bayes model 
f5 = plt.figure(5)
for i, each_point in enumerate(test_data):
    this_prediction = np.zeros((3,))
    condition = naive_bayes_builtin_predict[i]
    if condition == 0:
        plt.plot(each_point[0],each_point[1],'o',c='r')
    elif condition == 1:
        plt.plot(each_point[0],each_point[1],'o',c='g')
    elif condition == 2:
        plt.plot(each_point[0],each_point[1],'o',c='b')
plt.axis('equal')
pop_a = mpatches.Patch(color='r', label='Population A')
pop_b = mpatches.Patch(color='g', label='Population B')
pop_c = mpatches.Patch(color='b', label='Population C')
plt.title('Training data sorted with built in Naive Bayes')
plt.legend(handles=[pop_a,pop_b,pop_c])
f5.show()




### 3 
# Let the set N_ints ∈[1,...,n,...,N] be a set of integers and p be a probability 
# distribution p_ints = [p1,...,pn,...,pN] such that pk is the probability 
# of observing k∈ N. Note that since p is a distribution then 1Tp = 1 and
# 0≤pk≤1∀n. Write a function sample(M,p) that returns M indices sampled 
# from the distribution p. Provide evidence that your function is working as desired. 
# Note that all sampling is assumed to be i.i.d. You must include a couple of 
# paragraphs and documented code that discusses how you were able to 
# accomplish this task


# assuming that i.i.d is Independemt and identially distributed random variable sample
# assuming that each variable has equal probalility and are mutually independent
N_ints = np.random.randint(1,101, 100000)

# assume that N_ints is perfectly unifrom (it is very close and can be approximately shown in the
# plotted histogram) 
p_uniform = np.array(np.ones((np.max(N_ints),))/np.max(N_ints))

# for number of samples M and probability vector p
# generates M iid samples from probability p 
def sample(M,p_in):

    # this function generates a random sample from the list of integers (1:N)
    # given the probabilities p_in 
    # N is a sorted integer list the same length of p 
    # where max(N) == length(p)+1   (because of indexing from 0)
    out = numpy.random.choice(numpy.arange(1, len(p_in)+1), size=M,p=p_in) 

    return out


M = 1000
temp = sample(M,p_uniform)

# in order to prove the uniformity of the random samples from p_k
# create a show of the histogram mapping from the taken samples
f6 = plt.figure(6)
n, bins, patches = plt.hist(temp, len(temp),density=True,facecolor='r')
plt.xlabel('histogram bins')
plt.ylabel('Probability')
plt.title('Histogram of sample(M,p)')
plt.xlim(0, np.max(temp))
plt.ylim(0, np.max(n))
plt.grid(True)
f6.show()

plt.ioff()
plt.show()
print('end of program')

# plt.close('all')