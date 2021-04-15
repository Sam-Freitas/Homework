# HW5 problem 1
# Samuel Freitas
# ECE 523

"""
Generate 2D Gaussian data that is similar to the data set shown in Figure 1. 

Using data sets that have 1000 samples (500 from each class) implement the self-training algorithm, 
and test on a dataset of 1000 samples. The requirements for this problem are as follows:

•Your implementation of self-training must use a classifier (e.g., neural network) that can give probabilities 
    to select the data points that will have pseudo labels assigned to them. 
    I will not make a restriction on the classifier other than the probabilities requirement. 
    You will need to choose a suitable threshold to determine the data samples 
    that will be labeled for the nextround of self-training.
•Report the error of the self-training algorithm on the testing data at: 
    (1) the first time a classifier is trained using only the labeled; 
    (2) at least one timepoint during the self training process (i.e., when pseudo labels are used); and 
    (3) after self-training is completed. Comment on the results.
•Perform an experiment reporting the above requirements with 10% and when 25% of the training data are labeled.
"""

import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from fit_module import FitModule

def plot_2D_labeled_data(X,y,fig_number,fig_title):
    # put plt.ioff() and plt.show() at end 
    plt.ion()
    f = plt.figure(fig_number)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.title(fig_title)
    f.show()

# generate data constraints
mean1 = [-1.1,-1.1]
mean2 = [1.1,1.1]
cov = [[0.95,0.05],[0.05,0.95]]

num_points_per = 500

# generate data 
d1 = np.random.multivariate_normal(mean1, cov, num_points_per)
d2 = np.random.multivariate_normal(mean2, cov, num_points_per)

# generate labels
l1 = np.zeros((1,num_points_per))+1
l2 = np.zeros((1,num_points_per))+2

# plot the training data
plot_2D_labeled_data(np.append(d1,d2,axis = 0),np.append(l1,l2),1,"Training data")

# create the unlabeled dataset
unlabeled_data = np.append(d1,d1,axis=0)

input_size = unlabeled_data.shape


        
model = Net()
            
# criterion1 = nn.MSELoss()
# criterion2 = nn.BCELoss()
            
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# epochs = 150
   
# for epoch in range(epochs): 
#     for inputs, labels in train_loader:
#         inputs = Variable(inputs)
#         labels = Variable(labels)       
#         optimizer.zero_grad()
#         decoded, out = model(inputs)
       
#         loss1 = criterion1(decoded, inputs) 
#         loss2 = criterion2(out, labels)
#         loss = loss1 + loss2
#         loss.backward()
#         optimizer.step()







plt.ioff()
plt.show()

print("end")