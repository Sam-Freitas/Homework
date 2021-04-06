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
    You will need tochoose a suitable threshold to determine the data samples 
    that will be labeled for the nextround of self-training.
•Report the error of the self-training algorithm on the testing data at: 
    (1) the first time a classifier is trained using only the labeled; 
    (2) at least one timepoint during the self training process (i.e., when pseudo labels are used); and 
    (3) after self-training is completed. Comment on the results.
•Perform an experiment reporting the above requirements with 10% and when 25% of thetraining data are labeled.
"""

import numpy as np 

mean1 = [-1,-1]
mean2 = [1,1]
cov = [[1,0],[0,1]]
d1 = np.random.multivariate_normal(mean1, cov, 10000)
d2 = np.random.multivariate_normal(mean2, cov, 10000)


