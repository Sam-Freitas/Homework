# HW 4 problem 2
# Samuel Freitas
# ECE 523 

"""
Write a class that implements the Adaboost algorithm. 
Your class should be similar to sklearn’s 
in that it should have a fitand predict method to train and test the 
classifier, respectively. You should also use the sampling function
from Homework #1 to train the weak learning algorithm,
which should be a shallow decision tree.  
The Adaboost class should be compared to sklearn’s 
implementation on datasets from the course Github page.
"""

import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification,make_multilabel_classification,make_sparse_spd_matrix
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import multivariate_normal as N


# Decision stump used as weak classifier
class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):

        M = X.shape[0]
        N_ints = np.random.randint(1,M+1, M)
        p_uniform = np.array(np.ones((np.max(N_ints),))/np.max(N_ints))

        rand_choice = np.random.choice(np.arange(1, len(p_uniform)+1), size=1,p=p_uniform) 

        this_threshold = X[rand_choice][0][0]

        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < this_threshold] = -1
        else:
            predictions[X_column > this_threshold] = -1

        return predictions


class Adaboost():

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            # greedy search to find best threshold and feature
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = .5
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # calculate predictions and update weights
            predictions = clf.predict(X)

            w *= np.exp(-clf.alpha * y * predictions)
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


def plot_2D_labeled_data(X,y,fig_number,fig_title):
    # put plt.ioff() and plt.show() at end 
    plt.ion()
    f = plt.figure(fig_number)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.title(fig_title)
    f.show()

def gen_multilabel_data(num_labels,num_samples_per,size_constraint):
    for i in range(num_labels):
        this_center = np.random.randint(size_constraint,size=2)
        # this_cov_array = make_sparse_spd_matrix(2)
        this_cov_array = np.array([[1,0],[0,1]])

        dX = N(this_center,this_cov_array,size = num_samples_per)
        dy = np.zeros((1,num_samples_per))+i
        
        if i == 0:
            X = dX
            y = dy
        else:
            X = np.append(X,dX,0)
            y = np.append(y,dy)

    return(X,y)

X,y = gen_multilabel_data(num_labels = 2,num_samples_per = 100,size_constraint = 4)

plot_2D_labeled_data(X,y,1,"Given data")

clf = Adaboost()

clf.fit(X,y)

y2 = clf.predict(X)

plot_2D_labeled_data(X,y2,2,"Adaboost data")

print("end")
plt.ioff()
plt.show()