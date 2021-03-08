# Samuel Freitas
# 3/8/21
# ECE 523

# problem #3 

import numpy as np
import sklearn
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm
from numpy.random import rand
import cvxopt
from cvxopt.solvers import qp
from cvxopt import matrix

def plot_2D_labeled_data(X,y,fig_number,fig_title):
    # put plt.ioff() and plt.show() at end 
    plt.ion()
    f = plt.figure(fig_number)
    plt.scatter(X[:,0],X[:,1],c=y)
    plt.axis('equal')
    plt.title(fig_title)
    f.show()

# read in data from csv, split into data and labels
source_csv = genfromtxt('source_train.csv', delimiter=',')
source_labels = source_csv[:,2]
source_data = source_csv[:,0:2]
target_csv = genfromtxt('target_train.csv', delimiter=',')
target_labels = target_csv[:,2]
target_data = target_csv[:,0:2]

C = 10
B = 1
n = len(target_labels)
svc_source = svm.SVC(kernel='linear', C=C).fit(source_data, source_labels)
Ws = svc_source.coef_[0]

WsT = np.full((50,2),Ws)
q = matrix((target_labels.dot(WsT)).dot(target_data.T))

G = matrix(0.0, (n,n))
G[::n+1] = -1.0
h = matrix(0.0, (n,1))
A = matrix(1.0, (1,n))
b = matrix(1.0)

P = matrix(np.zeros((len(target_data),len(target_data))))
for i in range(len(target_data)):
    for j in range(len(target_data)):
        a1 = target_labels[i]*target_labels[j]
        a2 = (target_data[i].T).dot(target_data[j])
        P[i,j] = a1*a2

solv = qp(P,q,G,h)

ai = np.asarray(solv['x'])

np.argmin(np.asarray(solv['x']))

svc_target = svm.SVC(kernel='linear', C=C).fit(target_data, target_labels)
Wt = svc_target.coef_[0]




plot_2D_labeled_data(source_data,source_labels,1,'source data')
plot_2D_labeled_data(source_data,svc_source.predict(source_data),2,'source data SVM predicted')

plot_2D_labeled_data(target_data,target_labels,3,'target data')
plot_2D_labeled_data(target_data,svc_target.predict(target_data),4,'target data SVM predicted')

print('Ws:',Ws)
print('Wt:',Wt)

plt.ioff()
plt.show()