import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random

def plot_2D_labeled_data(X,y,fig_number,fig_title):
    # put plt.ioff() and plt.show() at end 

    colors = cm.cool(y/max(y))

    plt.ion()
    f = plt.figure(fig_number)
    plt.scatter(X[:,0],X[:,1],c=colors,alpha=0.5)
    # plt.axis('tight')
    plt.title(fig_title)
    f.show()
    return fig_number+1

def get_acc(predictions,labels):

    conf_mat = confusion_matrix(predictions, labels)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return acc

def covert_to_lables(predictions):

    labels = np.zeros(shape=(len(predictions)))

    for count, this_prediction in enumerate(predictions):
        labels[count] = np.argmax(this_prediction)+1
    return labels

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

unlabeled_data = np.append(d1,d2,axis=0)
labels = np.append(l1,l2)

# plot the training data
fig_num = plot_2D_labeled_data(unlabeled_data,labels,1,"Training data")

# create labels
unique_labels = np.unique(labels)
unique_labels2 = np.append(unique_labels,np.max(unique_labels)+1)

binned_labels = label_binarize(labels,classes= unique_labels2)
binned_labels = binned_labels[:,0:int(np.max(unique_labels))]

input_size = unlabeled_data.shape
label_size = binned_labels.shape
print("input shape",input_size)

# Call neural network API
model = tf.keras.Sequential()
# Apply linear activation function to input layer
# Generate hidden layer with 14 nodes, the same as the input layer
model.add(layers.Dense(units=128, activation='linear',input_dim=input_size[1]))
model.add(layers.Dense(units=15, activation='relu'))
model.add(layers.Dropout(0.2))
# Apply linear activation function to hidden layer
# Generate output layer with 14 nodes
model.add(layers.Dense(units=label_size[1], activation='sigmoid'))
# Compile the model
model.compile(optimizer='SGD',
              loss='CategoricalCrossentropy',
              metrics=['accuracy'])

model.save('my_model.h5')

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10,restore_best_weights=True)

# Train the model
num_epochs = 50
batch_size = 256

percent_given_labels = [.1,.25]

for count, val in enumerate(percent_given_labels):

    num_to_give_labels = round(len(unlabeled_data)*val)
    idx_list = list(range(len(unlabeled_data)))

    idx = random.choices(idx_list,k=num_to_give_labels)

    this_X = unlabeled_data[idx,:]
    this_y = binned_labels[idx]

    model = tf.keras.models.load_model('my_model.h5')

    history = model.fit(x=this_X, y=this_y,
                        epochs=num_epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(this_X, this_y),
                        verbose=0,
                        callbacks=[callback])

    acc_hist = history.history['accuracy']
    err_hist = 1-np.asarray(acc_hist)

    print('Using ', val, ' of data')

    print('(1) - first classifier training error with only assigned labels:', err_hist[0])

    num_epochs_used = len(acc_hist)
    middle_epoch = round(num_epochs_used/2)
    
    print('(2) - error at time point epoch num:', middle_epoch, ' - gives error:', err_hist[middle_epoch])
    
    predictions = model.predict(unlabeled_data)

    prediction_labels = covert_to_lables(predictions)

    this_accuracy = get_acc(prediction_labels,labels)

    print('(3) - Error after entire SSL :', np.min(err_hist))

    print('Accuracy of model on entire data set is :', this_accuracy)

    fig_title = "Predictions using" + str(val) + " of data with acc:" + str(this_accuracy) + "accurate"

    fig_num = plot_2D_labeled_data(unlabeled_data,prediction_labels,fig_num,fig_title)


plt.ioff()
plt.show()