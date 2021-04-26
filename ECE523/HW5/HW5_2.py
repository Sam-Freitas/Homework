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
import glob

def get_acc(predictions,labels):

    conf_mat = confusion_matrix(predictions, labels)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return acc

def covert_to_lables(predictions):

    labels = np.zeros(shape=(len(predictions)))

    for count, this_prediction in enumerate(predictions):
        labels[count] = np.argmax(this_prediction)+1
    return labels

def get_model(data_input,binned_labels_output):

    input_size = data_input.shape
    label_size = binned_labels_output.shape

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

    return model

def bin_these_labels(labels):
    unique_labels = np.unique(labels)
    unique_labels2 = np.append(unique_labels,np.max(unique_labels)+1)

    binned_labels = label_binarize(labels,classes= unique_labels2)
    binned_labels = binned_labels[:,0:int(np.max(unique_labels2))]

    return(binned_labels)

csv_paths = glob.glob(os.path.join(os.getcwd(),'*.csv'))
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10,restore_best_weights=True)
num_epochs = 100
batch_size = 256

for count,this_csv in enumerate(csv_paths):

    print('Dataset:',os.path.basename(csv_paths[count]))

    this_data_whole = np.genfromtxt(this_csv,delimiter=',')

    this_labels = this_data_whole[:,-1]

    this_unlabeled_data = this_data_whole[:,0:(this_data_whole.shape[1]-1)]

    print('Found',len(this_labels),'data points with', this_unlabeled_data.shape[1], 'features and', len(np.unique(this_labels)), 'separate labels')

    binned_labels = bin_these_labels(this_labels)

    this_model = get_model(this_unlabeled_data,binned_labels)

    acc_storage = []

    idx_list = list(range(len(this_unlabeled_data)))
    num_to_give_labels = round(len(this_unlabeled_data)*(.15*5))
    k_fold_choices = random.choices(idx_list,k=num_to_give_labels)

    k_fold_choices_split5 = np.array_split(k_fold_choices,5)

    print('Using',len(k_fold_choices_split5[0]),'data points per k-fold')
    
    for count,this_idx in enumerate(k_fold_choices_split5):

        this_model = tf.keras.models.load_model('my_model.h5')

        this_X = this_unlabeled_data[this_idx,:]
        this_y = binned_labels[this_idx]

        history = this_model.fit(x=this_X, y=this_y,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(this_X, this_y),
                    verbose=0,
                    callbacks=[callback])
        
        kfold_predictions = this_model.predict(this_unlabeled_data)
        kfold_prediction_labels = covert_to_lables(kfold_predictions)
        kfold_this_accuracy = get_acc(kfold_prediction_labels,this_labels)

        acc_storage.append(kfold_this_accuracy)

        print('Accuracy of k-model',count+1,'on entire data set is:', kfold_this_accuracy)
