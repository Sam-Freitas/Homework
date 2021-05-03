#Courtney Comrie and Sam Freitas
#ECE 523 Final Project
#UNET-CNN for segmenting brains from the skull

import numpy as np #import needed libraries and commands
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from cv2 import imread
import os
import sys
from tqdm import tqdm
import random
import warnings
from itertools import chain
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf

plt.ion() #turn ploting on

#dataset_path = r"C:\Users\cjoy1\Documents\Grad School\Second Year Spring\ECE 523\Homework\Final_Project\mr_images_png" #initialize paths and dataset
dataset_path = os.getcwd()
image_path = os.path.join(dataset_path, "images")
mask_path = os.path.join(dataset_path,"masks")
dataset = pd.read_csv('dataset.csv')

def data_generator(dataset, image_path, mask_path, height, width): #function for generating data
    X_train = np.zeros((len(dataset),height,width,3), dtype = np.uint8) #initialize training sets (and testing sets)
    y_train = np.zeros((len(dataset),height,width,1), dtype = np.uint8)

    sys.stdout.flush() #write everything to buffer ontime 

    for i in tqdm(range(len(dataset)),total=len(dataset)): #iterate through datatset and build X_train,y_train

        new_image_path = os.path.join(image_path,dataset.iloc[i][0])
        new_mask_path = os.path.join(mask_path,dataset.iloc[i][1])

        image = imread(new_image_path)
        mask = imread(new_mask_path)[:,:,:1]

        img_resized = resize(image,(height,width), mode = 'constant',preserve_range = True)
        mask_resized = resize(mask, (height,width), mode = 'constant', preserve_range = True)


        X_train[i] = img_resized
        y_train[i] = mask_resized

    return X_train, y_train

def unet_cnn(height,width,channels): #Unet-cnn model 
    inputs = Input((height, width, channels))
    s = Lambda(lambda x: x/255)(inputs)

    con1 = Conv2D(16, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same') (s)
    con1 = Dropout(0.1)(con1)
    con1 = Conv2D(16, (3,3), activation = 'elu', kernel_initializer = 'he_normal', padding = 'same')(con1)
    pool1 = MaxPooling2D((2,2))(con1)

    con2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool1)
    con2 = Dropout(0.1) (con2)
    con2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con2)
    pool2 = MaxPooling2D((2, 2)) (con2)

    con3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool2)
    con3 = Dropout(0.2) (con3)
    con3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con3)
    pool3 = MaxPooling2D((2, 2)) (con3)

    con4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool3)
    con4 = Dropout(0.2) (con4)
    con4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (con4)

    con5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (pool4)
    con5 = Dropout(0.3) (con5)
    con5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (con5)
    u6 = concatenate([u6, con4])
    con6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    con6 = Dropout(0.2) (con6)
    con6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (con6)
    u7 = concatenate([u7, con3])
    con7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    con7 = Dropout(0.2) (con7)
    con7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (con7)
    u8 = concatenate([u8, con2])
    con8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    con8 = Dropout(0.1) (con8)
    con8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (con8)
    u9 = concatenate([u9, con1], axis=3)
    con9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    con9 = Dropout(0.1) (con9)
    con9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (con9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (con9)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return(model)

def plot_figures(image,orig_mask,pred_mask,num): #function for plotting figures
    plt.figure(num,figsize=(12,12))
    plt.subplot(131)
    plt.imshow(image)
    plt.title("MR Image")
    plt.subplot(132)
    plt.imshow(orig_mask.squeeze(),cmap='gray')
    plt.title("Original Mask")
    plt.subplot(133)
    plt.imshow(pred_mask.squeeze(),cmap='gray')
    plt.title('Predicted Mask')

def plot_acc_loss(results): #plot accuracy and loss
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
        
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 

total = len(dataset) #set variables
test_split = 0.2
height = 128
width = 128
channels = 3 
batch_size = 32

#######Training
train, test = train_test_split(dataset, test_size = test_split, random_state = 50) #randomly split up the test and training datasets
X_train, y_train = data_generator(train, image_path, mask_path, height, width) #set up training data
y_train = y_train / 255 #thresh y_training set

model = unet_cnn(height,width,channels) #call unet model function and compile 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly = True)
model.summary() #display model summary

model_path = "nuclei_finder_unet_1.h5" #store model here
checkpoint = ModelCheckpoint(model_path,monitor="val_loss",mode="min",save_best_only = True,verbose=1) #use checkpoint instead of sequential() module
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 1,restore_best_weights = True) #stop at best epoch
results = model.fit(X_train, y_train, validation_split=0.1, batch_size=32, epochs=100,callbacks=[earlystop, checkpoint]) #fit model

plot_acc_loss(results) #plot the accuracy and loss functions

model = load_model('nuclei_finder_unet_1.h5') #load weights
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1) 
preds_train_t = (preds_train > 0.5).astype(np.uint8) #predict mask
ix = random.randint(1, 10)
plot_figures(X_train[ix],y_train[ix],preds_train[ix], 1) #plot images and masks

#######Testing

#model = load_model("nuclei_finder_unet_1.h5") #reload model for testing
X_test,y_test = data_generator(test,image_path, mask_path,height,width) #get test set
y_test = y_test / 255 #thresh y_test
results = model.evaluate(X_test,y_test,steps=1) #get evaluation results

count = 2 #counter for figures in for loops
for image,mask in zip(X_test,y_test): #for loop for plotting images
    
    img = image.reshape((1,height,width,channels)).astype(np.uint8)
    pred_mask = model.predict(img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)

    plot_figures(image,mask,pred_mask, count)
    count += 1

    if count>10:
        break

plt.ioff()
plt.show()
