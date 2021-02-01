import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import Tuple, List, Dict

import tensorflow

import tensorflow as tf

from numpy import floor,log


def create_toy_rnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for a toy problem.

    The network will take as input a sequence of number pairs, (x_{t}, y_{t}),
    where t is the time step. It must learn to produce x_{t-3} - y{t} as the
    output of time step t.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    # initalize variables 
    o_activation = 'linear'
    thisLoss = 'MSE'

    # create a sequential model
    model = tf.keras.models.Sequential()
    # create a LSTM RNN network
    model.add(tf.keras.layers.LSTM(20,input_shape=input_shape, return_sequences=True, activation='tanh'))
    # create an output layer 
    model.add(tf.keras.layers.Dense(10,activation=o_activation))
    # specify which optimizer to use
    thisOptimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    # compile the model
    model.compile(loss=thisLoss, optimizer=thisOptimizer, metrics=['accuracy'])

    # create keyword arguments 
    kwargs = {"batch_size": 1,"verbose":0}

    return([model,kwargs])


def create_mnist_cnn(input_shape: tuple, n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for digit classification.

    The network will take as input a 28x28 grayscale image, and produce as
    output one of the digits 0 through 9. The network will be trained and tested
    on a fraction of the MNIST data: http://yann.lecun.com/exdb/mnist/

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param input_shape: The shape of the inputs to the model.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """

    # Create the optimizer  
    rms = tf.keras.optimizers.RMSprop(learning_rate=0.005)#1, rho=0.9, epsilon=None, decay=0.0)


    # create sequential model 
    model = tf.keras.models.Sequential()
    # create convolutional layer
    model.add(tf.keras.layers.Conv2D(64, (5,5), activation='relu', input_shape=(28,28,1)))
    # pool 
    model.add(tf.keras.layers.MaxPool2D((3,3)))
    # flatten
    model.add(tf.keras.layers.Flatten())
    # create a dense layer 
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    # add a dropout layer 
    model.add(tf.keras.layers.Dropout(0.5))
    # normalize before final processing 
    model.add(tf.keras.layers.BatchNormalization())
    # final output layer 
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=rms,
                metrics=['accuracy'])
    # send keywords 
    kwargs = {"batch_size": 128,"verbose":0}

    return([model,kwargs])


def create_youtube_comment_rnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a recurrent neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # create optimizer 
    adam = tf.keras.optimizers.Adam(learning_rate=0.005)

    # create sequential model
    model = tf.keras.models.Sequential()
    # embedd te vocabulary 
    model.add(tf.keras.layers.Embedding(len(vocabulary), 64))
    # add a bidirectional LSTM 
    #, activation='relu', return_sequences=False
    model.add(tf.keras.layers.Bidirectional((tf.keras.layers.LSTM(15))))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    # add a dense output layer 
    model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    # compile the model
    model.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

    kwargs = {"batch_size": 128,"verbose":0}
    return ([model,kwargs])


def create_youtube_comment_cnn(vocabulary: List[str], n_outputs: int) \
        -> Tuple[tensorflow.keras.models.Model, Dict]:
    """Creates a convolutional neural network for spam classification.

    This network will take as input a YouTube comment, and produce as output
    either 1, for spam, or 0, for ham (non-spam). The network will be trained
    and tested on data from:
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection

    Each comment is represented as a series of tokens, with each token
    represented by a number, which is its index in the vocabulary. Note that
    comments may be of variable length, so in the input matrix, comments with
    fewer tokens than the matrix width will be right-padded with zeros.

    This method does not call Model.fit, but the dictionary it returns alongside
    the model will be passed as extra arguments whenever Model.fit is called.
    This can be used to, for example, set the batch size or use early stopping.

    :param vocabulary: The vocabulary defining token indexes.
    :param n_outputs: The number of outputs from the model.
    :return: A tuple of (neural network, Model.fit keyword arguments)
    """
    # create optimizer 
    adam = tf.keras.optimizers.Adam(learning_rate=0.005)

    # create sequential model
    model = tf.keras.models.Sequential() 
    # embedd te vocabulary 
    # model.add(tf.keras.layers.Embedding(len(vocabulary), 64))
    model.add(tf.keras.layers.Embedding(len(vocabulary), 64 ))
    # create 1D convolutional layer
    model.add(tf.keras.layers.Conv1D(16,2))#,padding='valid',activation='relu')
    # pool 
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    # flatten 
    model.add(tf.keras.layers.Flatten())
    # create a dense processing layer
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    # create the output layer
    model.add(tf.keras.layers.Dense(n_outputs, activation='sigmoid'))
    # compile the model
    model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=['accuracy'])
    # send keywords 
    kwargs = {"batch_size": 128,"verbose":0}
    return ([model,kwargs])
