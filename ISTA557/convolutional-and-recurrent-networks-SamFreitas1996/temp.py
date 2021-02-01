import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json

import h5py
import numpy as np
import pytest
import tensorflow
import tensorflow as tf
import logging

def layers(model: tensorflow.keras.models.Model):
    return [x.layer if isinstance(x, tensorflow.keras.layers.Wrapper) else x
            for x in model.layers]


def is_convolution(layer: tensorflow.keras.layers.Layer):
    return isinstance(layer, tensorflow.python.keras.layers.convolutional.Conv)


def is_recurrent(layer: tensorflow.keras.layers.Layer):
    return isinstance(layer, tensorflow.keras.layers.RNN)


def loss(model):
    if isinstance(model.loss, str):
        return getattr(tensorflow.keras.losses, model.loss)
    else:
        return model.loss


def output_activation(model: tensorflow.keras.models.Model):
    return model.layers[-1].activation


def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean() ** 0.5


def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))


def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)


os.environ["TF_DETERMINISTIC_OPS"] = "1"
tensorflow.random.set_seed(42)
tensorflow.config.threading.set_intra_op_parallelism_threads(1)
tensorflow.config.threading.set_inter_op_parallelism_threads(1)

with h5py.File("data/mnist.hdf5", 'r') as f:
    train = f["train"]
    train_out = np.array(train["output"])
    train_in = np.array(train["input"])
    test = f["test"]
    test_out = np.array(test["output"])
    test_in = np.array(test["input"])

# request a model
input_shape = train_in.shape[1:]
(_, n_outputs) = train_out.shape

#model = create_custom_nn('rnn',n_hidden_layers,n_units,input_shape,n_outputs,h_activation,o_activation,thisLoss,thisOptimizer)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((3,3)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))
rms = tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
                optimizer=rms,
                metrics=['accuracy'])
kwargs = {"batch_size": 128}

# check that model contains a convolutional layer
assert any(is_convolution(layer) for layer in layers(model))

# check that model contains no recurrent layers
assert all(not is_recurrent(layer) for layer in layers(model))

# check that output type and loss are appropriate
assert "categorical" in loss(model).__name__
assert output_activation(model) == tensorflow.keras.activations.softmax

# set training data, epochs and validation data
kwargs.update(x=train_in, y=train_out,
                epochs=10, validation_data=(test_in, test_out))

# call fit, including any arguments supplied alongside the model
model.fit(**kwargs)

# make sure accuracy is high enough
accuracy = multi_class_accuracy(model.predict(test_in), test_out)
print("\n{:.1%} accuracy for CNN on MNIST sample".format(accuracy))
assert accuracy > 0.8

print('hello there')


