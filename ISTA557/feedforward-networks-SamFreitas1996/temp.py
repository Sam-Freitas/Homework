import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import List

import h5py
import numpy as np
import pytest
import tensorflow
import tensorflow as tf
import logging

import numpy as np

import nn
print('Hello there')

def set_seeds():
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tensorflow.random.set_seed(42)
    tensorflow.config.threading.set_intra_op_parallelism_threads(1)
    tensorflow.config.threading.set_inter_op_parallelism_threads(1)

def load_hdf5(path):
    with h5py.File(path, 'r') as f:
        train = f["train"]
        train_out = np.array(train["output"])
        train_in = np.array(train["input"])
        test = f["test"]
        test_out = np.array(test["output"])
        test_in = np.array(test["input"])
    return train_in, train_out, test_in, test_out
def assert_layers_equal(layers1: List[tensorflow.keras.layers.Layer],
                        layers2: List[tensorflow.keras.layers.Layer]):
    def layer_info(layer):
        return (layer.__class__,
                getattr(layer, "units", None),
                getattr(layer, "activation", None))

    assert [layer_info(l) for l in layers1] == [layer_info(l) for l in layers2]
def assert_compile_parameters_equal(model1: tensorflow.keras.models.Model,
                                    model2: tensorflow.keras.models.Model):
    def to_dict(obj):
        return dict(__class__=obj.__class__.__name__, **vars(obj))

    assert to_dict(model1.optimizer) == to_dict(model2.optimizer)
def loss(model):
    if isinstance(model.loss, str):
        return getattr(tensorflow.keras.losses, model.loss)
    else:
        return model.loss
def hidden_activations(model):
    return [layer.activation
            for layer in model.layers[:-1] if hasattr(layer, "activation")]
def output_activation(model):
    return model.layers[-1].activation
def root_mean_squared_error(system: np.ndarray, human: np.ndarray):
    return ((system - human) ** 2).mean(axis=0) ** 0.5
def multi_class_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.argmax(system, axis=1) == np.argmax(human, axis=1))
def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)

def create_network(n_hidden_layers,n_units,n_inputs,n_outputs,h_activation,o_activation,thisLoss,thisOptimizer,dropout_bool,dropout_rate):

    # Create a sequential model
    thisModel = tf.keras.models.Sequential()
    # create the input layer
    thisModel.add(tensorflow.keras.layers.Dense(n_inputs, input_shape=(n_inputs,),activation=h_activation))
    # create each hidden layer
    for i in range(n_hidden_layers):
        thisModel.add(tf.keras.layers.Dense(n_units,activation=h_activation))
        if dropout_bool:
            thisModel.add(tf.keras.layers.Dropout(dropout_rate))
    # create the output layer
    thisModel.add(tf.keras.layers.Dense(n_outputs,activation=o_activation))
    # compile the model 
    thisModel.compile(optimizer=thisOptimizer,loss=thisLoss)

    # thisModel.summary()

    return thisModel


train_in, train_out, test_in, test_out = load_hdf5("data/income.hdf5")

# keep only every 10th training example
train_out = train_out[::10, :]
train_in = train_in[::10, :]

n_inputs, n_outputs = train_in.shape[-1], train_out.shape[-1]


# for thisIdx,thisNum in enumerate(range(1,10)):

# thisMom = np.arange(0,.2,.02).tolist()

# for i in range(len(thisMom)):

thisCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',patience=3)#min_delta=0.001)#patience=8

early_fit_kwargs = {"callbacks":thisCallback}
late_fit_kwargs = {}

mySgd = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.1, nesterov=True, name='SGD')

early= create_network(1,16,n_inputs,n_outputs,'tanh','sigmoid','binary_crossentropy',mySgd,1,0.2)
late = create_network(1,16,n_inputs,n_outputs,'tanh','sigmoid','binary_crossentropy',mySgd,1,0.2)



late_fit_kwargs.update(verbose=0, epochs=50)
late_hist = late.fit(train_in, train_out, **late_fit_kwargs)
early_fit_kwargs.update(verbose=0, epochs=50,
                    validation_data=(test_in, test_out))
early_hist = early.fit(train_in, train_out, **early_fit_kwargs)

all1_accuracy = np.sum(test_out == 1) / test_out.size
early_accuracy = binary_accuracy(early.predict(test_in), test_out)
late_accuracy = binary_accuracy(late.predict(test_in), test_out)

accuracy_format = "{1:.1%} accuracy for {0} on census income".format
print(0.1)
print(accuracy_format("baseline", all1_accuracy))
print(accuracy_format("early", early_accuracy))
print(accuracy_format("late", late_accuracy))

print('early',len(early_hist.history["loss"]),'late',len(late_hist.history["loss"]))


print('General Kenobi')