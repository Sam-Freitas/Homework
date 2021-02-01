import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from typing import List

import h5py
import numpy as np
import pytest
import tensorflow
import tensorflow as tf
import logging

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

def create_network(n_hidden_layers,n_units,n_inputs,n_outputs,activations):

    thisModel = tf.keras.models.Sequential()
    thisModel.add(tensorflow.keras.layers.Dense(n_inputs, input_shape=(n_inputs,),activation=activations))
    for i in range(n_hidden_layers):
        thisModel.add(tf.keras.layers.Dense(n_units,activation=activations))

    thisModel.add(tf.keras.layers.Dense(n_outputs,activation=activations))

    thisModel.compile(optimizer='rmsprop',loss='MAE')

    thisModel.summary()

    return thisModel


train_in, train_out, test_in, test_out = load_hdf5("data/auto-mpg.hdf5")

n_inputs = train_in.shape[-1]
n_outputs = train_out.shape[-1]

deep = create_network(12,3,n_inputs,n_outputs,'linear')
wide = create_network(9,4,n_inputs,n_outputs,'linear')

deep.fit(train_in, train_out, verbose=0, epochs=100)
wide.fit(train_in, train_out, verbose=0, epochs=100)

mean_predict = np.full(shape=test_out.shape, fill_value=np.mean(train_out))
[baseline_rmse] = root_mean_squared_error(mean_predict, test_out)
[deep_rmse] = root_mean_squared_error(deep.predict(test_in), test_out)
[wide_rmse] = root_mean_squared_error(wide.predict(test_in), test_out)

rmse_format = "{1:.1f} RMSE for {0} on Auto MPG".format
print()
print(rmse_format("baseline", baseline_rmse))
print(rmse_format("deep", deep_rmse))
print(rmse_format("wide", wide_rmse))

print('Goodbye there')