from typing import Tuple, Dict
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow
import tensorflow as tf 


# A somewhat complete general model creation tool 
def create_network(n_hidden_layers,n_units,n_inputs,n_outputs,h_activation,o_activation,thisLoss,thisOptimizer,dropout_bool,dropout_rate):

    # Create a sequential model
    thisModel = tf.keras.models.Sequential()
    # create the input layer
    thisModel.add(tensorflow.keras.layers.Dense(n_inputs, input_shape=(n_inputs,),activation=h_activation))
    # create each hidden layer
    for _ in range(n_hidden_layers):
        thisModel.add(tf.keras.layers.Dense(n_units,activation=h_activation))
        # if dropout is necessary then add it between the layers
        if dropout_bool:
            thisModel.add(tf.keras.layers.Dropout(dropout_rate))
    # create the output layer
    thisModel.add(tf.keras.layers.Dense(n_outputs,activation=o_activation))
    # compile the model 
    thisModel.compile(optimizer=thisOptimizer,loss=thisLoss)

    # thisModel.summary()

    return thisModel

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """

    # create the deep and wide networks with specified paramaters 
    deep = create_network(15,3,n_inputs,n_outputs,'linear','linear','MAE','rmsprop',0,0.2)
    wide = create_network(9,4,n_inputs,n_outputs,'linear','linear','MAE','rmsprop',0,0.2)

    return [deep,wide]

def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """
    
    # create the relu and tanh networks with specified paramaters 
    relu = create_network(1,7,n_inputs,n_outputs,'relu','sigmoid','hinge','adam',0,0.2)
    tanh = create_network(1,7,n_inputs,n_outputs,'tanh','sigmoid','hinge','adam',0,0.2)

    return [relu,tanh]


def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                tensorflow.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """

    # create the dropout and no-dropout networks with specified paramaters 
    drop    = create_network(1,11,n_inputs,n_outputs,'tanh','softmax','categorical_crossentropy','SGD',1,0.2)
    no_drop = create_network(1,11,n_inputs,n_outputs,'tanh','softmax','categorical_crossentropy','SGD',0,0.2)

    return [drop,no_drop]


def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tensorflow.keras.models.Model,
                                                Dict,
                                                tensorflow.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """

    # create a callback function that enables early stopping
    thisCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',patience=7)

    # allow the callback as Dicts with specific parameters
    early_fit_kwargs = {"callbacks":thisCallback}
    late_fit_kwargs = {}

    # create a custom SGD that prioritizes an early stopping method
    mySgd = tf.keras.optimizers.SGD(
        learning_rate=0.02, momentum=0.9, nesterov=False, name='SGD')

    # create the early and late networks with specified paramaters 
    early= create_network(1,16,n_inputs,n_outputs,'tanh','sigmoid','binary_crossentropy',mySgd,1,0.1)
    late = create_network(1,16,n_inputs,n_outputs,'tanh','sigmoid','binary_crossentropy',mySgd,1,0.1)

    return [early,early_fit_kwargs,late,late_fit_kwargs]
