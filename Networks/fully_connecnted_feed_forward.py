import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_fully_connected_feed_forward(input_shape, num_classes, loss='mse'):
    """
    Create a simple fully connected feed-forward neural network model.
    This function builds a sequential model with two hidden layers and an output layer.

    Parameters:
    - input_shape: tuple, shape of the input data (number of features,)
    - num_classes: int, number of output classes
    - loss: str, loss function to use for compiling the model

    Returns:
    - model: keras Model instance
    """
    model = Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Hidden layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    if num_classes == 1:
        # Output layer for regression
        model.add(layers.Dense(1, activation='linear'))
    else:
        # Output layer for classification
        model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=loss)

    return model


def create_fully_connected_feed_forward_with_dropout(input_shape, num_classes, loss='mse', dropout_rate=0.5):
    """
    Create a fully connected feed-forward neural network model with dropout layers.

    Parameters:
    - input_shape: tuple, shape of the input data (number of features,)
    - num_classes: int, number of output classes
    - loss: str, loss function to use for compiling the model
    - dropout_rate: float, dropout rate for regularization

    Returns:
    - model: keras Model instance
    """
    model = Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Hidden layers with dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(dropout_rate))

    if num_classes == 1:
        # Output layer for regression
        model.add(layers.Dense(1, activation='linear'))
    else:
        # Output layer for classification
        model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss=loss)

    return model
