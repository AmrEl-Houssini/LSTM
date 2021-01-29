from comet_ml import Experiment

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.optimizers import SGD, Adam
from tensorflow_core.python.keras.api._v2 import keras


def get_model(WINDOW_SIZE, x_train):
    model = Sequential()
    # Input layer
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True), input_shape=(WINDOW_SIZE, x_train.shape[-1])))
    """Bidirectional RNNs allows to train on the sequence data in forward and backward direction."""
    model.add(Dropout(rate=0.2))
    # 1st Hidden layer
    model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    # 2nd Hidden layer
    model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))
    # output layer
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    """Output layer has a single neuron. 
    We use Linear activation function which activation is proportional to the input."""

    lr_dc = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.6)
    opt = keras.optimizers.Adam(learning_rate=lr_dc)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['MAE'])
    return model
