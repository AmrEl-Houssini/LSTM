from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from tensorflow_core.python.keras.api._v2 import keras


def get_model(window_size, x_train):
    """
    Building a BiLSTM model with 3 layers and a drop out of 0.2 to control over-fitting during training

    Args:
        (int) window_size - number of units in the model layer
        (np array) x_train - a 3D shaped array input to the model

    Returns:
        model - the model after being compiled
    """

    model = Sequential()
    # Input layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=True), input_shape=(window_size, x_train.shape[-1])))
    model.add(Dropout(rate=0.2))
    # 1st Hidden layer
    model.add(Bidirectional(LSTM((window_size * 2), return_sequences=True)))
    model.add(Dropout(rate=0.2))
    # 2nd Hidden layer
    model.add(Bidirectional(LSTM(window_size, return_sequences=False)))
    # output layer
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    """Output layer has a single neuron. 
    We use Linear activation function which activation is proportional to the input."""

    lr_dc = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.6)
    opt = keras.optimizers.Adam(learning_rate=lr_dc)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['MAE'])
    return model
