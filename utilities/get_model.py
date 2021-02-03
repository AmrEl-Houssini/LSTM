from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD


def get_model(x_train, first_layer_units=40, second_layer_units=40, thirds_layer_units=60,
              dropout=0.2, loss='mean_squared_error', metrics='MAE'):
    """
    Building the LSTM architecture to have three LSTM layer and dense layer with 1 neuron as output layer

    Args:
        (np array) x_train - a 3D shaped array input to the model
        (int) first_layer_units - number of units in the first layer
        (int) second_layer_units - number of units in the second layer
        (int) thirds_layer_units - number of units in the second layer
        (int) dropout - Dropout percentage to control over-fitting during training
        (str) loss - the loss function to be used
        (str) metrics - the metric to be used
    Returns:
        model - the model after being compiled
    """
    model = Sequential()
    model.add(LSTM(units=first_layer_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=second_layer_units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=thirds_layer_units))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer=SGD(lr=0.001), loss=loss, metrics=[metrics])
    return model
