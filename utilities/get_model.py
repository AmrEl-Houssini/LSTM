from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import SGD


# building the LSTM architecture to have three LSTM layer and dence layer with 1 neuron as output layer
# with a dropout of 0.2


def get_model(x_train, first_layer_units=40, second_layer_units=40, thirds_layer_units=60,
              dropout=0.2, loss='mean_squared_error', metrics='MAPE'):
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
