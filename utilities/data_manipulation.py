# importing the required libraries
import math
import pandas_datareader as pdr
from utilities.visuals import plotting
import numpy as np


def get_data(data_set='AAPL', start='2012-01-01', end='2020-9-20'):
    # getting the data
    df = pdr.DataReader(data_set, data_source='yahoo', start=start, end=end)
    print("Data set shape:", df.shape)

    # plotting the data
    plotting(df['Close'], title="Close Price History", x_label='Date', y_label='Close Price US ($)')

    # discarding data that may consider noise and creating a new df with Xt - Xt-1 values
    close_df = df['2012-01-01':].reset_index()['Close']
    plotting(close_df, title="Close price in the last four years",
             x_label="Date", y_label="Price")

    close_diff = close_df.diff().dropna()
    # print(close_diff)
    plotting(close_diff, "close price after data differencing in the last four years", "Date")

    # splitting the data and getting the number of length to train the model on
    # the training data set contains about 80% of the data.
    training_data_len = math.ceil(len(close_diff) * 0.8)
    data = np.array(close_diff).reshape(-1, 1)

    # for doing a forecast based on the past 60 days,,,
    # creating x_train having the first 60 data (from 0 to 59) in column 1 and so on
    # creating y_train having the value of the 61s (located at index 60) as first column and so on
    train_data = data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    print("Training set shape:", train_data.shape)

    # converting x_train, y_train to np arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshaping the data to 3D to be accepted by our LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # creating the test data set
    real_test_data = close_df[training_data_len : ]
    test_data = data[training_data_len - 60:, :]
    x_test = []
    y_test = data[training_data_len:, :]
    print("Testing set shape:", test_data.shape)
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # converting x_test to be a 3D np array to be accepted by the LSTM model
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return df, x_train, y_train, x_test, y_test, training_data_len, close_df, real_test_data, test_data


