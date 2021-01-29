import yfinance as yf
from utilities.visuals import plotting
import math
import numpy as np
import pandas_datareader as pdr


def data(data_set='AAPL', start='2012-01-01', end='2020-9-20'):
    apple_data = pdr.DataReader(data_set, data_source='yahoo', start=start, end=end)

    #print("Data set shape:", apple_data.shape)

    # visualizing the data
    #plotting(apple_data['Close'], title="Close Price History", x_label='Date', y_label='Close Price US ($)')

    # discarding data that may consider noise and creating a new df with Xt - Xt-1 values
    close_data = apple_data['2012-01-01':]['Close']
    #plotting(close_data, title="Close price in the last four years",
             #x_label="Date", y_label="Price")
    #print(close_data.shape)

    # differencing the data
    close_diff = close_data.diff().dropna()
    # print(close_diff)
    #plotting(close_diff, "close price after data differencing in the last four years", "Date")
    #print(close_diff.shape)
    close_diff = np.array(close_diff).reshape(-1, 1)
    return close_diff


def to_sequences(data, seq_len):
    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    return np.array(d)


def preprocess(data_raw, seq_len, train_split):
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    x_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]
    x_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]
    return x_train, y_train, x_test, y_test



