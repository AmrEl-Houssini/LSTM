from utilities.visuals import plotting
import math
import numpy as np
import pandas_datareader as pdr


def data(data_set='AAPL', start='2012-01-01', end='2020-9-20'):
    """
       Getting the desired data from yahoo, then doing some data manipulation such as plotting and data differencing

       Args:
           (str) data_set - the ticker of desired dataset (company)
           (str) start - the start date of the desired dataset
           (str) end - the end date of the desired dataset

       Returns:
           (np array) close_diff - close price data after being differenced

       """

    apple_data = pdr.DataReader(data_set, data_source='yahoo', start=start, end=end)
    #print("Data set shape:", apple_data.shape)

    # visualizing the data
    #plotting(apple_data['Close'], title="Close Price History", x_label='Date', y_label='Close Price US ($)')

    # creating a new df with Xt - Xt-1 values of the close prices
    close_data = apple_data['2012-01-01':]['Close']
    # differencing the data
    close_diff = close_data.diff().dropna()
    # print(close_diff)
    #plotting(close_diff, "close price after data differencing", "Date")
    #print(close_diff.shape)
    close_diff = np.array(close_diff).reshape(-1, 1)
    return close_diff


def to_sequences(data, seq_len):
    """
    Building some sequences. Sequences work like walk forward validation approach,
    where initial sequence length will be defined and subsequently will be shifting one position
    to the right to create another sequence. This way the process is repeated until all possible positions are used.

    Args:
        data - data to be divided into sequences
        (int) seq_len - the desired sequence length (in days)

    Returns:
        (np array) d - sequences of data in np arrays
    """

    d = []
    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])
    #print(np.array(d))
    return np.array(d)


def preprocess(data_raw, seq_len, train_split):
    """
    LSTMs require 3-D data shape; therefore,
    we need to split the data into the shape of : [batch_size, sequence_length, n_features]

    Args:
         data_raw - data to be divided into sequences
        (int) seq_len - the desired sequence length
        (int) train_split - percentage of training data to testing data

    Returns:
        x_train, y_train - 3-D arrays to train the model with
        x_test, y_test - 3-D arrays to test the model on
    """
    data = to_sequences(data_raw, seq_len)
    num_train = int(train_split * data.shape[0])
    x_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]
    x_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]
    return x_train, y_train, x_test, y_test



