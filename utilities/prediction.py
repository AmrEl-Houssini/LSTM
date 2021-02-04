import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def predict(model, x_test, y_test, training_data_len, close_df):
    """
    Testing the model and validating it's predictions

    Args:
        model -  pre-trained and compiled model
        (np array) x_test - reshaped array to test the model with
        (np array) y_test - to validate the model on
        (int) training_data_len - the number to split the data with into train and test
        close_df - a data frame of the close price after resetting the index

    Returns:
        validation_df - a df contains the predicted prices and the real data
    """
    predictions = model.predict(x_test)

    # getting the real prediction values instead of the price change in each prediction....
    # reshaping the close_df to be the same shape as the model output
    close_df = np.array(close_df).reshape(-1, 1)
    # real test data
    test_df = pd.DataFrame(close_df[training_data_len:, :])
    # real test data shifted
    test_df_shifted = close_df[training_data_len+1:, :]
    # the logic of reversing the data from difference to real
    real_data_prediction = test_df_shifted - predictions

    # Calculate/Get the value of MSE
    mse = mean_squared_error(predictions, y_test)
    print("MSE value:", mse)

    # creating a new df to assign the predictions to its equivalent days and comparing them to the real data
    validation_df = pd.DataFrame(real_data_prediction, columns=["predictions"])
    validation_df['real data'] = test_df
    validation_df.dropna(inplace=True)
    print(validation_df)

    return validation_df
