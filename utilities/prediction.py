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

    # getting the real prediction values instead of the price change in each prediction
    close_df = np.array(close_df).reshape(-1, 1)
    close_df = close_df[training_data_len + 1:, :]
    real_data_prediction = predictions + close_df

    # Calculate/Get the value of RMSE
    rmse = mean_squared_error(predictions, y_test, squared=False)
    print("RMSE value:", rmse)

    # creating a new df to assign the predictions to its equivalent dates and comparing them to the real data
    validation_df = pd.DataFrame(close_df, columns=["real data"])
    validation_df['predictions'] = real_data_prediction
    print(validation_df.head())

    return validation_df
