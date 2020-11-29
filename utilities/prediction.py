import numpy as np
from utilities.visuals import plotting


def predict(model, x_test, y_test, training_data_len, close_last_two_years_df ,df):
    predictions = model.predict(x_test)

    # getting the real prediction values instead of the price change in each prediction
    close_last_two_years = np.array(close_last_two_years_df).reshape(-1, 1)
    close_price_test_data = close_last_two_years[training_data_len+1:, :]

    real_data_prediction = predictions + close_price_test_data

    # Calculate/Get the value of RMSE
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print("RMSE value:", rmse)

    # creating a new df to assign the predictions to its equivalent dates
    recent_data = df['2018-01-08':]

    validation_df = recent_data[training_data_len+1:]
    validation_df['predictions'] = real_data_prediction

    return real_data_prediction, validation_df, recent_data

