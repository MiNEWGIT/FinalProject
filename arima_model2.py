import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import random
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['HRMonitoring']
users_collection = db['Users']

def get_user_data(user_name):
    """Retrieve HR data for the specified user."""
    return users_collection.find_one({"Name": user_name})

def process_hr_data(hr_data):
    """Process raw HR data into a time series starting from the current time rounded up to the nearest half-hour in 24-hour format."""
    hr_series = pd.Series(list(hr_data.values()))
    hr_series = pd.to_numeric(hr_series, errors='coerce')
    hr_series.dropna(inplace=True)
    
    # Get the current time
    now = pd.Timestamp.now()
    
    # Determine the next half-hour boundary
    if now.minute % 30 == 0 and now.second == 0:
        rounded_time = now
    else:
        # Compute the next half-hour boundary
        minutes = (now.minute // 30 + 1) * 30
        if minutes == 60:
            minutes = 0
            rounded_time = now.replace(hour=now.hour + 1, minute=minutes, second=0, microsecond=0)
        else:
            rounded_time = now.replace(minute=minutes, second=0, microsecond=0)
    
    # Set the start time to the next half-hour
    start_time = rounded_time
    
    # Generate a date range with a frequency of 30 minutes
    hr_series.index = pd.date_range(start=start_time, periods=len(hr_series), freq='30T')
    
    return hr_series



def fit_arima_model(hr_series):
    """Fit an ARIMA model to the HR time series."""
    model = ARIMA(hr_series, order=(1, 1, 1))  # (p, d, q) parameters can be adjusted
    return model.fit()

def generate_forecast(model_fit, steps):
    """Generate forecasted HR values using the fitted ARIMA model."""
    forecast = model_fit.forecast(steps=steps)
    return forecast

def create_forecast_series(forecast, hr_series, forecast_steps):
    """Create a pandas Series with forecasted HR values."""
    forecast_index = pd.date_range(hr_series.index[-1] + pd.Timedelta(minutes=30), periods=forecast_steps, freq='30T')
    return pd.Series(forecast, index=forecast_index)

def plot_hr_and_forecast(hr_series, forecast_series, save_path="ARIMA_Backgroung.png"):
    """Plot the original and forecasted HR data and save the plot as an image file."""
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)  # Original data plot
    plt.plot(hr_series, label='Original HR Data', color='blue')
    plt.title("Original Heart Rate Data")
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()

    plt.subplot(1, 2, 2)  # Forecasted data plot
    plt.plot(forecast_series, label='Forecasted HR', color='red')
    plt.title("Forecasted Heart Rate Data")
    plt.xlabel('Time')
    plt.ylabel('Heart Rate (BPM)')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(save_path)
    plt.close()

def update_user_with_forecast(user_name, forecast_series):
    """Update the MongoDB database with forecasted HR values."""
    for time, forecast_value in forecast_series.items():
        hr_key = f"HR at {time.strftime('%H:%M (%A)')}"
        print(f"Updating database: {hr_key} = {forecast_value:.2f}")
        users_collection.update_one(
            {"Name": user_name},
            {"$set": {hr_key: int(forecast_value)}}
        )
    print("Database updated with forecasted HR values.")

def arima_forecast_for_user(user_name, forecast_steps):
    """Main function to forecast HR data and update the database."""
    user_data = get_user_data(user_name)
    if user_data:
        hr_data = {key: value for key, value in user_data.items() if 'HR at' in key}
        if hr_data:
            hr_series = process_hr_data(hr_data)
            model_fit = fit_arima_model(hr_series)
            forecast = generate_forecast(model_fit, steps=forecast_steps)
            forecast_series = create_forecast_series(forecast, hr_series, forecast_steps)
            plot_hr_and_forecast(hr_series, forecast_series)
            update_user_with_forecast(user_name, forecast_series)
        else:
            print("No heart rate data found for user.")
    else:
        print(f"No user found with the name '{user_name}'.")
        
