import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def moving_average_forecast(hr_data, window_size=3):
    try:
        if len(hr_data) < window_size:
            raise ValueError("Not enough data to apply moving average")
        # Calculate moving average for the last window_size points
        moving_avg = np.mean(hr_data[-window_size:])
        return moving_avg
    except Exception as e:
        print(f"Moving Average could not be used due to: {e}")
        return None

def plot_moving_avg_forecast(hr_times, hr_data, forecast, time_steps):
    try:
        if not hr_times or not hr_data or forecast is None:
            raise ValueError("Insufficient data for plotting.")

        # Generate future time points based on the number of forecasts
        future_times = [hr_times[-1] + timedelta(minutes=30 * (i + 1)) for i in range(len(forecast))]

        # Ensure forecast has the same length as future_times
        if len(future_times) != len(forecast):
            forecast = [forecast[0]] * len(future_times)
        
        print("Future Times:", future_times)
        print("Forecast Data:", forecast)

        plt.figure(figsize=(12, 6))
        plt.plot(hr_times, hr_data, label='Historical Heart Rate Data', marker='o')

        # Plot historical data and forecasted data
        plt.plot(future_times, forecast, 'gx--', label='Moving Average Forecasted Heart Rate', markersize=10)
        plt.title('Moving Average Heart Rate Forecast')
        plt.xlabel('Time')
        plt.ylabel('Heart Rate')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot to a file
        file_path = 'moving_avg_plot.png'
        plt.savefig(file_path)
        
        # Check if the file was saved
        import os
        if os.path.isfile(file_path):
            print(f"Moving Average plot saved as '{file_path}'")
        else:
            print(f"Failed to save plot as '{file_path}'")
        
        plt.close()
    except Exception as e:
        print(f"Plotting could not be done due to: {e}")
