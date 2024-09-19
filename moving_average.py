import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def round_to_half_hour(dt):
    """Round datetime to the nearest half-hour."""
    if dt.minute < 15:
        return dt.replace(minute=0, second=0, microsecond=0)
    elif dt.minute < 45:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        return dt.replace(hour=(dt.hour + 1) % 24, minute=0, second=0, microsecond=0)

def moving_average_forecast(hr_data, window_size=3):
    try:
        if len(hr_data) < window_size:
            raise ValueError("Not enough data to apply moving average")
        moving_avg = np.mean(hr_data[-window_size:])
        return moving_avg
    except Exception as e:
        print(f"Moving Average could not be used due to: {e}")
        return None

def plot_moving_avg_forecast(hr_times, hr_data, forecast, num_forecast_points):
    try:
        if not hr_times or not hr_data or forecast is None:
            raise ValueError("Insufficient data for plotting.")
        
        # Generate future time points based on the number of forecasts
        last_time = hr_times[-1]
        future_times = [last_time + timedelta(minutes=30 * (i + 1)) for i in range(num_forecast_points)]
        
        # Convert datetime objects to time strings for plotting
        hr_times_str = [t.strftime('%H:%M') for t in hr_times]
        future_times_str = [t.strftime('%H:%M') for t in future_times]

        # Ensure forecast has the same length as future_times
        if len(future_times) != len(forecast):
            forecast = [forecast[0]] * len(future_times)  # Fill with the last moving average value

        plt.figure(figsize=(12, 6))
        plt.plot(hr_times_str, hr_data, label='Historical Heart Rate Data', marker='o')
        
        # Plot forecasted values
        plt.plot(future_times_str, forecast, 'gx--', label='Moving Average Forecasted Heart Rate', markersize=10)
        
        plt.title('Moving Average Heart Rate Forecast')
        plt.xlabel('Time')
        plt.ylabel('Heart Rate')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot to a file
        plt.savefig('moving_avg_plot.png')
        plt.close()
    except Exception as e:
        print(f"Plotting could not be done due to: {e}")
