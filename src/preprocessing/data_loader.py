import pandas as pd

def load_data(ride_data_path, weather_data_path):
    ride_data = pd.read_csv(ride_data_path)
    weather_data = pd.read_csv(weather_data_path)

    return ride_data, weather_data

