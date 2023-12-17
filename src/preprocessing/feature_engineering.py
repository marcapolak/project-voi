import pandas as pd

def feature_engineering(ride_data, weather_data):
    # Convert 'start_time' to datetime
    ride_data['start_time'] = pd.to_datetime(ride_data['start_time'])

    # Additional feature engineering
    ride_data['date'] = ride_data['start_time'].dt.date
    ride_data['date'] = pd.to_datetime(ride_data['date'])
    ride_data['day_of_week'] = ride_data['start_time'].dt.dayofweek
    ride_data['month'] = ride_data['start_time'].dt.month

    # Aggregate the ride data at the h3index_small level
    ride_counts_h3index_small = ride_data.groupby(['date', 'h3index_small']).agg(
        num_rides=('ride_id', 'count'),
        start_lon=('start_lon', 'mean'),  
        start_lat=('start_lat', 'mean')   
    ).reset_index()

    # Sorting by 'date' and 'h3index_small' for correct lag calculation
    ride_counts_h3index_small.sort_values(by=['h3index_small', 'date'], inplace=True)

    # Creating lag and rolling features within each h3index_small group
    ride_counts_h3index_small['lag_1_day_num_rides'] = ride_counts_h3index_small.groupby('h3index_small')['num_rides'].shift(1)
    ride_counts_h3index_small['lag_7_days'] = ride_counts_h3index_small.groupby('h3index_small')['num_rides'].shift(7)
    ride_counts_h3index_small['rolling_avg_7_days'] = ride_counts_h3index_small.groupby('h3index_small')['num_rides'].rolling(window=7).mean().reset_index(0, drop=True)

    ride_counts_h3index_small.fillna(0, inplace=True)

    # Ensure weather_data 'date' is in datetime format
    weather_data['date'] = pd.to_datetime(weather_data['date'])

    # Merge 'day_of_week' and 'month' from ride_data to aggregated data
    additional_features = ride_data[['date', 'day_of_week', 'month']].drop_duplicates()
    processed_data = pd.merge(ride_counts_h3index_small, additional_features, on='date', how='left')

    # Merge with weather data on 'date'
    processed_data = pd.merge(processed_data, weather_data, on='date', how='left')

    return processed_data
