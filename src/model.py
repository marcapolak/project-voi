import os
import pandas as pd
from preprocessing.data_loader import load_data
from preprocessing.data_preprocessor import preprocess_data, make_preprocessor
from preprocessing.feature_engineering import feature_engineering
from train_and_evaluate import train_and_evaluate

# Set the input and output directories
input_folder = '/app/input'
output_folder = '/app/output/results'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load data
ride_data_path = os.path.join(input_folder, 'voiholm.csv')  
weather_data_path = os.path.join(input_folder, 'weather_data.csv')  
ride_data, weather_data = load_data(ride_data_path, weather_data_path)

# Feature engineering 
processed_data = feature_engineering(ride_data, weather_data)

numerical_features = [
    'start_lon', 'start_lat', 'temperature', 'max_temperature', 
    'min_temperature', 'precipitation', 'lag_1_day_num_rides', 
    'lag_7_days', 'rolling_avg_7_days', 'day_of_week'
]
categorical_features = ['month']

# Create preprocessor
preprocessor = make_preprocessor(numerical_features, categorical_features)

# Preprocess data
preprocessed_data = preprocess_data(processed_data, preprocessor, numerical_features, categorical_features)

# Splitting the data
preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'])

max_date = preprocessed_data['date'].max()
cutoff_date = max_date - pd.DateOffset(weeks=1)
train_data = preprocessed_data[preprocessed_data['date'] <= cutoff_date]
test_data = preprocessed_data[preprocessed_data['date'] > cutoff_date]

features = ['start_lon', 'start_lat', 'temperature', 'max_temperature', 'min_temperature', 'precipitation',
            'lag_1_day_num_rides', 'lag_7_days', 'rolling_avg_7_days', 'day_of_week', 'month']
target = 'num_rides'

X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Train and evaluate the model
train_and_evaluate(X_train, y_train, X_test, y_test, output_folder)
