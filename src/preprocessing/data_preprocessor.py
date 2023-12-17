import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

def make_preprocessor(numerical_features, categorical_features):
    # Exclude 'date' and 'num_rides' from numerical features
    numerical_features = [feature for feature in numerical_features if feature not in ['date', 'num_rides']]

    # Create the preprocessor    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', make_pipeline(SimpleImputer(strategy="median")), numerical_features),
            ('categorical', make_pipeline(SimpleImputer(strategy="most_frequent")), categorical_features)
        ]
    )
    return preprocessor

def preprocess_data(merged_data_with_weather, preprocessor, numerical_features, categorical_features):
    # Extract the 'date' and 'num_rides' columns
    dates = merged_data_with_weather[['date']]
    num_rides = merged_data_with_weather[['num_rides']]

    # Drop non-feature columns
    features_data = merged_data_with_weather.drop(['date', 'num_rides'], axis=1)

    # Fit and transform the data using the preprocessor
    transformed_data = preprocessor.fit_transform(features_data)

    # Combine numerical and categorical features
    all_features = numerical_features + categorical_features

    # Create a DataFrame with the transformed data and specified columns
    transformed_df = pd.DataFrame(transformed_data, columns=all_features)

    # Add back the 'date' and 'num_rides' columns
    transformed_df['date'] = dates.reset_index(drop=True)
    transformed_df['num_rides'] = num_rides.reset_index(drop=True)

    print("Columns after preprocessing:", transformed_df.columns)

    return transformed_df










