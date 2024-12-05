"""
Script for processing and preparing groundwater prediction data.

This script is designed to process large CSV files containing meteorological
and hydrogeological data. It handles missing values, performs feature
engineering, and prepares the data for model training. The data processing
is done in chunks to efficiently handle large datasets.

Functions:
    1. filter_and_impute_missing_values: Filters columns with excessive missing
       values and imputes missing values for numeric columns with the median.
    2. process_data_in_chunks: Processes the dataset in chunks, drops
       unnecessary columns, handles time series data, and separates features
       and target (only for training data).
    3. create_features: Generates new features such as time-based features,
       lag features, interaction features, and more for predictive modeling.
    4. main: The main function that orchestrates the entire data loading,
       processing, feature engineering, and preparation for modeling.

Usage:
    To run the script, simply execute it from the command line:

    python preprocess_data.py --path <data_file_path> [--is_train]

    --path: Path to the CSV data file (training or test).
    --is_train: Flag to indicate training data (with target column).

    It processes both training and test data, applies filtering and imputation
    for missing values, and creates features.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


def filter_and_impute_missing_values(
    X: pd.DataFrame,
    missing_threshold: float = 0.8
    ) -> pd.DataFrame:
    """
    Filters out columns with more missing values than the specified threshold
    and imputes missing values with the median for numeric columns.

    Parameters:
    - X (pd.DataFrame): The feature DataFrame.
    - missing_threshold (float): The threshold for the maximum allowed missing
                                 values in a column (default is 0.8).

    Returns:
    - X (pd.DataFrame): The filtered and imputed DataFrame.
    """
    # Filter columns with missing values above the threshold
    cols_to_drop = X.columns[X.isna().mean() > missing_threshold]
    X.drop(columns=cols_to_drop, inplace=True)

    # Impute missing values with the median for numeric columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    return X


def process_data_in_chunks(
    path: str,
    n: int,
    columns_to_exclude: list = None,
    is_train: bool = True
    ) -> (pd.DataFrame, pd.Series):
    """
    Processes a CSV file in chunks, selects every n-th row,
    converts `meteo_date` to datetime, removes categorical variables,
    and separates features (X) from target (y) if the dataset is for training.

    Parameters:
    - path (str): The path to the CSV file.
    - n (int): The interval for selecting every n-th row.
    - columns_to_exclude (list): Columns to exclude from the dataset.
    - is_train (bool): Flag to indicate if the data is training.

    Returns:
    - X (pd.DataFrame): The features DataFrame after processing.
    - y (pd.Series or None): The target `piezo_groundwater_level_category` if
                             training data, else None for test data.
    """
    # Read the data in chunks
    data = pd.read_csv(path, index_col=0, chunksize=200_000, low_memory=False)

    chunks_X = []
    chunks_y = []

    for chunk in tqdm(data):
        # Drop columns to exclude
        if columns_to_exclude:
            chunk.drop(columns=columns_to_exclude, inplace=True)

        # Select every n-th row
        if is_train:
            chunk = chunk.iloc[::n, :]

        # Separate features (X) and target (y) for training
        X = chunk.drop(
            columns=['piezo_groundwater_level_category'],
            errors='ignore'
            )
        y = chunk['piezo_groundwater_level_category'] if is_train else None

        # Convert 'meteo_date' to datetime format
        if 'meteo_date' in X.columns:
            X['meteo_date'] = pd.to_datetime(X['meteo_date'], errors='coerce')

        # Remove categorical columns (excluding 'meteo_date')
        categorical_cols = X.select_dtypes(include=['object']).columns
        categorical_cols = categorical_cols[categorical_cols != 'meteo_date']
        X.drop(columns=categorical_cols, inplace=True)

        # Append processed chunks of X and y to the lists
        chunks_X.append(X)
        if is_train:
            chunks_y.append(y)

    # Concatenate all chunks back into a single DataFrame
    X = pd.concat(chunks_X, ignore_index=True)
    y = pd.concat(chunks_y, ignore_index=True) if is_train else None

    return X, y


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from the existing DataFrame.
    Features include: date-based features, lag features, rolling averages,
    interaction features, temperature range, evapotranspiration to rainfall
    ratio, and altitude difference.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features to transform.

    Returns:
    - df (pd.DataFrame): The DataFrame with new features created.
    """

    # Extract date-based features
    df['day'] = df['meteo_date'].dt.day
    df['month'] = df['meteo_date'].dt.month
    df['quarter'] = df['meteo_date'].dt.quarter
    df['year'] = df['meteo_date'].dt.year

    # Drop the 'meteo_date' column
    df.drop(columns=['meteo_date'], inplace=True)

    # sin/cos transformation of date-based features
    df['day_sin'] = df['day'].apply(lambda x: np.sin(2 * np.pi * x / 31))
    df['day_cos'] = df['day'].apply(lambda x: np.cos(2 * np.pi * x / 31))
    df['month_sin'] = df['month'].apply(lambda x: np.sin(2 * np.pi * x / 12))
    df['month_cos'] = df['month'].apply(lambda x: np.cos(2 * np.pi * x / 12))
    df['quarter_sin'] = df['quarter'].apply(lambda x: np.sin(2 * np.pi * x / 4))
    df['quarter_cos'] = df['quarter'].apply(lambda x: np.cos(2 * np.pi * x / 4))

    # Lag features (lag of 1 year)
    df['meteo_temperature_avg_lag_1'] = df['meteo_temperature_avg'].shift(
        250 * 365
    )
    df['meteo_rain_height_lag_1'] = df['meteo_rain_height'].shift(250 * 365)

    # Rolling average (7-day window)
    rolling_window = 7 * 250
    df['meteo_temperature_avg_rolling_mean_7'] = df[ 
        'meteo_temperature_avg'
    ].rolling(window=rolling_window).mean()

    df['meteo_rain_height_rolling_sum_7'] = df[
        'meteo_rain_height'
    ].rolling(window=rolling_window).sum()

    # Interaction features
    df['temperature_wind_interaction'] = (
        df['meteo_temperature_avg'] * df['meteo_wind_speed_avg_10m']
    )
    df['humidity_rain_interaction'] = (
        df['meteo_humidity_avg'] * df['meteo_rain_height']
    )

    # Maximum/Minimum Temperature Range
    df['temperature_range'] = (
        df['meteo_temperature_max'] - df['meteo_temperature_min']
    )

    # Evapotranspiration to Rainfall Ratio
    df['evapotranspiration_to_rain_ratio'] = (
        df['meteo_evapotranspiration_grid'] / (df['meteo_rain_height'] + 1)
    )

    # Altitude Difference (Piezo vs Meteo)
    df['altitude_difference'] = (
        df['piezo_station_altitude'] - df['meteo_altitude']
    )

    # Cumulative Rainfall (30 days)
    df['cumulative_rainfall_30_days'] = df[
        'meteo_rain_height'
    ].rolling(window=30 * 250).sum()

    # Fill missing values with the median
    df.fillna(df.median(), inplace=True)

    return df

def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the features in the DataFrame using the provided StandardScaler.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features to scale.
    
    Returns:
    - df (pd.DataFrame): The DataFrame with scaled features.
    """
    # Select only numeric columns for scaling
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

def encode_lables(y_train: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in the DataFrame using one-hot encoding.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features to encode.
    
    Returns:
    - df (pd.DataFrame): The DataFrame with encoded categorical features.
    """
    # Extract the target column
    y_train = y_train['piezo_groundwater_level_category']

    # Map the target labels to numerical values
    mapping = {
        'Very Low': 0,
        'Low': 1,
        'Average': 2,
        'High': 3,
        'Very High': 4
    }
    y_train = y_train.map(mapping)

    return y_train


def main():
    """
    Main function for data loading, processing, and feature engineering.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description='Process groundwater prediction data.'
        )

    parser.add_argument('--path',
                        type=str,
                        help='Path to the CSV data file (train or test).')
    parser.add_argument('--is_train',
                        action='store_true',
                        help='Flag to indicate training data (with target).')

    args = parser.parse_args()

    # Columns to exclude during processing
    columns_to_exclude = [
        "piezo_station_update_date", "piezo_station_department_code",
        "piezo_station_department_name", "piezo_continuity_code",
        "piezo_station_bdlisa_codes", "piezo_station_bss_code",
        "piezo_station_commune_name", "piezo_station_bss_id",
        "piezo_station_pe_label", "piezo_bss_code", "piezo_producer_code",
        "piezo_producer_name", "meteo_name", "meteo_id", "hydro_station_code",
        "prelev_structure_code_0", "prelev_structure_code_1", 
        "prelev_structure_code_2", "piezo_measure_nature_code", 
        "meteo_DRR", "piezo_measure_nature_name", "hydro_method_code"
    ]

    # Process the data in chunks
    X, y = process_data_in_chunks(
        path=args.path,
        n=10,
        columns_to_exclude=columns_to_exclude,
        is_train=args.is_train
    )

    # Apply missing value filtering and imputation
    X = filter_and_impute_missing_values(X, missing_threshold=0.8)

    # Apply feature engineering
    X = create_features(X)

    # Scale features
    X = scale_features(X)

    # Encode labels
    if args.is_train:
        y = encode_lables(y)
    

    # Save the processed data to CSV files
    if args.is_train:
        y.to_csv("data/y_train.csv", index=False)
        X.to_csv("data/X_train.csv", index=False)
    else:
        X.to_csv("data/X_test.csv", index=False)

    print(f"Data saved for {'training' if args.is_train else 'testing'}.")


if __name__ == "__main__":
    main()
