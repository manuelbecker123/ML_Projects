import pandas as pd
import os

def process_data(raw_path='data/raw/AEP_hourly.csv', processed_path='data/processed/AEP_processed.csv'):
    """
    Loads raw data, performs feature engineering, and saves the processed data.
    """
    print("Loading raw data...")
    df = pd.read_csv(raw_path)
    
    print("Processing data and creating features...")
    # Convert to datetime and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    
    # Create time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Create lag and rolling window features
    df['lag_24h'] = df['AEP_MW'].shift(24)
    df['rolling_mean_24h'] = df['AEP_MW'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['AEP_MW'].rolling(window=24).std()
    
    # Drop rows with NaN values created by lag/rolling features
    df.dropna(inplace=True)
    
    # Save processed data
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path)
    print(f"Processed data saved to {processed_path}")
    return df

if __name__ == '__main__':
    process_data()