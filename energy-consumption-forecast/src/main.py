import argparse
import pandas as pd
from data_ingestion import download_and_extract_data
from feature_engineering import process_data
from model_training import train_prophet_model, train_lstm_model
from evaluate import plot_forecast, calculate_metrics
import os
from datetime import datetime

def main(model_type):
    # --- Step 1: Data Ingestion & Processing ---
    raw_data_path = 'data/raw/AEP_hourly.csv'
    if not os.path.exists(raw_data_path):
        download_and_extract_data()
    
    processed_data_path = 'data/processed/AEP_processed.csv'
    if not os.path.exists(processed_data_path):
        process_data(raw_path=raw_data_path, processed_path=processed_data_path)
        
    df = pd.read_csv(processed_data_path, index_col='Datetime', parse_dates=True)
    df.sort_index(inplace=True)
    
    test_size = 365 * 24 
    test_df = df.iloc[-test_size:]

    # --- Step 2: Model Training & Evaluation ---
    metrics = None
    if model_type == 'prophet':
        prophet_model, forecast = train_prophet_model(df, test_size)
        forecast_values = forecast['yhat'].iloc[-test_size:].values
        
        print("\n--- Prophet Model Evaluation ---")
        metrics = calculate_metrics(test_df['AEP_MW'], forecast_values)
        plot_forecast(test_df['AEP_MW'], forecast_values, 'Prophet')

    elif model_type == 'lstm':
        lstm_model, predictions = train_lstm_model(df, test_size=test_size, epochs=10)
        
        print("\n--- LSTM Model Evaluation ---")
        metrics = calculate_metrics(test_df['AEP_MW'], predictions)
        plot_forecast(test_df['AEP_MW'], predictions, 'LSTM')
        
    else:
        print("Invalid model type. Please choose 'prophet' or 'lstm'.")
        return

    # --- Step 3: Save Results to Text File ---
    if metrics:
        results_path = 'reports/evaluation_results.txt'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'a') as f:
            f.write(f"--- {model_type.upper()} Model Results ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}\n")
            f.write(f"Mean Squared Error (MSE): {metrics['mse']:.2f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}\n")
            f.write("-" * 50 + "\n\n")
        print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument('model', type=str, choices=['prophet', 'lstm'], help="The model to train and evaluate ('prophet' or 'lstm').")
    args = parser.parse_args()
    main(args.model)