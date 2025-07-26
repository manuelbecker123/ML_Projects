import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

def plot_forecast(test_data, forecast_data, model_name):
    """Plots the actual vs. forecasted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data.values, label='Actual')
    plt.plot(test_data.index, forecast_data, label='Forecast', alpha=0.7)
    plt.title(f'{model_name} Forecast vs. Actuals')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(True)
    
    save_path = f'reports/figures/{model_name.lower()}_forecast_plot.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Forecast plot saved to {save_path}")
    plt.show()

def calculate_metrics(y_true, y_pred):
    """Calculates, prints, and returns evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Return the metrics as a dictionary
    return {'mae': mae, 'mse': mse, 'rmse': rmse}