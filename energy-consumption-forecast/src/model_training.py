import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

def train_prophet_model(df, test_size=365*24):
    """Trains a Prophet model and returns the forecast."""
    print("Training Prophet model...")
    train_df = df.iloc[:-test_size].copy()
    train_df.reset_index(inplace=True)
    train_df.rename(columns={'Datetime': 'ds', 'AEP_MW': 'y'}, inplace=True)

    model = Prophet()
    model.fit(train_df[['ds', 'y']])
    
    future = model.make_future_dataframe(periods=test_size, freq='H')
    forecast = model.predict(future)
    
    print("Prophet model training complete.")
    return model, forecast

def create_lstm_dataset(dataset, look_back=24):
    """Creates sequences for LSTM model."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# --- CORRECTED FUNCTION SIGNATURE AND LOGIC ---
def train_lstm_model(df, test_size, look_back=24, epochs=10):
    """Trains an LSTM model and returns the forecast."""
    print("Training LSTM model...")
    data = df['AEP_MW'].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # **FIX:** Use the test_size from main.py, not a hardcoded 80/20 split
    train_size = len(scaled_data) - test_size
    train_data = scaled_data[0:train_size, :]
    
    X_train, y_train = create_lstm_dataset(train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=1)
    
    # --- CORRECTED FORECASTING LOGIC ---
    # **FIX:** Create the test input based on the correct test_size
    inputs = scaled_data[len(scaled_data) - test_size - look_back:]
    
    X_test = []
    # This loop will now create exactly `test_size` samples
    for i in range(look_back, len(inputs)):
        X_test.append(inputs[i-look_back:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    print("LSTM model training complete.")
    return model, predictions