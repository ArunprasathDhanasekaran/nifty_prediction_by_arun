import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import streamlit as st

# Step 1: Fetch Historical Data
@st.cache_data
def fetch_data():
    nifty_ticker = "^NSEI"  # Nifty 50 index
    gold_ticker = "GC=F"    # Gold futures
    crude_ticker = "CL=F"   # Crude oil futures
    sp500_ticker = "^GSPC"  # S&P 500 index
    usd_inr_ticker = "USDINR=X"  # USD to INR exchange rate

    # Download historical data using yfinance
    nifty_data = yf.download(nifty_ticker, start="2015-01-01")
    gold_data = yf.download(gold_ticker, start="2015-01-01")
    crude_data = yf.download(crude_ticker, start="2015-01-01")
    sp500_data = yf.download(sp500_ticker, start="2015-01-01")
    usd_inr_data = yf.download(usd_inr_ticker, start="2015-01-01")

    # Get the last available date
    last_date = nifty_data.index[-1].date()  # Get last date for Nifty data

    # Keep only the 'Close' price from each data source
    nifty_data = nifty_data[['Close']].rename(columns={'Close': 'Nifty_Close'})
    gold_data = gold_data[['Close']].rename(columns={'Close': 'Gold_Close'})
    crude_data = crude_data[['Close']].rename(columns={'Close': 'Crude_Close'})
    sp500_data = sp500_data[['Close']].rename(columns={'Close': 'SP500_Close'})
    usd_inr_data = usd_inr_data[['Close']].rename(columns={'Close': 'USD_INR'})

    # Merge datasets based on date
    data = nifty_data.join([gold_data, crude_data, sp500_data, usd_inr_data], how='inner')

    # Handle Missing Data
    data.dropna(inplace=True)
    
    return data, last_date

# Step 2: Train Models
@st.cache_resource
def train_models(data):
    # Prepare Feature and Target Variables
    X = data[['Gold_Close', 'Crude_Close', 'SP500_Close', 'USD_INR']].values
    y = data['Nifty_Close'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced number of trees
    rf_model.fit(X_train, y_train)

    # Train Neural Network Model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    nn_model = Sequential()
    nn_model.add(Dense(32, activation='relu', input_dim=X_train_scaled.shape[1]))  # Reduced neurons
    nn_model.add(Dense(32, activation='relu'))  # Reduced neurons
    nn_model.add(Dense(1))
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=0)  # Reduced epochs

    return rf_model, nn_model, scaler

# Step 3: Make Predictions
def predict_next_nifty(rf_model, nn_model, scaler, last_known_values):
    future_nifty_rf = rf_model.predict(last_known_values.reshape(1, -1))
    last_known_values_scaled = scaler.transform(last_known_values.reshape(1, -1))
    future_nifty_nn = nn_model.predict(last_known_values_scaled).flatten()
    return future_nifty_rf[0], future_nifty_nn[0]

# Main Streamlit App
st.title("Nifty Index Prediction")

# Fetch and prepare data
data, last_date = fetch_data()

# Train models
rf_model, nn_model, scaler = train_models(data)

# Get the last known values
last_known_values = data[['Gold_Close', 'Crude_Close', 'SP500_Close', 'USD_INR']].iloc[-1].values

# Create a button to predict
if st.button("Predict"):
    future_nifty_rf, future_nifty_nn = predict_next_nifty(rf_model, nn_model, scaler, last_known_values)
    
    # Get the current Nifty value
    current_nifty_value = data['Nifty_Close'].iloc[-1]

    # Calculate market direction for Random Forest
    rf_direction = "Bullish" if future_nifty_rf > current_nifty_value else "Bearish"
    rf_percent_change = abs((future_nifty_rf - current_nifty_value) / current_nifty_value) * 100

    # Calculate market direction for Neural Network
    nn_direction = "Bullish" if future_nifty_nn > current_nifty_value else "Bearish"
    nn_percent_change = abs((future_nifty_nn - current_nifty_value) / current_nifty_value) * 100

    # Display results
    st.write(f"Last available market date: {last_date}")
    st.write(f"Predicted Next Nifty Value (Random Forest): {future_nifty_rf[0]:.2f} - {rf_direction} ({rf_percent_change:.2f}% change)")
    st.write(f"Predicted Next Nifty Value (Neural Network): {future_nifty_nn:.2f} - {nn_direction} ({nn_percent_change:.2f}% change)")
