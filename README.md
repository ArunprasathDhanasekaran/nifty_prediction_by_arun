# Nifty Index Prediction

This project is a predictive model for the Nifty 50 index using historical data from related assets like gold, crude oil, S&P 500, and USD/INR exchange rate. The model employs both Random Forest and Neural Network algorithms to predict the future Nifty value and classify the market as bullish or bearish.

## Features

- Fetches historical data from Yahoo Finance
- Trains Random Forest and Neural Network models
- Predicts future Nifty value based on the latest market data
- Provides a Streamlit UI for interaction

## Requirements

To run this project, you need the following Python packages:

- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- streamlit

You can install these packages using the following command:

```bash
pip install -r requirements.txt
