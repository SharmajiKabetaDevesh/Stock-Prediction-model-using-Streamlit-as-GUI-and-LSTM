import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
import yfinance as yf

from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

def get_stock_ticker(stock_name):
  """Gets the stock ticker for a given stock name."""
  stock_info = yf.YahooFinancials(stock_name).get_stock_quote()
  return stock_info['symbol']

def get_stock_data(stock_name, start_date, end_date):
  """Gets the historical stock data for a given stock."""
  ticker = get_stock_ticker(stock_name)
  data = yf.Ticker(ticker).history(start=start_date, end=end_date)
  return data

def predict_stock_trend(stock_name, start_date, end_date):
  """Predicts the stock trend for a given stock."""
  data = get_stock_data(stock_name, start_date, end_date)
  close_price = data['Close']
  model = load_model('keras_modelv2.h5')
  x_test = close_price[-60:].to_numpy().reshape(1, 60, 1)
  y_predicted = model.predict(x_test)
  return y_predicted[0][0]

st.title("Stock Trend Prediction")

# User input for stock name
user_input = st.text_input("Enter Company Name", 'Apple Inc.')
start_date = st.date_input("Start Date", value='2010-01-01')
end_date = st.date_input("End Date", value='2023-07-30')

if st.button("Predict"):
  stock_trend = predict_stock_trend(user_input, start_date, end_date)
  if stock_trend > 0:
    st.write("The stock price is expected to go up.")
  elif stock_trend < 0:
    st.write("The stock price is expected to go down.")
  else:
    st.write("The stock price is expected to remain flat.")
