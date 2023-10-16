import pandas as pd
import pandas_datareader as data
from pandas_datareader import data as pdr
from keras.models import load_model
import yfinance as yf    
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf

import streamlit as st

yf.pdr_override()     
start = "2010-01-01"
end = "2023-07-30"

st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker", 'AAPL')

# Create a drop-down menu for major stock exchanges
stock_exchanges = ['NYSE', 'NASDAQ', 'TSE', 'LSE', 'HKEX','NSE','BSE']
selected_stock_exchange = st.selectbox('Select a stock exchange:', stock_exchanges)

# Concatenate the stock exchange with the stock exchange keyword
stock_exchange_keyword = user_input + '.'+selected_stock_exchange

# Search for the stock price in the yfinance API
df = yf.download(stock_exchange_keyword, start, end)



df = pdr.get_data_yahoo(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.ylim([df.Close.min(), df.Close.max()])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
plt.ylim([df.Close.min(), df.Close.max()])
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig1 = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.Close)
plt.ylim([df.Close.min(), df.Close.max()])
st.pyplot(fig1)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Loading the model saved 
model = load_model('keras_modelv2.h5')

# ... (previous code remains the same)

# Testing Part
past_100_days = data_training.tail(100)
# Append next 2 days' data for prediction
next_2_days = data_testing.head(2)
final_df = pd.concat([past_100_days, next_2_days], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(60, input_data.shape[0]):  # Adjusted starting point to 60
    x_test.append(input_data[i-60:i])     # Adjusted window size to 60
    y_test.append(input_data[i, 0])



x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor

# Plotting
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
