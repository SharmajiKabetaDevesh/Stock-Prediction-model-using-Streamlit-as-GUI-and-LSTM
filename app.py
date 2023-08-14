import numpy as np
import pandas as pd
from yahoofinancials import YahooFinancials
  # Importing YahooFinancials
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

st.title("Stock Trend Prediction")

# User input for stock name
user_input = st.text_input("Enter Company Name", 'Apple Inc.')
time_frames = ['5 min', '15 min', '1 hour', '1 week']
selected_time_frame = st.selectbox("Select Time Frame", time_frames)

# Map user-friendly time frame to yfinance time interval
time_frame_mapping = {
    '5 min': '5m',
    '15 min': '15m',
    '1 hour': '1h',
    '1 week': '1wk'
}

# Convert selected time frame to yfinance time interval
selected_interval = time_frame_mapping[selected_time_frame]

# Get stock ticker using yahoofinancials library
yahoo_financials = YahooFinancials(user_input)
stock_ticker = yahoo_financials.get_stock_ticker()

# Fetch historical stock data from Yahoo Finance using yfinance
df = yf.download(stock_ticker, start="2010-01-01", end="2023-07-30", interval=selected_interval)


# Describing Data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig1 = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig1)

# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Loading the model saved 
model = load_model('keras_modelv2.h5')

# Testing Part
# Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

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
y_test = y_test * scale_factor

st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
