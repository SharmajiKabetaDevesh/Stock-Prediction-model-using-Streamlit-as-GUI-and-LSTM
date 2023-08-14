import numpy as np
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
df = pdr.get_data_yahoo(user_input, start, end)

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
model = load_model('Trial4kera_model.keras')

# Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])  # Fix typo here

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
