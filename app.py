import numpy as np
import pandas as pd
import pandas_datareader as data
from pandas_datareader import data as pdr
from keras.models import load_model
import yfinance as yf    
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import streamlit as st

yf.pdr_override()      # Overrides Yahoo Finance to allow Pandas DataReader to fetch data
start = "2010-01-01"
end = "2023-07-30"

st.title("Stock Trend Prediction")  # Creates a title in the Streamlit app

user_input = st.text_input("Enter Stock Ticker", 'AAPL')  # User input for stock ticker

# Fetches historical stock data from Yahoo Finance
df = pdr.get_data_yahoo(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())  # Displays a summary of the data

# Creating a plot of Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)  # Displays the plot in Streamlit


# Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# Data scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Loading the pre-trained Keras model
model = load_model('keras_modelv2.h5')

# Testing Part
past_100_days = data_training.tail(100)
next_2_days = data_testing.head(2)
final_df = pd.concat([past_100_days, next_2_days], ignore_index=True)

# Data scaling and preparation for prediction
input_data = scaler.fit_transform(final_df)

# Creating input sequences for prediction
x_test = []
y_test = []
for i in range(60, input_data.shape[0]):
    x_test.append(input_data[i-60:i])
    y_test.append(input_data[i, 0])

# Conversion to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions using the loaded model
y_predicted = model.predict(x_test)

# Rescaling the predicted values to the original scale
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor

# Plotting the predictions vs original data
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)  # Displays the plot in Streamlit


#Please note that the code refers to a model named 'keras_modelv2.h5' and relies on libraries like Keras, Pandas, and Streamlit. The main purpose of this code is to fetch stock data, preprocess it, use a pre-trained model to predict future stock prices, and display the results using Streamlit.
