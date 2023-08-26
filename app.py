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
if not data_training.empty:
    data_training_array = scaler.fit_transform(data_training)
else:
    st.warning("Data training is empty or has missing values.")

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

# ... (previous code remains the same)

# Getting the last 60 days of the original data for continuation
last_60_days = data_training[-60:].values

# Rescaling the data
last_60_days_scaled = scaler.transform(last_60_days)

# Creating an empty list for holding future predicted data
X_test = []

# Appending the last 60 days of scaled data to X_test
X_test.append(last_60_days_scaled)

# Converting X_test to a numpy array
X_test = np.array(X_test)

# Predicting the future stock prices
predicted_price = model.predict(X_test)

# Rescaling the predicted price
predicted_price = predicted_price * scaler[0]

# Creating a new dataframe for the future 2 days
future_dates = pd.date_range(start=data_testing.index[-1], periods=2)
future_df = pd.DataFrame(index=future_dates, columns=df.columns)

# Adding the predicted prices to the dataframe
future_df['Close'] = predicted_price

# Plotting the graph for original data and predicted data for the future 2 days
st.subheader('Predicted Price for the Next 2 Days')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(data_testing.index, data_testing.values, 'b', label='Original Price')
plt.plot(future_df.index, future_df['Close'], 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)  # Displays the plot in Streamlit



#Please note that the code refers to a model named 'keras_modelv2.h5' and relies on libraries like Keras, Pandas, and Streamlit. The main purpose of this code is to fetch stock data, preprocess it, use a pre-trained model to predict future stock prices, and display the results using Streamlit.
