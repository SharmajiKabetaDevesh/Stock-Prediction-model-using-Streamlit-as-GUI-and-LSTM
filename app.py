import numpy as np
import pandas as pd
import pandas_datareader as data
from pandas_datareader import data as pdr
from keras.models import load_model
import yfinance as yf    
import tensorflow.compat.v2 as tf


import streamlit as st

yf.pdr_override()     
start = "2010-01-01"
end = "2023-07-30"

st.title("STock Trend Prediction")

user_input =st.text_input("Enter STock Ticker",'AAPL')
df = pdr.get_data_yahoo(user_input, start, end)

# Describing Data
st.subheader('Data from 2010 - 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize(12,6))
plt.plot(df.close)
st.pyplot(fig)
