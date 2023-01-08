# -*- coding: utf-8 -*-
"""StockPred_6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qrgo4c7NQpWOwHtYBqp1kAZ5aEBKj0O6
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install yfinance

from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pickle

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#st.title('Stock Forecast App')

stocks = ('AAPL')
#selected_stock = st.selectbox('Select dataset for prediction', stocks)

#n_years = st.slider('Years of prediction:', 1, 4)
#period = n_years * 365
#period = 365

#@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#data_load_state = st.text('Loading data...')
data = load_data(stocks)
#data_load_state.text('Loading data... done!')

#st.subheader('Raw data')
#st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode="lines", marker_color='#636EFA'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode="lines", marker_color= '#EF553B'))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, template= 'plotly') 
	fig.show()
	
plot_raw_data()

data.head()

data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.7)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.7): int(len(data))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range= (0,1))

data_training_scale = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(99, data_training_scale.shape[0]-1):
  x_train.append(data_training_scale[i-99: i])
  y_train.append(data_training_scale[i: i+1])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0],y_train.shape[1]))

from sklearn.svm import SVR

model4 = SVR(C= 10, gamma= 0.0001,kernel='rbf')

np.random.seed(42)
model4.fit(x_train, y_train)

past_99_days = data_training.tail(99)

final_df = past_99_days.append(data_testing, ignore_index=True)

final_df_scale = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(99, final_df_scale.shape[0]-1):
  x_test.append(final_df_scale[i-99: i])
  y_test.append(final_df_scale[i: i+1])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0],y_test.shape[1]))

y_predicted = model4.predict(x_test)

y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))

y_predicted= scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test)

y_test_1d =np.reshape(y_test,(y_test.shape[0],))
y_predicted = np.reshape(y_predicted, (y_predicted.shape[0],))

# calculate the mean squared error of test data
from sklearn.metrics import mean_squared_error

# calculate errors
errors = mean_squared_error(y_test_1d, y_predicted)
# report error
print(errors)

# calculate the root mean squared error of test data
from sklearn.metrics import mean_squared_error

# calculate errors
errors = mean_squared_error(y_test_1d, y_predicted, squared=False)
# report error
print(errors)

filename= 'SVR_model'
pickle.dump(model4, open(filename, 'wb'))