import datetime
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import date
import base64
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.layers import LeakyReLU
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix , classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


# ------------- SETTINGS --------------------
stocks = ['Please select stock','AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'ORCL', 'INTC']
algorithm = ['Please select Machine Learning model', 'SVR', 'LSTM', 'XG Boost Regressor', 'AdaBoost Regressor']
page_title = 'Stock Price Prediction and Analysis'
page_icon = ':chart_with_upwards_trend:'
layout = 'wide'
# -------------------------------------------

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_stock = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_yRrgc4.json")

col30, col31 = st.columns([7, 3])
with col30:
    st.title(page_title + " " + page_icon)
with col31:
    st_lottie(
        lottie_stock,
        height=240,
        width=None,
        quality="high",
        key=None,
    )

st.write(
    """<style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HIDE STREAMLIT STYLE ---

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Stock Predictions", "Technical Indicators", "Returns", "About & Credits"],
    icons=["calendar-week-fill", "bar-chart-fill", "currency-dollar", "people-fill"],
    orientation="horizontal",
)

# --------------------------------------------Stock Predictions -----------------------------------------------------

if selected == "Stock Predictions":
    st.write("""[**AAPL**: Apple Inc, **TSLA**: Tesla Inc, **AMZN**: Amazon Inc, **GOOGL**: Alphabet Inc, **MSFT**: Microsoft Corporation, **META**: Meta Platforms Inc, **ORCL** :Oracle Corporation, **INTC**: Intel Corporation]""")
    selected_stock = st.selectbox('Select stock to train and predict: ', stocks)
    selected_algorithm = st.selectbox('Select Machine Learning model to train: ', algorithm)
    period = st.slider('Days of prediction: ', min_value=5, max_value=365, value=180, step=5, help='Minimum = 5 days | Maximum = 365 days')

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    if selected_algorithm == "SVR":
        START = "2018-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        "---"

        st.subheader(f'Raw data for {selected_stock}')
        st.write("""The data frame below shows the raw data of the selected stock. The data is fetched from the [Yahoo Finance website](https://finance.yahoo.com/).""")
        st.write("""**Date**: In yyyy-mm-dd format
                    , **Open**: Price of the stock at market open
                    , **High**: Highest price reached in the day
                    , **Low**: Lowest price reached in the day
                    , **Close**: Price of the stock at market close
                    , **Adj Close**: Closing price after adjustments for all applicable splits and dividend distributions
                    , **Volume**: Number of shares traded 
                    \n  """)
        st.write("""In this project, the 'Close' column will be used instead of 'Adj Close' as it represents the exact cash value of the shares. 
                    The stock data used has a timeframe of 5 years, beginning on January 1, 2018, and ending today.""")


        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_data.csv">Download CSV File</a>'
            return href

        col1, col2, col3 = st.columns([3, 6, 2])
        with col1:
            st.write("")
        with col2:
            st.write(data)
            st.markdown(filedownload(data), unsafe_allow_html=True)
        with col3:
            st.write("")


        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode="lines", marker_color='#636EFA'))
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode="lines", marker_color='#EF553B'))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True, template='plotly')
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""The graph above shows the open and close price of selected stock . **Start Date**: 2018-01-01 , **End Date**: Today""")

        data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_scale = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(99, data_training_scale.shape[0] - 1):
            x_train.append(data_training_scale[i - 99: i])
            y_train.append(data_training_scale[i: i + 1])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        filename = 'SVR_model'
        loaded_model = pickle.load(open(filename, 'rb'))

        past_99_days = data_training.tail(99)
        final_df = past_99_days.append(data_testing, ignore_index=True)
        final_df_scale = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(99, final_df_scale.shape[0] - 1):
            x_test.append(final_df_scale[i - 99: i])
            y_test.append(final_df_scale[i: i + 1])

        x_test, y_test = np.array(x_test), np.array(y_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        y_predicted = loaded_model.predict(x_test)
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test)
        y_test_1d = np.reshape(y_test, (y_test.shape[0],))
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0],))
        errors = mean_squared_error(y_test_1d, y_predicted)
        # report error
        errors_r = mean_squared_error(y_test_1d, y_predicted, squared=False)
        # report error

        date_test = pd.DataFrame(data['Date'][int(len(data) * 0.7) + 1: int(len(data))])

        "---"

        df_viz1 = pd.DataFrame(date_test, columns=['Date'])
        df_viz1['ytest'] = y_test_1d
        df_viz1['ypred'] = y_predicted

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ytest'], name="Actual Close Price", mode="lines",
                       marker_color='#636EFA'))
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ypred'], name="Predicted Close Price", mode="lines",
                       marker_color='#EF553B'))
        fig1.layout.update(title_text='Comparison between Actual and Predicted Close Price (Test data)',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig1.update_layout(height=550)
        fig1.layout.xaxis.title.text = 'Date'
        fig1.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""The graph above illustrates the actual close price and predicted close price by the selected machine learning model. The comparison is carried out using test data to determine how well the model handles unseen data.""")


        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        training_array = np.array(data_training)
        Train_and_Predicted = np.append(training_array, y_predicted)
        Train_and_Predicted_df = pd.DataFrame(Train_and_Predicted, columns=['Close'])

        date_train = pd.DataFrame(data['Date'][0:int(len(data) * 0.7)])
        date_combine = pd.concat([date_train, date_test])
        Train_and_Predicted_df['Date'] = date_combine

        "---"

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][0:len(training_array)],
                       y=Train_and_Predicted_df['Close'][0:len(training_array)], name="Actual Close Price (Training)",
                       mode="lines",
                       marker_color='#636EFA'))
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][len(training_array):],
                       y=Train_and_Predicted_df['Close'][len(training_array):], name="Predicted Close Price (Test)",
                       mode="lines",
                       marker_color='#EF553B'))
        fig2.layout.update(title_text='Predicted Close Price by SVR model',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig2.update_layout(height=550)
        fig2.layout.xaxis.title.text = 'Date'
        fig2.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""The graph above depicts the close price prediction by the selected machine learning algorithm. The blue line represents the actual data that is used as the training set. It is then extended by the red line, which represents the prediction result of ML model on test data. """)
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).
                    \n- Stock market does not operate on weekends, eg: after June 25, the market reopens on June 28, 2021""")


        y_test_1d_df = pd.DataFrame(y_test_1d, columns=['Close'])
        direction_tmp = y_test_1d_df.shift(-1)
        y_test_1d_df['close_nextday'] = direction_tmp
        y_test_1d_df = y_test_1d_df.dropna()
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] >= y_test_1d_df['Close'], 'direction'] = 1
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] < y_test_1d_df['Close'], 'direction'] = 0

        y_predicted_df = pd.DataFrame(y_predicted, columns=['Predicted_Close'])
        direction_tmp_predicted = y_predicted_df.shift(-1)
        y_predicted_df['Predicted_close_nextday'] = direction_tmp_predicted
        y_predicted_df = y_predicted_df.dropna()
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] >= y_predicted_df['Predicted_Close'], 'direction_f'] = 1
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] < y_predicted_df['Predicted_Close'], 'direction_f'] = 0

        report = classification_report(y_test_1d_df.direction, y_predicted_df.direction_f, output_dict=True)
        SVR = {'Class': ['Downwards Trend', 'Upwards Trend'],
               'Precision': [report['0.0']['precision'], report['1.0']['precision']],
               'Recall': [report['0.0']['recall'], report['1.0']['recall']],
               'F1-score': [report['0.0']['f1-score'], report['1.0']['f1-score']],
               'Accuracy': [report['accuracy'], report['accuracy']]}

        SVR_df = pd.DataFrame(SVR)
        SVR_df = SVR_df.round(decimals=3)

        matrix = confusion_matrix(y_test_1d_df.direction, y_predicted_df.direction_f)
        plt.figure(figsize=(3, 2))
        heatmap = sn.heatmap(matrix, annot=True, fmt='d', annot_kws={
            'size': 7
        })
        plt.xlabel('Predicted', fontdict={'fontsize': 6})
        plt.ylabel('Truth', fontdict={'fontsize': 6})
        plt.title('Confusion Matrix for XGBoost Regressor', fontdict={'fontsize': 7})
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=6)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=6)

        "---"

        st.subheader('Model Evaluation')
        col10, col11, col25, col12 = st.columns([0.7, 4, 0.3, 4.1])
        with col10:
            st.write("""""")
        with col11:
            st.pyplot(plt)
        with col25:
            st.write("""""")
        with col12:
            st.dataframe(SVR_df)
            st.metric(label='Mean Squared Errors', value="{:.3f}".format(errors), help='MSE for test data')
            st.metric(label='Root Mean Squared Errors', value="{:.3f}".format(errors_r), help='RMSE for test data')

        st.write("""For the regression part (Stock Prices), Mean Squared Errors (MSE) and Root Mean Squared Errors (RMSE) are used as the evaluation metrics. The model is more accurate when the MSE and RMSE are near to zero.""")
        st.write("""For the stock trend prediction (Upwards or downwards), evaluation metrics such as precision, recall, F1-score, and accuracy are used. The confusion matrix is also displayed.""")

        "---"

        last = x_test[-1:]
        for i in range(99, (99 - 1 + period)):
            last_copy = np.reshape(last[0][i - 99:i], (1, 99))
            lastvalue = loaded_model.predict(last_copy)
            lastvalue_2d = np.reshape(lastvalue, (1, 1))
            last = np.append(last, lastvalue_2d)
            last = np.reshape(last, (1, i + 1))

        last = np.reshape(last, (last.shape[1], 1))
        last_df = pd.DataFrame(last, columns=['Predict_Close'])
        last_df = scaler.inverse_transform(last_df)

        overall_data = data['Close']
        overall_predict = last_df[99:]
        overall_combine = np.append(overall_data, overall_predict)
        overall_combine_df = pd.DataFrame(overall_combine, columns=['Close'])

        overall_combine_df['Day'] = overall_combine_df.index

        st.subheader('Future Prediction')

        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][0:len(overall_data)],
                       y=overall_combine_df['Close'][0:len(overall_data)], name="Actual Close Price",
                       mode="lines",
                       marker_color='#636EFA'))
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][len(overall_data):].index,
                       y=overall_combine_df['Close'][len(overall_data):], name="Predicted Future Close Price",
                       mode="lines",
                       marker_color='#EF553B'))
        fig4.layout.update(title_text=f'Predicted Future {period} days Close Price',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig4.update_layout(height=550)
        fig4.layout.xaxis.title.text = 'Day'
        fig4.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig4, use_container_width=True)
        st.write("""The blue line indicates the actual stock closing price from the starting date until today. The red line represents the forecasted stock future close price by the selected ML model.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day). """)


    if selected_algorithm == "LSTM":
        START = "2018-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        "---"

        st.subheader(f'Raw data for {selected_stock}')
        st.write("""The data frame below shows the raw data of the selected stock. The data is fetched from the [Yahoo Finance website](https://finance.yahoo.com/).""")
        st.write("""**Date**: In yyyy-mm-dd format
                    , **Open**: Price of the stock at market open
                    , **High**: Highest price reached in the day
                    , **Low**: Lowest price reached in the day
                    , **Close**: Price of the stock at market close
                    , **Adj Close**: Closing price after adjustments for all applicable splits and dividend distributions
                    , **Volume**: Number of shares traded 
                    \n  """)
        st.write("""In this project, the 'Close' column will be used instead of 'Adj Close' as it represents the exact cash value of the shares. 
                    The stock data used has a timeframe of 5 years, beginning on January 1, 2018, and ending today.""")

        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_data.csv">Download CSV File</a>'
            return href

        col1, col2, col3 = st.columns([3, 6, 2])
        with col1:
            st.write("")
        with col2:
            st.write(data)
            st.markdown(filedownload(data), unsafe_allow_html=True)
        with col3:
            st.write("")


        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode="lines", marker_color='#636EFA'))
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode="lines", marker_color='#EF553B'))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True,
                          template='plotly')
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""The graph above shows the open and close price of selected stock . **Start Date**: 2018-01-01 , **End Date**: Today""")

        data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_scale = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(99, data_training_scale.shape[0] - 1):
            x_train.append(data_training_scale[i - 99: i])
            y_train.append(data_training_scale[i: i + 1])

        x_train, y_train = np.array(x_train), np.array(y_train)

        loaded_model = load_model('model.h5')

        past_99_days = data_training.tail(99)
        final_df = past_99_days.append(data_testing, ignore_index=True)
        final_df_scale = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(99, final_df_scale.shape[0] - 1):
            x_test.append(final_df_scale[i - 99: i])
            y_test.append(final_df_scale[i: i + 1])

        x_test, y_test = np.array(x_test), np.array(y_test)
        y_predicted = loaded_model.predict(x_test)
        y_test_2d = np.reshape(y_test, (y_test.shape[0], 1))
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test_2d = scaler.inverse_transform(y_test_2d)

        # calculate the mean squared error of test data
        # calculate errors
        errors = mean_squared_error(y_test_2d, y_predicted)
        # report error
        errors_r = mean_squared_error(y_test_2d, y_predicted, squared=False)
        # report error

        date_test = pd.DataFrame(data['Date'][int(len(data) * 0.7) + 1: int(len(data))])

        "---"

        df_viz1 = pd.DataFrame(date_test, columns=['Date'])
        df_viz1['ytest'] = y_test_2d
        df_viz1['ypred'] = y_predicted

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ytest'], name="Actual Close Price", mode="lines",
                       marker_color='#636EFA'))
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ypred'], name="Predicted Close Price", mode="lines",
                       marker_color='#EF553B'))
        fig1.layout.update(title_text='Comparison between Actual and Predicted Close Price (Test data)',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig1.update_layout(height=550)
        fig1.layout.xaxis.title.text = 'Date'
        fig1.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""The graph above illustrates the actual close price and predicted close price by the selected machine learning model. The comparison is carried out using test data to determine how well the model handles unseen data.""")


        training_array = np.array(data_training)
        Train_and_Predicted = np.append(training_array, y_predicted)
        Train_and_Predicted_df = pd.DataFrame(Train_and_Predicted, columns=['Close'])

        date_train = pd.DataFrame(data['Date'][0:int(len(data) * 0.7)])
        date_combine = pd.concat([date_train, date_test])
        Train_and_Predicted_df['Date'] = date_combine

        "---"

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][0:len(training_array)],
                       y=Train_and_Predicted_df['Close'][0:len(training_array)], name="Actual Close Price (Training)",
                       mode="lines",
                       marker_color='#636EFA'))
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][len(training_array):],
                       y=Train_and_Predicted_df['Close'][len(training_array):], name="Predicted Close Price (Test)",
                       mode="lines",
                       marker_color='#EF553B'))
        fig2.layout.update(title_text='Predicted Close Price by LSTM model',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig2.update_layout(height=550)
        fig2.layout.xaxis.title.text = 'Date'
        fig2.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""The graph above depicts the close price prediction by the selected machine learning algorithm. The blue line represents the actual data that is used as the training set. It is then extended by the red line, which represents the prediction result of ML model on test data.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).
                    \n- Stock market does not operate on weekends, eg: after June 25, the market reopens on June 28, 2021""")

        y_test_2d_df = pd.DataFrame(y_test_2d, columns=['Close'])
        direction_tmp = y_test_2d_df.shift(-1)
        y_test_2d_df['close_nextday'] = direction_tmp
        y_test_2d_df = y_test_2d_df.dropna()
        y_test_2d_df.loc[y_test_2d_df['close_nextday'] >= y_test_2d_df['Close'], 'direction'] = 1
        y_test_2d_df.loc[y_test_2d_df['close_nextday'] < y_test_2d_df['Close'], 'direction'] = 0

        y_predicted_df = pd.DataFrame(y_predicted, columns=['Predicted_Close'])
        direction_tmp_predicted = y_predicted_df.shift(-1)
        y_predicted_df['Predicted_close_nextday'] = direction_tmp_predicted
        y_predicted_df = y_predicted_df.dropna()
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] >= y_predicted_df['Predicted_Close'], 'direction_f'] = 1
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] < y_predicted_df['Predicted_Close'], 'direction_f'] = 0

        report = classification_report(y_test_2d_df.direction, y_predicted_df.direction_f, output_dict=True)
        LSTM = {'Class': ['Downwards Trend', 'Upwards Trend'],
               'Precision': [report['0.0']['precision'], report['1.0']['precision']],
               'Recall': [report['0.0']['recall'], report['1.0']['recall']],
               'F1-score': [report['0.0']['f1-score'], report['1.0']['f1-score']],
               'Accuracy': [report['accuracy'], report['accuracy']]}

        LSTM_df = pd.DataFrame(LSTM)
        LSTM_df = LSTM_df.round(decimals=3)

        matrix = confusion_matrix(y_test_2d_df.direction, y_predicted_df.direction_f)
        plt.figure(figsize=(3, 2))
        heatmap = sn.heatmap(matrix, annot=True, fmt='d', annot_kws={
            'size': 7
        })
        plt.xlabel('Predicted', fontdict={'fontsize': 6})
        plt.ylabel('Truth', fontdict={'fontsize': 6})
        plt.title('Confusion Matrix for XGBoost Regressor', fontdict={'fontsize': 7})
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=6)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=6)

        "---"

        st.subheader('Model Evaluation')
        col10, col11, col18, col12 = st.columns([0.7, 4, 0.3, 4.1])
        with col10:
            st.write("""""")
        with col11:
            st.pyplot(plt)
        with col18:
            st.write("""""")
        with col12:
            st.dataframe(LSTM_df)
            st.metric(label='Mean Squared Errors', value="{:.3f}".format(errors), help='MSE for test data')
            st.metric(label='Root Mean Squared Errors', value="{:.3f}".format(errors_r), help='RMSE for test data')

        st.write("""For the regression part (Stock Prices), Mean Squared Errors (MSE) and Root Mean Squared Errors (RMSE) are used as the evaluation metrics. The model is more accurate when the MSE and RMSE are near to zero.""")
        st.write("""For the stock trend prediction (Upwards or downwards), evaluation metrics such as precision, recall, F1-score, and accuracy are used. The confusion matrix is also displayed.""")

        "---"

        last = x_test[-1:]
        for i in range(99, (99 - 1 + period)):
            last_copy = np.reshape(last[0][i - 99:i], (1, 99, 1))
            lastvalue = loaded_model.predict(last_copy)
            lastvalue_3d = np.reshape(lastvalue, (1, 1, 1))
            last = np.append(last, lastvalue_3d)
            last = np.reshape(last, (1, i + 1, 1))

        last_2d = np.reshape(last, (last.shape[1], 1))
        last_df = pd.DataFrame(last_2d, columns=['Predict_Close'])
        last_df = scaler.inverse_transform(last_df)

        overall_data = data['Close']
        overall_predict = last_df[99:]
        overall_combine = np.append(overall_data, overall_predict)
        overall_combine_df = pd.DataFrame(overall_combine, columns=['Close'])

        overall_combine_df['Day'] = overall_combine_df.index

        st.subheader('Future Prediction')
        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][0:len(overall_data)],
                       y=overall_combine_df['Close'][0:len(overall_data)], name="Actual Close Price",
                       mode="lines",
                       marker_color='#636EFA'))
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][len(overall_data):].index,
                       y=overall_combine_df['Close'][len(overall_data):], name="Predicted Future Close Price",
                       mode="lines",
                       marker_color='#EF553B'))
        fig4.layout.update(title_text=f'Predicted Future {period} days Close Price',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig4.update_layout(height=550)
        fig4.layout.xaxis.title.text = 'Day'
        fig4.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig4, use_container_width=True)
        st.write("""The blue line indicates the actual stock closing price from the starting date until today. The red line represents the forecasted stock future close price by the selected ML model.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day). """)


    if selected_algorithm == "XG Boost Regressor":
        START = "2018-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        "---"

        st.subheader(f'Raw data for {selected_stock}')
        st.write("""The data frame below shows the raw data of the selected stock. The data is fetched from the [Yahoo Finance website](https://finance.yahoo.com/).""")
        st.write("""**Date**: In yyyy-mm-dd format
                    , **Open**: Price of the stock at market open
                    , **High**: Highest price reached in the day
                    , **Low**: Lowest price reached in the day
                    , **Close**: Price of the stock at market close
                    , **Adj Close**: Closing price after adjustments for all applicable splits and dividend distributions
                    , **Volume**: Number of shares traded 
                    \n  """)
        st.write("""In this project, the 'Close' column will be used instead of 'Adj Close' as it represents the exact cash value of the shares. 
                    The stock data used has a timeframe of 5 years, beginning on January 1, 2018, and ending today.""")

        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_data.csv">Download CSV File</a>'
            return href

        col1, col2, col3 = st.columns([3, 6, 2])
        with col1:
            st.write("")
        with col2:
            st.write(data)
            st.markdown(filedownload(data), unsafe_allow_html=True)
        with col3:
            st.write("")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode="lines", marker_color='#636EFA'))
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode="lines", marker_color='#EF553B'))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True,
                          template='plotly')
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""The graph above shows the open and close price of selected stock . **Start Date**: 2018-01-01 , **End Date**: Today""")

        data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_scale = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(99, data_training_scale.shape[0] - 1):
            x_train.append(data_training_scale[i - 99: i])
            y_train.append(data_training_scale[i: i + 1])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        filename = 'XGB_model'
        loaded_model = pickle.load(open(filename, 'rb'))

        past_99_days = data_training.tail(99)
        final_df = past_99_days.append(data_testing, ignore_index=True)
        final_df_scale = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(99, final_df_scale.shape[0] - 1):
            x_test.append(final_df_scale[i - 99: i])
            y_test.append(final_df_scale[i: i + 1])

        x_test, y_test = np.array(x_test), np.array(y_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        y_predicted = loaded_model.predict(x_test)
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test)
        y_test_1d = np.reshape(y_test, (y_test.shape[0],))
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0],))
        errors = mean_squared_error(y_test_1d, y_predicted)
        # report error
        errors_r = mean_squared_error(y_test_1d, y_predicted, squared=False)
        # report error

        date_test = pd.DataFrame(data['Date'][int(len(data) * 0.7) + 1: int(len(data))])

        "---"

        df_viz1 = pd.DataFrame(date_test, columns=['Date'])
        df_viz1['ytest'] = y_test_1d
        df_viz1['ypred'] = y_predicted

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ytest'], name="Actual Close Price", mode="lines", marker_color='#636EFA'))
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ypred'], name="Predicted Close Price", mode="lines", marker_color='#EF553B'))
        fig1.layout.update(title_text='Comparison between Actual and Predicted Close Price (Test data)', xaxis_rangeslider_visible=True,
                          template='plotly')
        fig1.update_layout(height=550)
        fig1.layout.xaxis.title.text = 'Date'
        fig1.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""The graph above illustrates the actual close price and predicted close price by the selected machine learning model. The comparison is carried out using test data to determine how well the model handles unseen data.""")


        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        training_array = np.array(data_training)
        Train_and_Predicted = np.append(training_array, y_predicted)
        Train_and_Predicted_df = pd.DataFrame(Train_and_Predicted, columns=['Close'])

        date_train = pd.DataFrame(data['Date'][0:int(len(data) * 0.7)])
        date_combine = pd.concat([date_train, date_test])
        Train_and_Predicted_df['Date'] = date_combine

        "---"

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][0:len(training_array)], y=Train_and_Predicted_df['Close'][0:len(training_array)], name="Actual Close Price (Training)", mode="lines",
                       marker_color='#636EFA'))
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][len(training_array):], y=Train_and_Predicted_df['Close'][len(training_array):], name="Predicted Close Price (Test)", mode="lines",
                       marker_color='#EF553B'))
        fig2.layout.update(title_text='Predicted Close Price by XG Boost Regressor model',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig2.update_layout(height=550)
        fig2.layout.xaxis.title.text = 'Date'
        fig2.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""The graph above depicts the close price prediction by the selected machine learning algorithm. The blue line represents the actual data that is used as the training set. It is then extended by the red line, which represents the prediction result of ML model on test data. """)
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).
                    \n- Stock market does not operate on weekends, eg: after June 25, the market reopens on June 28, 2021""")


        y_test_1d_df = pd.DataFrame(y_test_1d, columns=['Close'])
        direction_tmp = y_test_1d_df.shift(-1)
        y_test_1d_df['close_nextday'] = direction_tmp
        y_test_1d_df = y_test_1d_df.dropna()
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] >= y_test_1d_df['Close'], 'direction'] = 1
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] < y_test_1d_df['Close'], 'direction'] = 0

        y_predicted_df = pd.DataFrame(y_predicted, columns=['Predicted_Close'])
        direction_tmp_predicted = y_predicted_df.shift(-1)
        y_predicted_df['Predicted_close_nextday'] = direction_tmp_predicted
        y_predicted_df = y_predicted_df.dropna()
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] >= y_predicted_df['Predicted_Close'], 'direction_f'] = 1
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] < y_predicted_df['Predicted_Close'], 'direction_f'] = 0

        report = classification_report(y_test_1d_df.direction, y_predicted_df.direction_f, output_dict=True)
        XGB = {'Class': ['Downwards Trend', 'Upwards Trend'],
               'Precision': [report['0.0']['precision'], report['1.0']['precision']],
               'Recall': [report['0.0']['recall'], report['1.0']['recall']],
               'F1-score': [report['0.0']['f1-score'], report['1.0']['f1-score']],
               'Accuracy': [report['accuracy'], report['accuracy']]}

        XGB_df = pd.DataFrame(XGB)
        XGB_df = XGB_df.round(decimals=3)

        matrix = confusion_matrix(y_test_1d_df.direction, y_predicted_df.direction_f)
        plt.figure(figsize=(3, 2))
        heatmap = sn.heatmap(matrix, annot=True, fmt='d', annot_kws={
                'size': 7
            })
        plt.xlabel('Predicted', fontdict={'fontsize': 6})
        plt.ylabel('Truth', fontdict={'fontsize': 6})
        plt.title('Confusion Matrix for XGBoost Regressor', fontdict={'fontsize': 7})
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=6)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=6)

        "---"

        st.subheader('Model Evaluation')
        col10, col11, col25 ,col12 = st.columns([0.7, 4, 0.3, 4.1])
        with col10:
            st.write("""""")
        with col11:
            st.pyplot(plt)
        with col25:
            st.write("""""")
        with col12:
            st.dataframe(XGB_df)
            st.metric(label='Mean Squared Errors', value="{:.3f}".format(errors), help='MSE for test data')
            st.metric(label='Root Mean Squared Errors', value="{:.3f}".format(errors_r), help='RMSE for test data')

        st.write("""For the regression part (Stock Prices), Mean Squared Errors (MSE) and Root Mean Squared Errors (RMSE) are used as the evaluation metrics. The model is more accurate when the MSE and RMSE are near to zero.""")
        st.write("""For the stock trend prediction (Upwards or downwards), evaluation metrics such as precision, recall, F1-score, and accuracy are used. The confusion matrix is also displayed.""")

        "---"

        last = x_test[-1:]
        for i in range(99, (99 - 1 + period)):
            last_copy = np.reshape(last[0][i - 99:i], (1, 99))
            lastvalue = loaded_model.predict(last_copy)
            lastvalue_2d = np.reshape(lastvalue, (1, 1))
            last = np.append(last, lastvalue_2d)
            last = np.reshape(last, (1, i + 1))

        last = np.reshape(last, (last.shape[1], 1))
        last_df = pd.DataFrame(last, columns=['Predict_Close'])
        last_df = scaler.inverse_transform(last_df)

        overall_data = data['Close']
        overall_predict = last_df[99:]
        overall_combine = np.append(overall_data, overall_predict)
        overall_combine_df = pd.DataFrame(overall_combine, columns=['Close'])

        overall_combine_df['Day'] = overall_combine_df.index

        st.subheader('Future Prediction')
        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][0:len(overall_data)], y=overall_combine_df['Close'][0:len(overall_data)], name="Actual Close Price",
                       mode="lines",
                       marker_color='#636EFA'))
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][len(overall_data):].index, y=overall_combine_df['Close'][len(overall_data):], name="Predicted Future Close Price",
                       mode="lines",
                       marker_color='#EF553B'))
        fig4.layout.update(title_text=f'Predicted Future {period} days Close Price',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig4.update_layout(height=550)
        fig4.layout.xaxis.title.text = 'Day'
        fig4.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig4, use_container_width=True)
        st.write("""The blue line indicates the actual stock closing price from the starting date until today. The red line represents the forecasted stock future close price by the selected ML model.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).""")


    if selected_algorithm == "AdaBoost Regressor":
        START = "2018-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        "---"

        st.subheader(f'Raw data for {selected_stock}')
        st.write("""The data frame below shows the raw data of the selected stock. The data is fetched from the [Yahoo Finance website](https://finance.yahoo.com/).""")
        st.write("""**Date**: In yyyy-mm-dd format
                    , **Open**: Price of the stock at market open
                    , **High**: Highest price reached in the day
                    , **Low**: Lowest price reached in the day
                    , **Close**: Price of the stock at market close
                    , **Adj Close**: Closing price after adjustments for all applicable splits and dividend distributions
                    , **Volume**: Number of shares traded 
                    \n  """)
        st.write("""In this project, the 'Close' column will be used instead of 'Adj Close' as it represents the exact cash value of the shares. 
                    The stock data used has a timeframe of 5 years, beginning on January 1, 2018, and ending today.""")

        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="stock_data.csv">Download CSV File</a>'
            return href

        col1, col2, col3 = st.columns([3, 6, 2])
        with col1:
            st.write("")
        with col2:
            st.write(data)
            st.markdown(filedownload(data), unsafe_allow_html=True)
        with col3:
            st.write("")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Open'], name="stock_open", mode="lines", marker_color='#636EFA'))
        fig.add_trace(
            go.Scatter(x=data['Date'], y=data['Close'], name="stock_close", mode="lines", marker_color='#EF553B'))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True,
                          template='plotly')
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""The graph above shows the open and close price of selected stock . **Start Date**: 2018-01-01 , **End Date**: Today""")

        data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
        data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.7): int(len(data))])

        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_scale = scaler.fit_transform(data_training)

        x_train = []
        y_train = []

        for i in range(99, data_training_scale.shape[0] - 1):
            x_train.append(data_training_scale[i - 99: i])
            y_train.append(data_training_scale[i: i + 1])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1]))

        filename = 'ADB_model'
        loaded_model = pickle.load(open(filename, 'rb'))

        past_99_days = data_training.tail(99)
        final_df = past_99_days.append(data_testing, ignore_index=True)
        final_df_scale = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(99, final_df_scale.shape[0] - 1):
            x_test.append(final_df_scale[i - 99: i])
            y_test.append(final_df_scale[i: i + 1])

        x_test, y_test = np.array(x_test), np.array(y_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))

        y_predicted = loaded_model.predict(x_test)
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test)
        y_test_1d = np.reshape(y_test, (y_test.shape[0],))
        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0],))
        errors = mean_squared_error(y_test_1d, y_predicted)
        # report error
        errors_r = mean_squared_error(y_test_1d, y_predicted, squared=False)
        # report error

        date_test = pd.DataFrame(data['Date'][int(len(data) * 0.7) + 1: int(len(data))])

        "---"

        df_viz1 = pd.DataFrame(date_test, columns=['Date'])
        df_viz1['ytest'] = y_test_1d
        df_viz1['ypred'] = y_predicted

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ytest'], name="Actual Close Price", mode="lines",
                       marker_color='#636EFA'))
        fig1.add_trace(
            go.Scatter(x=df_viz1['Date'], y=df_viz1['ypred'], name="Predicted Close Price", mode="lines",
                       marker_color='#EF553B'))
        fig1.layout.update(title_text='Comparison between Actual and Predicted Close Price (Test data)',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig1.update_layout(height=550)
        fig1.layout.xaxis.title.text = 'Date'
        fig1.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig1, use_container_width=True)
        st.write("""The graph above illustrates the actual close price and predicted close price by the selected machine learning model. The comparison is carried out using test data to determine how well the model handles unseen data.""")


        y_predicted = np.reshape(y_predicted, (y_predicted.shape[0], 1))
        training_array = np.array(data_training)
        Train_and_Predicted = np.append(training_array, y_predicted)
        Train_and_Predicted_df = pd.DataFrame(Train_and_Predicted, columns=['Close'])

        date_train = pd.DataFrame(data['Date'][0:int(len(data) * 0.7)])
        date_combine = pd.concat([date_train, date_test])
        Train_and_Predicted_df['Date'] = date_combine

        "---"

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][0:len(training_array)],
                       y=Train_and_Predicted_df['Close'][0:len(training_array)], name="Actual Close Price (Training)",
                       mode="lines",
                       marker_color='#636EFA'))
        fig2.add_trace(
            go.Scatter(x=Train_and_Predicted_df['Date'][len(training_array):],
                       y=Train_and_Predicted_df['Close'][len(training_array):], name="Predicted Close Price (Test)",
                       mode="lines",
                       marker_color='#EF553B'))
        fig2.layout.update(title_text='Predicted Close Price by AdaBoost Regressor model',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig2.update_layout(height=550)
        fig2.layout.xaxis.title.text = 'Date'
        fig2.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig2, use_container_width=True)
        st.write("""The graph above depicts the close price prediction by the selected machine learning algorithm. The blue line represents the actual data that is used as the training set. It is then extended by the red line, which represents the prediction result of ML model on test data.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).
                    \n- Stock market does not operate on weekends, eg: after June 25, the market reopens on June 28, 2021""")


        y_test_1d_df = pd.DataFrame(y_test_1d, columns=['Close'])
        direction_tmp = y_test_1d_df.shift(-1)
        y_test_1d_df['close_nextday'] = direction_tmp
        y_test_1d_df = y_test_1d_df.dropna()
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] >= y_test_1d_df['Close'], 'direction'] = 1
        y_test_1d_df.loc[y_test_1d_df['close_nextday'] < y_test_1d_df['Close'], 'direction'] = 0

        y_predicted_df = pd.DataFrame(y_predicted, columns=['Predicted_Close'])
        direction_tmp_predicted = y_predicted_df.shift(-1)
        y_predicted_df['Predicted_close_nextday'] = direction_tmp_predicted
        y_predicted_df = y_predicted_df.dropna()
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] >= y_predicted_df['Predicted_Close'], 'direction_f'] = 1
        y_predicted_df.loc[y_predicted_df['Predicted_close_nextday'] < y_predicted_df['Predicted_Close'], 'direction_f'] = 0

        report = classification_report(y_test_1d_df.direction, y_predicted_df.direction_f, output_dict=True)
        ADB = {'Class': ['Downwards Trend', 'Upwards Trend'],
               'Precision': [report['0.0']['precision'], report['1.0']['precision']],
               'Recall': [report['0.0']['recall'], report['1.0']['recall']],
               'F1-score': [report['0.0']['f1-score'], report['1.0']['f1-score']],
               'Accuracy': [report['accuracy'], report['accuracy']]}

        ADB_df = pd.DataFrame(ADB)
        ADB_df = ADB_df.round(decimals=3)

        matrix = confusion_matrix(y_test_1d_df.direction, y_predicted_df.direction_f)
        plt.figure(figsize=(3, 2))
        heatmap = sn.heatmap(matrix, annot=True, fmt='d', annot_kws={
            'size': 7
        })
        plt.xlabel('Predicted', fontdict={'fontsize': 6})
        plt.ylabel('Truth', fontdict={'fontsize': 6})
        plt.title('Confusion Matrix for XGBoost Regressor', fontdict={'fontsize': 7})
        heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=6)
        heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=6)

        "---"

        st.subheader('Model Evaluation')
        col10, col11, col25, col12 = st.columns([0.7, 4, 0.3, 4.1])
        with col10:
            st.write("""""")
        with col11:
            st.pyplot(plt)
        with col25:
            st.write("""""")
        with col12:
            st.dataframe(ADB_df)
            st.metric(label='Mean Squared Errors', value="{:.3f}".format(errors), help='MSE for test data')
            st.metric(label='Root Mean Squared Errors', value="{:.3f}".format(errors_r), help='RMSE for test data')

        st.write("""For the regression part (Stock Prices), Mean Squared Errors (MSE) and Root Mean Squared Errors (RMSE) are used as the evaluation metrics. The model is more accurate when the MSE and RMSE are near to zero.""")
        st.write("""For the stock trend prediction (Upwards or downwards), evaluation metrics such as precision, recall, F1-score, and accuracy are used. The confusion matrix is also displayed.""")

        "---"

        last = x_test[-1:]
        for i in range(99, (99 - 1 + period)):
            last_copy = np.reshape(last[0][i - 99:i], (1, 99))
            lastvalue = loaded_model.predict(last_copy)
            lastvalue_2d = np.reshape(lastvalue, (1, 1))
            last = np.append(last, lastvalue_2d)
            last = np.reshape(last, (1, i + 1))

        last = np.reshape(last, (last.shape[1], 1))
        last_df = pd.DataFrame(last, columns=['Predict_Close'])
        last_df = scaler.inverse_transform(last_df)

        overall_data = data['Close']
        overall_predict = last_df[99:]
        overall_combine = np.append(overall_data, overall_predict)
        overall_combine_df = pd.DataFrame(overall_combine, columns=['Close'])

        overall_combine_df['Day'] = overall_combine_df.index

        st.subheader('Future Prediction')
        fig4 = go.Figure()
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][0:len(overall_data)],
                       y=overall_combine_df['Close'][0:len(overall_data)], name="Actual Close Price",
                       mode="lines",
                       marker_color='#636EFA'))
        fig4.add_trace(
            go.Scatter(x=overall_combine_df['Day'][len(overall_data):].index,
                       y=overall_combine_df['Close'][len(overall_data):], name="Predicted Future Close Price",
                       mode="lines",
                       marker_color='#EF553B'))
        fig4.layout.update(title_text=f'Predicted Future {period} days Close Price',
                           xaxis_rangeslider_visible=True,
                           template='plotly')
        fig4.update_layout(height=550)
        fig4.layout.xaxis.title.text = 'Day'
        fig4.layout.yaxis.title.text = 'Close Price'
        st.plotly_chart(fig4, use_container_width=True)
        st.write("""The blue line indicates the actual stock closing price from the starting date until today. The red line represents the forecasted stock future close price by the selected ML model.""")
        st.write("""The gap between the blue and red lines is due to:
                    \n- Differences between the actual (last day) and forecasted close prices (first day).""")


# --------------------------------------------Technical Indicators -----------------------------------------------------

if selected == "Technical Indicators":
    st.write("""[**AAPL**: Apple Inc, **TSLA**: Tesla Inc, **AMZN**: Amazon Inc, **GOOGL**: Alphabet Inc, **MSFT**: Microsoft Corporation, **META**: Meta Platforms Inc, **ORCL** :Oracle Corporation, **INTC**: Intel Corporation]""")
    START = "2018-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    selected_stock = st.selectbox('Select stock for analysis: ', stocks, index=1)

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    "---"

    st.subheader(f'Raw data for {selected_stock}')

    col1, col2, col3 = st.columns([3, 6, 2])
    with col1:
        st.write("")
    with col2:
        st.write(data)
    with col3:
        st.write("")

    "---"

    st.subheader(f'Technical Indicator Analysis for {selected_stock}')
    st.write("""An technical indicator mathematically derived from price, trading volume, investor sentiments, or open interest data and applied to interpret stock market trends and investment decisions.""")
    st.write("""The indicator used for analysis here is **100-day Moving Average** and **200-day Moving Average**. """)
    st.write("""A moving average is an arithmetic mean of a certain number of data points. """)
    st.write("""The 100-day moving average is calculated by summing up the past 100 data points and then dividing the result by 100, while the 200-day moving average is calculated by summing the past 200 days and dividing the result by 200.""")

    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(data.Close)
    plt.plot(ma100, 'r', label='100-day Moving Average (MA100)')
    plt.plot(ma200, 'g', label='200-day Moving Average (MA200)')
    plt.xlabel('Day')
    plt.ylabel('Close Price')
    plt.title('Technical Indicator Analysis')
    plt.legend()

    col4, col5, col6 = st.columns([1, 6, 1])
    with col4:
        st.write("")
    with col5:
        st.pyplot(plt)
    with col6:
        st.write("")

    st.write("""**Golden Cross Rule**:
                \n-When the shorter-term MA (MA100) crosses above the longer-term MA (MA200), it's a buy signal, as it indicates that the trend is shifting up.""")
    st.write("""**Death Cross Rule**:
                \n-When the shorter-term MA crosses below the longer-term MA, it's a sell signal, as it indicates that the trend is shifting down.""")

# --------------------------------------------Returns -----------------------------------------------------

if selected == "Returns":

    st.write("""[**AAPL**: Apple Inc, **TSLA**: Tesla Inc, **AMZN**: Amazon Inc, **GOOGL**: Alphabet Inc, **MSFT**: Microsoft Corporation, **META**: Meta Platforms Inc, **ORCL** :Oracle Corporation, **INTC**: Intel Corporation]""")
    selected_stock = st.multiselect('Select stock(s) here: ', stocks, ['AAPL', 'AMZN'])
    START = st.date_input('Start Date', datetime.date(2018,1,1))
    TODAY = st.date_input('End Date', datetime.date.today())

    START = START.strftime("%Y-%m-%d")
    TODAY = TODAY.strftime("%Y-%m-%d")

    "---"

    def relativeret(df):
        rel = df.pct_change()
        cumret = (1 + rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret

    df = relativeret(yf.download(selected_stock, START, TODAY)['Close'])

    st.subheader(f'Returns comparison between {selected_stock}')
    st.line_chart(df)
    st.write("""The graph above shows the returns of selected stocks. Each of the stocks is adjusted to the **same scale** so that they can be compared. Returns of multiple stocks can be compared together within specific timeframe.""")


# --------------------------------------------About & Credits -----------------------------------------------------

if selected == "About & Credits":
    st.subheader('About :question:')

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_about = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_sy6jjyct.json")

    col1, col2 = st.columns([6,3])
    with col1:
        st.write("""
            This purpose of this website is to act as an online platform that assists stock investors in analyzing the market and providing better decision support.
            \nThere are four sections in this website. 
            \n**Stock Prediction**:       Train and build different Machine Learning models and use to predict future stock prices. 
            \n**Technical Indicators**:   Analyze the stock trends with MA-100 and MA-200
            \n**Returns**:                Comparing the performance and returns between stocks
            \n**About & Credits**:        Submit your feedback for future improvements!
        """)
    with col2:
        st_lottie(
            lottie_about,
            height=340,
            width=None,
            quality="high",
            key=None,
        )

    "---"
    st.subheader('Credits :star2: :computer:')
    st.write("""\nThis website is made by Tiow Kit Keong from University of Malaya.
                \nThe project is supervised by Associate Prof. Dr. Azah Anir Binti Norman.""")

    "---"

    st.subheader(":mailbox: Get In Touch With Me!")
    st.write("""\nIf you have any feedback, don't hesitate to fill in this form.""")

    contact_form = """
    <form action="https://formsubmit.co/kitkeongtiow1105@gmail.com" method="POST">
         <input type="hidden" name="_captcha" value="false">
         <input type="text" name="name" placeholder="Your name" required>
         <input type="email" name="email" placeholder="Your email" required>
         <textarea name="message" placeholder="Your feedback here"></textarea>
         <button type="submit">Send</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    local_css("style/style.css")

# -------------------------------- CSS -------------------------------------------------





