import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.trend import ADXIndicator
from transformers import pipeline
from datetime import datetime
from scipy.stats import norm
import math

st.set_page_config(page_title="📈 Live Price Tracker", layout="wide")

# Enhanced Styling with Theme Toggle
st.sidebar.title("Settings")
theme = st.sidebar.radio("Select Theme:", ('Light', 'Dark'))

def apply_theme(theme):
    if theme == 'Dark':
        st.markdown('<style>body {background-color: #121212; color: white;}</style>', unsafe_allow_html=True)
    else:
        st.markdown('<style>body {background-color: #f0f2f6; color: #333;}</style>', unsafe_allow_html=True)

apply_theme(theme)


# Sentiment Analysis
sentiment_analyzer = pipeline("sentiment-analysis")

def fetch_news(stock):
    news = [
        f"{stock} sees significant upward movement.",
        f"{stock} faces pressure due to global market conditions.",
        f"{stock} achieves a new milestone in quarterly results."
    ]
    sentiments = [sentiment_analyzer(article)[0] for article in news]
    return pd.DataFrame({"News": news, "Sentiment": [s['label'] for s in sentiments], "Score": [s['score'] for s in sentiments]})


def plot_sentiment_analysis(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['News'], y=df['Score'], marker_color=np.where(df['Sentiment'] == 'POSITIVE', 'green', 'red')))
    fig.update_layout(title="Sentiment Analysis", xaxis_title="News Headlines", yaxis_title="Sentiment Score")
    st.plotly_chart(fig)


def technical_indicators(data):
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['SMA'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA'] = EMAIndicator(data['Close'], window=20).ema_indicator()
    data['ADX'] = ADXIndicator(data['High'], data['Low'], data['Close']).adx()

    macd = MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['Signal Line'] = macd.macd_signal()

    bb = BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    return data


def plot_indicators(data, stock):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='SMA (20)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], name='EMA (20)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name='Bollinger High', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name='Bollinger Low', line=dict(dash='dash')))

    fig.update_layout(title=f"Technical Indicators for {stock}", template="plotly_dark" if theme == 'Dark' else "plotly_white")
    st.plotly_chart(fig)

    st.write("RSI")
    st.line_chart(data['RSI'])

    st.write("MACD & Signal Line")
    st.line_chart(data[['MACD', 'Signal Line']])

    st.write("ADX")
    st.line_chart(data['ADX'])


def demand_supply_analysis(stock):
    data = yf.download(stock, period="1mo", interval="1d")
    data = technical_indicators(data)

    volume = data['Volume']
    open_interest = np.random.randint(1000, 5000, len(volume))  # Mock Open Interest Data

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=volume.index, y=volume.values, mode='lines', name="Volume"))
    fig.add_trace(go.Scatter(x=volume.index, y=open_interest, mode='lines', name="Open Interest", line=dict(dash='dash')))
    fig.update_layout(title="Demand-Supply Analysis", xaxis_title="Date", yaxis_title="Volume / Open Interest", template="plotly_dark" if theme == 'Dark' else "plotly_white")
    st.plotly_chart(fig)

    plot_indicators(data, stock)


def fetch_tweets(stock):
    tweets = [
        f"{stock} is looking bullish today!",  
        f"{stock} is facing some tough resistance.",
        f"I'm buying more of {stock} on this dip."
    ]
    sentiments = [sentiment_analyzer(tweet)[0] for tweet in tweets]
    return pd.DataFrame({"Tweet": tweets, "Sentiment": [s['label'] for s in sentiments], "Score": [s['score'] for s in sentiments]})


def plot_tweets_analysis(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Tweet'], y=df['Score'], marker_color=np.where(df['Sentiment'] == 'POSITIVE', 'green', 'red')))
    fig.update_layout(title="Tweets Sentiment Analysis", xaxis_title="Tweets", yaxis_title="Sentiment Score")
    st.plotly_chart(fig)


st.sidebar.title("📅 Date Range Selection")
selected_stocks = st.sidebar.multiselect("📊 Select Stocks:", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"], default=["RELIANCE.NS"])

for stock in selected_stocks:
    st.subheader(f"Sentiment Analysis for {stock}")
    news_df = fetch_news(stock)
    st.dataframe(news_df)
    plot_sentiment_analysis(news_df)

    st.subheader(f"Live Tweets Analysis for {stock}")
    tweets_df = fetch_tweets(stock)
    st.dataframe(tweets_df)
    plot_tweets_analysis(tweets_df)

    st.subheader(f"Demand-Supply Analysis & Technical Indicators for {stock}")
    demand_supply_analysis(stock)
