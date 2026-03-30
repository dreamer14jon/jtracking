"""
Interactive Stock Analysis Dashboard
This Streamlit app allows users to analyze a stock's technical indicators and
visualize historical price data. The dashboard fetches data from Yahoo Finance
using the `yfinance` library, calculates key metrics (RSI, simple moving
averages, exponential moving averages and MACD) and displays them in
interactive charts powered by Plotly. It also classifies the current market
condition of the selected stock as Bullish, Bearish or Oversold/Overbought
based on the computed indicators.

The app is designed for educational purposes and should not be used to make
actual trading decisions. Real-time data retrieval requires internet access and
the yfinance library installed in your Python environment.

How to run:

  1. Install dependencies (preferably in a virtual environment):
     pip install streamlit yfinance plotly pandas numpy

  2. Execute the app with Streamlit:
     streamlit run stock_analysis_streamlit.py

  3. A local web page will open in your default browser. Enter a stock ticker
     symbol (e.g., AAPL, TSLA, MSFT) and select the analysis period. The
     dashboard will display the stock's price chart with moving averages,
     RSI indicator and MACD along with a summary of the current market
     condition.

This script references educational insights about building interactive stock
analysis dashboards and the importance of using technical indicators.
Articles such as "Building an Automated Technical Stock Analysis Dashboard
with Python" highlight that using Streamlit and yfinance enables custom
dashboards where users can select tickers, choose between short- or
long-term moving averages, and toggle the RSI indicator to detect market
momentum【917613252318592†L62-L82】. They also emphasize that interactive
charts powered by Plotly allow users to zoom and explore data【917613252318592†L111-L116】.

Author: ChatGPT (2026)
"""

import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical daily price data for a given ticker using yfinance.

    Args:
        ticker: Stock symbol to download (e.g., "AAPL").
        period: The period for which to fetch data (e.g., "1y", "2y").

    Returns:
        A DataFrame containing the open, high, low, close and volume columns.
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        df = df.rename(columns=str.lower)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators: SMA50, SMA200, EMA12, EMA26, MACD, Signal and RSI.

    Args:
        df: DataFrame with columns 'close' and optionally 'adj close'.

    Returns:
        DataFrame with additional columns for indicators.
    """
    if df.empty:
        return df

    # Simple moving averages
    df["sma50"] = df["close"].rolling(window=50).mean()
    df["sma200"] = df["close"].rolling(window=200).mean()

    # Exponential moving averages for MACD
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI calculation
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def classify_stock(df: pd.DataFrame) -> str:
    """Classify the stock condition as Bullish, Bearish, Oversold or Overbought.

    The classification uses the latest RSI value and the relationship between
    50- and 200-day moving averages to determine momentum and trend.

    Returns a descriptive string summarizing the condition.
    """
    if df.empty or df["rsi"].isnull().all():
        return "Not enough data to classify"

    latest_rsi = df["rsi"].iloc[-1]
    latest_sma50 = df["sma50"].iloc[-1]
    latest_sma200 = df["sma200"].iloc[-1]

    status = []
    if latest_rsi < 30:
        status.append("Oversold")
    elif latest_rsi > 70:
        status.append("Overbought")

    if pd.notnull(latest_sma50) and pd.notnull(latest_sma200):
        if latest_sma50 > latest_sma200:
            status.append("Bullish")
        elif latest_sma50 < latest_sma200:
            status.append("Bearish")

    return ", ".join(status) if status else "Neutral"


def plot_price_and_ma(df: pd.DataFrame, ticker: str):
    """Create a candlestick chart with 50- and 200-day simple moving averages."""
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )
    )
    fig.add_trace(go.Scatter(x=df['date'], y=df['sma50'], line=dict(color='blue', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['sma200'], line=dict(color='orange', width=1), name='SMA 200'))
    fig.update_layout(
        title=f'{ticker.upper()} Price & Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        margin=dict(l=40, r=10, t=40, b=40)
    )
    return fig


def plot_rsi(df: pd.DataFrame, ticker: str):
    """Plot the RSI indicator for the given DataFrame."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], line=dict(color='purple', width=1), name='RSI (14)'))
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), name='Overbought')
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), name='Oversold')
    fig.update_layout(
        title=f'{ticker.upper()} RSI',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=40, r=10, t=40, b=40)
    )
    return fig


def plot_macd(df: pd.DataFrame, ticker: str):
    """Plot the MACD indicator and its signal line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], line=dict(color='black', width=1), name='MACD'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['signal'], line=dict(color='red', width=1), name='Signal'))
    fig.update_layout(
        title=f'{ticker.upper()} MACD',
        xaxis_title='Date',
        yaxis_title='MACD',
        height=300,
        margin=dict(l=40, r=10, t=40, b=40)
    )
    return fig


def main():
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("📈 Interactive Stock Analysis Dashboard")

    st.markdown(
        """
        This dashboard fetches historical price data, computes technical indicators
        and visualizes them using interactive charts. Use it to explore how
        indicators like the RSI and moving averages behave for different stocks.
        **Note:** This tool is for educational purposes and should not be used
        for actual trading decisions. Consult a trusted adult or financial
        professional before investing.
        """
    )

    ticker = st.text_input("Enter a stock symbol", value="AAPL", max_chars=10)
    period = st.selectbox("Select period", options=["6mo", "1y", "2y", "5y"], index=1)

    if ticker:
        with st.spinner("Fetching data..."):
            data = fetch_data(ticker, period)
        if not data.empty:
            data = calculate_indicators(data)
            condition = classify_stock(data)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_price_and_ma(data, ticker), use_container_width=True)
                st.plotly_chart(plot_macd(data, ticker), use_container_width=True)
            with col2:
                st.plotly_chart(plot_rsi(data, ticker), use_container_width=True)
                st.subheader("Current Market Condition")
                st.write(condition)
                st.dataframe(data[["date", "close", "sma50", "sma200", "rsi", "macd", "signal"]].tail(10))
        else:
            st.warning("No data available for the given ticker.")


if __name__ == "__main__":
    main()