import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import Orthogonal
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio



st.set_page_config(page_title="Stocks dashboard and Prediction", page_icon="💹", layout="wide")
pio.templates.default = "plotly_white"

# Define custom objects if needed
custom_objects = {'Orthogonal': Orthogonal}

import streamlit.components.v1 as components
import base64

# Path to the local audio file
audio_file_path = "data/audio.mp3"  #
icon_play_path = "data/icon.jpg"  # 
icon_pause_path = "data/icon.jpg"  # 
# Function to read the audio file and encode it to base64
def get_base64_audio(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode()

# Function to read the icon file and encode it to base64
def get_base64_icon(file_path):
    with open(file_path, "rb") as icon_file:
        return base64.b64encode(icon_file.read()).decode()

# Encode the audio and icon files to base64
audio_base64 = get_base64_audio(audio_file_path)
icon_play_base64 = get_base64_icon(icon_play_path)
icon_pause_base64 = get_base64_icon(icon_pause_path)

# HTML and JavaScript to embed the audio player with autoplay and a toggle play/pause button
audio_html = f"""
<audio id="background-audio" loop>
    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    Your browser does not support the audio element.
</audio>
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var audio = document.getElementById('background-audio');
    var playButton = document.getElementById('play-button');
    var isPlaying = false;

    function togglePlayPause() {{
        if (isPlaying) {{
            audio.pause();
            playButton.src = "data:image/png;base64,{icon_play_base64}";
        }} else {{
            audio.play();
            playButton.src = "data:image/png;base64,{icon_pause_base64}";
        }}
        isPlaying = !isPlaying;
    }}

    playButton.addEventListener('click', togglePlayPause);

    var playPromise = audio.play();
    if (playPromise !== undefined) {{
        playPromise.then(_ => {{
            // Automatic playback started!
            playButton.src = "data:image/png;base64,{icon_pause_base64}";
            isPlaying = true;
        }}).catch(error => {{
            // Auto-play was prevented
            console.log('Autoplay prevented by browser: ', error);
            // Show the play button
            playButton.style.display = 'block';
        }});
    }}
}});
</script>
<img id="play-button" src="data:image/png;base64,{icon_play_base64}" style="display: none; cursor: pointer;" alt="Play Music">
"""

components.html(audio_html, height=150)


@st.cache_data 
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("data/image.jpg")

page_bg_img = f"""
<style>


[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}


</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_data
def download_stock_data(stock_list):
    stock_data = {}
    for stock in stock_list:
        data = yf.download(stock, period="1y", interval="1d")
        # Check if the index is a DatetimeIndex before localizing timezone
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_localize(None)  # Modified: Check index type before localizing
        data.reset_index(inplace=True)
        stock_info = yf.Ticker(stock).info
        market_cap = stock_info.get('marketCap', np.nan)
        data['Market Cap'] = market_cap
        stock_data[stock] = data
    return stock_data

def plot_sparkline(data):
    fig_spark = go.Figure(
        data=go.Scatter(
            y=data,
            mode="lines",
            fill="tozeroy",
            line_color="red",
            fillcolor="pink",
        ),
    )
    fig_spark.update_traces(hovertemplate="Price: $ %{y:.2f}")
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=False, fixedrange=True)
    fig_spark.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=50,
        margin=dict(t=10, l=0, b=0, r=0, pad=0),
    )
    return fig_spark

def display_watchlist_card(ticker, symbol_name, last_price, change_pct, open_prices):
    with st.container():
        tl, tr = st.columns([2, 1])
        bl, br = st.columns([1, 1])

        with tl:
            st.markdown(f"**{symbol_name}**")

        with tr:
            st.markdown(f"**{ticker}**")
            negative_gradient = float(change_pct) < 0
            st.markdown(
                f"<span style='color: {'red' if negative_gradient else 'green'}'>{'▼' if negative_gradient else '▲'} {change_pct:.2f} %</span>",
                unsafe_allow_html=True,
            )

        with bl:
            st.markdown(f"**Current Value**")
            st.markdown(f"${last_price:.2f}")

        with br:
            fig_spark = plot_sparkline(open_prices)
            st.plotly_chart(fig_spark, config=dict(displayModeBar=False), use_container_width=True)

def display_watchlist(ticker_df):
    n_cols = 4
    for i in range(0, len(ticker_df), n_cols):
        row = ticker_df.iloc[i:i + n_cols]
        cols = st.columns(len(row))
        for idx, col in enumerate(cols):
            ticker = row.iloc[idx]
            with col:
                display_watchlist_card(
                    ticker['Ticker'],
                    ticker['Symbol Name'],
                    ticker['Last Price'],
                    ticker['Change Pct'],
                    ticker['Open Prices']
                )

def plot_candlestick(history_df):
    f_candle = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
    )

    f_candle.add_trace(
        go.Candlestick(
            x=history_df['Date'],  # Ensure 'Date' column is present
            open=history_df['Open'],
            high=history_df['High'],
            low=history_df['Low'],
            close=history_df['Close'],
            name="Dollars",
        ),
        row=1,
        col=1,
    )
    f_candle.add_trace(
        go.Bar(x=history_df['Date'], y=history_df['Volume'], name="Volume Traded"),
        row=2,
        col=1,
    )
    f_candle.update_layout(
        title="Stock Price Trends",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        yaxis1=dict(title="OHLC"),
        yaxis2=dict(title="Volume"),
        hovermode="x",
    )
    f_candle.update_layout(
        title_font_family="Open Sans",
        title_font_color="#174C4F",
        title_font_size=32,
        font_size=16,
        margin=dict(l=80, r=80, t=100, b=80, pad=0),
        height=500,
    )
    f_candle.update_xaxes(title_text="Date", row=2, col=1)
    return f_candle

def display_symbol_history(ticker_df, history_dfs):
    selected_ticker = st.selectbox("📰 Currently Showing", list(history_dfs.keys()))
    selected_period = st.selectbox(
        "⌚ Period", ("Week", "Month", "Trimester", "Year"), 2
    )

    history_df = history_dfs[selected_ticker]

    mapping_period = {"Week": 7, "Month": 31, "Trimester": 90, "Year": 365}
    today = datetime.datetime.today().date()
    history_df['Date'] = pd.to_datetime(history_df['Date'], dayfirst=True)
    history_df = history_df.set_index('Date')
    history_df = history_df[
        (today - pd.Timedelta(mapping_period[selected_period], unit='d')): today
    ].reset_index()

    left_chart, right_indicator = st.columns([1.5, 1])

    f_candle = plot_candlestick(history_df)

    with left_chart:
        st.plotly_chart(f_candle, use_container_width=True)

    with right_indicator:
        st.subheader("Period Metrics")
        l, r = st.columns(2)

        with l:
            st.metric("Lowest Volume Day Trade", f'{history_df["Volume"].min():,}')
            st.metric("Lowest Close Price", f'{history_df["Close"].min():,} $')
        with r:
            st.metric("Highest Volume Day Trade", f'{history_df["Volume"].max():,}')
            st.metric("Highest Close Price", f'{history_df["Close"].max():,} $')

        st.metric("Average Daily Volume", f'{int(history_df["Volume"].mean()):,}')
        st.metric(
            "Current Market Cap",
            "{:,} $".format(
                ticker_df[ticker_df["Ticker"] == selected_ticker]["Market Cap"].values[0]
            ) if "Market Cap" in ticker_df.columns else "N/A"
        )

def display_overview(ticker_df):
    def format_currency(val):
        return "$ {:,.2f}".format(val)

    def format_percentage(val):
        return "{:,.2f} %".format(val)

    def format_change(val):
        return "color: red;" if (val < 0) else "color: green;"

    def apply_odd_row_class(row):
        return ["background-color: #f8f8f8" if row.name % 2 != 0 else "" for _ in row]

    with st.expander("📊 Stocks Preview"):
        styled_dataframe = (
            ticker_df.style.format(
                {
                    "Last Price": format_currency,
                    "Change Pct": format_percentage,
                }
            )
            .apply(apply_odd_row_class, axis=1)
            .applymap(format_change, subset=["Change Pct"])
        )

        st.dataframe(
            styled_dataframe,
            column_order=[column for column in list(ticker_df.columns)],
            hide_index=True,
            height=250,
            use_container_width=True,
        )

@st.cache_data
def transform_data(stock_data):
    ticker_df_list = []

    for ticker, df in stock_data.items():
        last_price = df['Close'].iloc[-1]
        change_pct = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        symbol_name = yf.Ticker(ticker).info.get('shortName', ticker)
        market_cap = df['Market Cap'].iloc[0] if 'Market Cap' in df else np.nan

        ticker_df_list.append({
            "Ticker": ticker,
            "Symbol Name": symbol_name,
            "Last Price": last_price,
            "Change Pct": change_pct,
            "Open Prices": df['Open'].tolist(),
            "Market Cap": market_cap
        })

    ticker_df = pd.DataFrame(ticker_df_list)
    return ticker_df, stock_data

def main():
    st.title("Stock Price Dashboard and Prediction")

    menu = ['Home', 'Prediction model', 'Dashboard']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader('We will predict next morning opening price of stocks, using previous 10 days info which include "Open", "High", "Low", "Close", "Volume". A very important point to note here is that this prediction model should not be considered as a sole basis to invest your hard-earned money. Please make your own research before investment.')
    elif choice == 'Prediction model':
        st.subheader('LSTM based Prediction model for next day stock's Price')
    elif choice == 'Dashboard':
        st.subheader('Dashboard')
    return choice

if __name__ == '__main__':
    choice = main()

if choice == 'Home':
    {}
elif choice == 'Prediction model':
    import os
    import pickle
    import numpy as np
    from tensorflow.keras.models import load_model

    # We will download our data from Yahoo Finance URL
    stock_url = "https://query1.finance.yahoo.com/v7/finance/download/{}"

    try:
        # Define stock symbols
        stock_symbols = ['AMZN', 'MSFT', 'GOOGL']

        for symbol in stock_symbols:
            try:
                # Load the stock data
                stock_data = download_stock_data([symbol])
                st.write(f"{symbol} data:")
                st.write(stock_data[symbol].tail(10))
                
                # Load the scalers
                scaler_x_path = f'scaler_x_{symbol}.sav'
                scaler_y_path = f'scaler_y_{symbol}.sav'

                if not os.path.isfile(scaler_x_path) or not os.path.isfile(scaler_y_path):
                    raise FileNotFoundError(f"Scalers for {symbol} not found. Expected paths: {scaler_x_path}, {scaler_y_path}")

                scaler_x = pickle.load(open(scaler_x_path, 'rb'))
                scaler_y = pickle.load(open(scaler_y_path, 'rb'))

                # Load the model
                model_path = f'stock_prediction_{symbol}.h5'
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(f"Model for {symbol} not found. Expected path: {model_path}")

                model = load_model(model_path, custom_objects=custom_objects)
                st.write(f"Model for {symbol} loaded successfully!")

                model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
                # Prepare data for prediction
                X = stock_data[symbol][['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).values
                X_scaled = scaler_x.transform(np.array(X))
                y_predict = model.predict(np.array([X_scaled]))
                return_pred = scaler_y.inverse_transform(y_predict)

                # Price tomorrow = Price today * (Return + 1)
                X_up_scaled = scaler_x.inverse_transform(np.array(X_scaled))
                pred_price = X_up_scaled[-1][0] * (return_pred[0] + 1)

                st.write(f"Next day prediction for {symbol} is ", pred_price)

            except FileNotFoundError as fnf_error:
                st.error(f"File error for {symbol}: {fnf_error}")
            except Exception as stock_error:
                st.error(f"An error occurred while predicting for {symbol}: {stock_error}")

    except Exception as main_error:
        st.error(f"An error occurred: {main_error}")

elif choice == 'Dashboard':
    # Example tickers to display
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "^GSPC"]
    stock_data = download_stock_data(tickers)
    ticker_df, history_dfs = transform_data(stock_data)

    display_watchlist(ticker_df)
    st.divider()
    display_symbol_history(ticker_df, history_dfs)
    display_overview(ticker_df)
