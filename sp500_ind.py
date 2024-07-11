import streamlit as st
import pandas as pd
#from pandasai  import PandasAI
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
import pickle
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import Orthogonal
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import requests
from pandas.tseries.offsets import DateOffset
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
from st_aggrid import AgGrid, GridOptionsBuilder,ColumnsAutoSizeMode
from st_aggrid.shared import GridUpdateMode
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder


st.set_page_config(page_title="Stocks dashboard and Prediction", page_icon="💹", layout="wide")
pio.templates.default = "plotly_white"

# Define custom objects if needed
custom_objects = {'Orthogonal': Orthogonal}

import streamlit.components.v1 as components
import base64

# Path to the local audio file
audio_file_path = "data/audio.mp3"  
icon_play_path = "data/icon.jpg"  
icon_pause_path = "data/icon.jpg"  

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
        if isinstance(data.index, pd.DatetimeIndex):
            data.index = data.index.tz_localize(None)  # Check index type before localizing
        data.reset_index(inplace=True)
        stock_info = yf.Ticker(stock).info
        market_cap = stock_info.get('marketCap', np.nan)
        data['Market Cap'] = market_cap
        data['Ticker'] = stock
        stock_data[stock] = data
    return stock_data

def plot_sparkline(open_prices, close_prices):
    # Determine color based on the last open and close prices
    if close_prices[-1] > open_prices[0]:
        line_color = 'green'
        fill_color = 'lightgreen'
    else:
        line_color = 'red'
        fill_color = 'pink'

    fig_spark = go.Figure(
        data=go.Scatter(
            y=close_prices,
            mode="lines",
            fill="tozeroy",
            line_color=line_color,
            fillcolor=fill_color,
        ),
    )
    fig_spark.update_traces(hovertemplate="Price: $ %{y:.2f}")
    fig_spark.update_xaxes(visible=False, fixedrange=True)
    fig_spark.update_yaxes(visible=False, fixedrange=True)
    fig_spark.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        height=50,
        width=50,  # Add this line to set a fixed width
        margin=dict(t=0, l=0, b=0, r=0, pad=10),
    )
    return fig_spark


def display_watchlist_card(ticker, symbol_name, last_price, change_pct, open_prices, close_prices):
    with st.container():
        tl, tr = st.columns([1, 0.5])  # Adjust the column ratios to reduce space
        bl, br = st.columns([1, 1])

        with tl:
            st.markdown(f"**{symbol_name}**")

        with tr:
            st.markdown(f"**{ticker}**")
            negative_gradient = float(change_pct) < 0.00
            st.markdown(
                f"<span style='color: {'red' if negative_gradient else 'green'}'>{'▼' if negative_gradient else '▲'} {change_pct:.2f} %</span>",
                unsafe_allow_html=True,
            )

        with bl:
            st.markdown(f"**Current Value**")
            st.markdown(f"${last_price:.2f}")

        with br:
            fig_spark = plot_sparkline(open_prices, close_prices)
            st.plotly_chart(fig_spark, config=dict(displayModeBar=False), use_container_width=True)

def display_watchlist(ticker_df):
    n_cols = 5
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
                    ticker['Open Prices'],
                    ticker['Close Prices']  # Pass Close Prices to the function
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
            x=history_df['Date'],
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
        go.Bar(x=history_df['Date'], y=history_df['Volume'], name="Volume Traded", marker_color='blue'),
        row=2,
        col=1,
    )
    f_candle.update_layout(
        title="Stock Price Trends",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        yaxis1=dict(
            title="OHLC",
            titlefont=dict(size=14, color='black', family='Arial', weight='bold'),
            tickfont=dict(size=12, color='black', family='Arial', weight='bold')
        ),
        yaxis2=dict(
            title="Volume",
            titlefont=dict(size=14, color='black', family='Arial', weight='bold'),
            tickfont=dict(size=12, color='black', family='Arial', weight='bold')
        ),
        xaxis=dict(
            title="Date",
            titlefont=dict(size=14, color='black', family='Arial', weight='bold'),
            tickfont=dict(size=12, color='black', family='Arial', weight='bold')
        ),
        hovermode="x",
        font=dict(size=16, color='black', family='Arial', weight='bold'),
    )
    f_candle.update_layout(
        title_font_family="Open Sans",
        title_font_color="#174C4F",
        title_font_size=32,
        font_size=16,
        margin=dict(l=80, r=80, t=100, b=80, pad=0),
        height=500,
    )
    return f_candle

def format_value(value):
    if value >= 1e9:
        return f'{value / 1e9:.2f}Bn $S'
    elif value >= 1e6:
        return f'{value / 1e6:.2f}Mn $'
    else:
        return f'{value:,.2f}'
        

def display_symbol_history(ticker_df, history_dfs):
    col1, col2 = st.columns(2)

    with col1:
        selected_ticker = st.selectbox("📰 Currently Showing", list(history_dfs.keys()), key="selectbox_symbol_history")
    
    with col2:
        selected_period = st.selectbox("⌚ Period", ("Week", "Month", "Trimester", "Year"), 3, key="selectbox_period")


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
            st.metric("Lowest Volume Day Trade", format_value(history_df["Volume"].min()))
            st.metric("Lowest Close Price", f'{history_df["Close"].min():,.2f} $')
        with r:
            st.metric("Highest Volume Day Trade", format_value(history_df["Volume"].max()))
            st.metric("Highest Close Price", f'{history_df["Close"].max():,.2f} $')

        st.metric("Average Daily Volume", format_value(history_df["Volume"].mean()))
        st.metric(
            "Current Market Cap",
            format_value(ticker_df[ticker_df["Ticker"] == selected_ticker]["Market Cap"].values[0])
            if "Market Cap" in ticker_df.columns else "N/A"
        )


def display_overview(ticker_df):
    def format_currency(val):
        return "$ {:,.2f}".format(val)

    def format_percentage(val):
        return "{:,.2f} %".format(val)

    def format_decimal(val):
            return "{:,.2f}".format(val)
    def format_change(val):
        return "color: red;" if (val < 0.00) else "color: green;"

    def apply_odd_row_class(row):
        return ["background-color: #f8f8f8" if row.name % 2 != 0 else "" for _ in row]

    ticker_df['Open Price'] = ticker_df['Open Prices'].apply(lambda x: x[0])
    ticker_df['Close Price'] = ticker_df['Close Prices'].apply(lambda x: x[-1])

    with st.expander("📊 Stocks Preview"):
        styled_dataframe = (
            ticker_df.style.format(
                {
                    "Last Price": format_currency,
                    "Change Pct": format_percentage,
                    "Open Price": format_currency,
                    "Close Price": format_currency,
                    "Market Cap" : format_decimal
                }
            )
            .apply(apply_odd_row_class, axis=1)
            .applymap(format_change, subset=["Change Pct"])
        )

        st.dataframe(
            styled_dataframe,
            column_order=["Ticker", "Symbol Name", "Last Price", "Change Pct", "Open Price", "Close Price", "Market Cap"],
            hide_index=True,
            height=250,
            use_container_width=True,
        )

@st.cache_data
def transform_data(stock_data):
    ticker_df_list = []

    for ticker, df in stock_data.items():
        last_price = df['Close'].iloc[-1]
        change_pct = ((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100
        symbol_name = yf.Ticker(ticker).info.get('shortName', ticker)
        market_cap = df['Market Cap'].iloc[0] if 'Market Cap' in df else np.nan
        open_prices = df['Open'].tolist()
        close_prices = df['Close'].tolist()
        
        ticker_df_list.append({
            "Ticker": ticker,
            "Symbol Name": symbol_name,
            "Last Price": last_price,
            "Change Pct": change_pct,
            "Open Prices": open_prices,
            "Close Prices": close_prices,
            "Market Cap": market_cap
        })

    ticker_df = pd.DataFrame(ticker_df_list)
    return ticker_df, stock_data
# List of stock symbols
stock_symbols = ["AAPL", "AMZN", "GOOGL", "MSFT"]

# Metrics to be extracted and renamed
metrics_map = {
    '52WeekPriceReturnDaily': '1Y Return',
    'totalDebt/totalEquityAnnual': 'Debt/Equity',
    'dividendGrowthRate5Y': '5Y Dividend Growth',
    'netProfitMarginAnnual': 'Net Profit Margin',
    'epsAnnual': '12M EPS Growth',
    'epsGrowth5Y': '5Y EPS Growth',
    'pfcfShareAnnual': 'FCF',
    'currentDividendYieldTTM': 'Dividend Yield',
    'assetTurnoverAnnual': 'Return on Asset',
    'peAnnual': 'P/E'
}

# Function to load JSON data and extract required metrics
def load_and_extract_metrics(symbol):
    with open(f'data/{symbol}.json', 'r') as file:
        data = json.load(file)
    
    metrics = data['metric']
    extracted_data = {metrics_map[key]: metrics.get(key, None) for key in metrics_map.keys()}
    return extracted_data


# Function to replace NaN values with the mean
def replace_nan_with_mean(df):
    return df.apply(lambda x: x.fillna(x.mean()), axis=0)

# normalization
def normalize_values(df):
    norm_df = (df - df.min()) / (df.max() - df.min())
    # Ensure that zero normalization is avoided for non-zero values
    norm_df.replace(0, norm_df[norm_df > 0].min().min(), inplace=True)
    return norm_df

# Updated plot_polar function
def plot_polar(data):
    polar_data = []
    for metric, value in data.items():
        if pd.isna(value):
            value = 0  # Handle missing values by setting them to 0
        normalized_value = min(value, 1.0)  # Ensure the value does not exceed 1
        percentiles = ["0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                       "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
        percentiles.reverse()
        
        for _ in range(10):  # Limit iterations to the length of percentiles list
            if normalized_value <= 0:
                break
            current_value = min(normalized_value, 0.1)
            polar_data.append([metric, percentiles.pop(), current_value])
            normalized_value -= current_value
    
    polar_df = pd.DataFrame(polar_data, columns=['metric', 'percentile', 'value'])
    fig = px.bar_polar(polar_df, r='value', theta='metric', color='percentile', template='plotly_dark',
                       color_discrete_sequence=px.colors.sequential.Plasma_r)
    
    
    # Update layout for enhanced readability
    fig.update_layout(
      
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=14, color='white', family='Arial')  # Set font color to black
            ),
            angularaxis=dict(
                tickfont=dict(size=28, color='black', family='Arial'),  # Set font color to black
                layer='below traces'
            )
        ),
        legend=dict(
            font=dict(size=14, color='black', family='Arial'),  # Set font color to black
        ),

    )
    return fig

# Insights page function
def insights_page():
    st.markdown('<h2> 💰Insights into the stock metrics for AAPL, AMZN, GOOGL, and MSFT:</h2>', unsafe_allow_html=True)
    
    # Radio button for stock selection
    selected_symbol = st.radio("Select a stock", stock_symbols)
    
    all_data = {}
    for symbol in stock_symbols:
        extracted_data = load_and_extract_metrics(symbol)
        all_data[symbol] = extracted_data
    
    df = pd.DataFrame(all_data).T  # Ensure DataFrame creation is outside the loop
    df = replace_nan_with_mean(df)
    df_normalized = normalize_values(df)
    st.write(all_data)
    st.write(df)
    #st.write(df_normalized)

    if selected_symbol:
        st.subheader(f"{selected_symbol} Stock Metrics Relative to S&P500 Companies plotted in Pie Chart: ")
        fig = plot_polar(df.loc[selected_symbol].to_dict())
        st.plotly_chart(fig)

def prediction_model_page():
    import os
    import pickle
    import numpy as np
    from tensorflow.keras.models import load_model

    # We will download our data from Yahoo Finance URL
    stock_url = "https://query1.finance.yahoo.com/v7/finance/download/{}"


    stock_symbols = ['AMZN', 'MSFT', 'GOOGL', 'AAPL', '^GSPC']    
    # Radio button for stock selection
    selected_stock = st.radio("Select a stock", stock_symbols)
    
    if selected_stock:
        st.subheader(f"Prediction model for {selected_stock}")
        # Add the existing prediction model code here, using the selected_stock
        try:
            stock_data = download_stock_data([selected_stock])
            st.write(f"{selected_stock} data:")
            st.write(stock_data[selected_stock].tail(10))
            
            # Load the scalers
            scaler_x_path = f'scaler_x_{selected_stock}.sav'
            scaler_y_path = f'scaler_y_{selected_stock}.sav'

            if not os.path.isfile(scaler_x_path) or not os.path.isfile(scaler_y_path):
                raise FileNotFoundError(f"Scalers for {selected_stock} not found. Expected paths: {scaler_x_path}, {scaler_y_path}")

            scaler_x = pickle.load(open(scaler_x_path, 'rb'))
            scaler_y = pickle.load(open(scaler_y_path, 'rb'))

            # Load the model
            model_path = f'stock_prediction_{selected_stock}.h5'
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model for {selected_stock} not found. Expected path: {model_path}")

            model = load_model(model_path, custom_objects=custom_objects)
            st.write(f"Model for {selected_stock} loaded successfully!")

            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
            # Prepare data for prediction
            X = stock_data[selected_stock][['Open', 'High', 'Low', 'Close', 'Volume']].tail(10).values
            X_scaled = scaler_x.transform(np.array(X))
            y_predict = model.predict(np.array([X_scaled]))
            return_pred = scaler_y.inverse_transform(y_predict)

            # Price tomorrow = Price today * (Return + 1)
            X_up_scaled = scaler_x.inverse_transform(np.array(X_scaled))
            pred_price = X_up_scaled[-1][0] * (return_pred[0] + 1)

            st.write(f"Next day prediction for {selected_stock} is ", pred_price)

        except FileNotFoundError as fnf_error:
            st.error(f"File error for {selected_stock}: {fnf_error}")
        except Exception as stock_error:
            st.error(f"An error occurred while predicting for {selected_stock}: {stock_error}")

import pandas as pd
import requests
from pandas.tseries.offsets import DateOffset

def query_news(symbol, alphavantage_apikey):
    date = pd.Timestamp.today() - DateOffset(days=10)
    date = date.strftime('%Y%m%d')

    if symbol is None:
        url = 'https://www.alphavantage.co/query' \
              '?function=NEWS_SENTIMENT' \
              '&sort=RELEVANCE' \
              '&time_from=%sT0000' \
              '&topics=financial_markets' \
              '&limit=20' \
              '&apikey=%s' % (date, alphavantage_apikey)
    else:
        url = 'https://www.alphavantage.co/query' \
              '?function=NEWS_SENTIMENT' \
              '&sort=RELEVANCE' \
              '&time_from=%sT0000' \
              '&limit=20' \
              '&tickers=%s' \
              '&apikey=%s' % (date, symbol, alphavantage_apikey)

    if alphavantage_apikey == "demo":
        url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
        symbol = "AAPL"

    r = requests.get(url)
    data = r.json()

    if 'feed' in data:
        df = pd.DataFrame(data['feed'])
        df = df.drop_duplicates(subset=['title'])

        def sentiment_filter(sentiments):
            return [x for x in sentiments if x['ticker'] == symbol][0]

        def split_relevance(x):
            return x['relevance_score']

        def split_sentiment(x):
            return x['ticker_sentiment_score']

        if symbol:
            df["ticker_sentiment"] = df["ticker_sentiment"].apply(sentiment_filter)
            df["relevance"] = df["ticker_sentiment"].apply(split_relevance)
            df["sentiment"] = df["ticker_sentiment"].apply(split_sentiment)

            del df['ticker_sentiment']
        else:
            df["relevance"] = 0.0
            df["sentiment"] = 0.0

        news_df = df[['title', 'summary', 'url', 'relevance', 'sentiment']]
        return news_df
    elif 'Information' in data:
        st.write("API response information:", data['Information'])
        raise ValueError("API response contains an information message.")
    else:
        st.write("API response keys:", data.keys())
        raise ValueError("API response does not contain 'feed' key.")

# Helper functions for the chatbot
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text):
        return self.model.encode([text])[0]

    def __call__(self, texts):
        return self.embed_documents(texts)

def df_loader(news_df):
    loader = DataFrameLoader(
        news_df,
        page_content_column="title"
    )

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()

    news_tool = create_retriever_tool(
        retriever,
        "search_stock_news",
        "Searches and returns news dataframe.",
    )

    return news_tool

# Updated build_agent function
def build_agent(news_df, openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print(f"API Key being used: {openai_api_key}")

    try:
        news_tool = df_loader(news_df)
        tools = [news_tool]

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=False)

        return agent_executor

    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        raise e

# StockSaavy Page with Chatbot Integration
def stocksavvy_page():
    st.markdown('<h1 style="color: #FFD700;">🤖 StockSaavy - Your Virtual Assistant</h1>', unsafe_allow_html=True)
    
    # Option to enter OpenAI API key
    st.session_state.api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="file_uploader_stocksaavy")
    
    if uploaded_file is not None:
        try:
            news_df = pd.read_csv(uploaded_file)
            if news_df.empty:
                st.warning("The uploaded CSV file is empty. Please upload a valid file.")
            else:
                st.success("CSV file successfully uploaded and read!")
                st.dataframe(news_df)
                if 'agent_executor' not in st.session_state:
                    st.session_state.agent_executor = build_agent(news_df, st.session_state.api_key)
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a valid file.")
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")

    # Create two columns for positioning the "Start New Session" button to the top right corner
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Option to start a new session
        if st.button("Start New Session"):
            st.session_state.uploaded_file = None
            st.session_state.chat_history = []
            st.session_state.api_key = ""
            st.session_state.agent_executor = None
            st.experimental_rerun()
    
    # User input for asking questions
    user_input = st.text_input("Ask questions out to StockSaavy e.g. analyze uploaded file, provide insights, draw charts, search in web, etc.", key="input_stocksaavy_unique")
    
    if user_input:
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history.append(("Question", user_input))
        
        try:
            # Ensure the agent executor is defined and ready to use
            if 'agent_executor' in st.session_state and st.session_state.agent_executor:
                response = st.session_state.agent_executor({"input": user_input})
                st.session_state.chat_history.append(("Answer", response["output"]))
            else:
                st.session_state.chat_history.append(("Error", "The agent executor is not ready. Please upload a CSV file and ensure the API key is entered."))
        except Exception as e:
            st.session_state.chat_history.append(("Error", f"An error occurred while processing your request: {e}"))
        
        for i, (role, message) in enumerate(st.session_state.chat_history):
            st.markdown(f"**{role}:** {message}")



def news_sentiment_page():
    st.subheader("News and Sentiment Analysis")
    stock_symbol_news = ['AAPL', 'AMZN', 'GOOG', 'MSFT']
    selected_symbol = st.radio("Select a stock", stock_symbol_news)

    #alphavantage_apikey = 'AFWMK3BFG40RU427'
    alphavantage_apikey = 'NO6ECJF05GQZXCTJ'

    
    #try:
    news_df = query_news(selected_symbol, alphavantage_apikey)
    
    news_df['relevance'] = pd.to_numeric(news_df['relevance'], errors='coerce')
    news_df = news_df.dropna(subset=['relevance'])
    
    # Display news data in a table using AgGrid
    gb = GridOptionsBuilder.from_dataframe(news_df)
    gb.configure_pagination()
    #gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
    gb.configure_column("title", header_name="Title", width=500, editable=True, cellStyle={'font-weight': 'bold', 'font-size': '16px'})
    gb.configure_column("summary", header_name="Summary", width=400, editable=True)
    gb.configure_column("url", header_name="URL", width=100, #cellRenderer=JsCode('''function(params) {return '<a href="' + params.value + '/view" target="_blank">' + params.value + '</a>'}''')
     editable=True)

    gb.configure_column("relevance", header_name="Relevance", width=100)
    gb.configure_column("sentiment", header_name="Sentiment", width=100)
    gb.configure_default_column(filterable=True)  # Make all columns filterable
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children")
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    
    # Configure DataFrame with clickable URLs
    st.data_editor(
        news_df,
        column_config={
            "title": st.column_config.Column("Title", help="The title of the news article"),
            "summary": st.column_config.Column("Summary", help="Summary of the news article"),
            "url": st.column_config.LinkColumn(
                "URL",
                display_text="Open link",
                help="Link to the news article",
            ),
            "relevance": st.column_config.Column("Relevance", help="Relevance score"),
            "sentiment": st.column_config.Column("Sentiment", help="Sentiment score"),
        },
        hide_index=True,
    )

    response = AgGrid(
        news_df,
        gridOptions=gridOptions,
        height=1600,
        width='100%',
        theme='alpine',
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED',
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
    )

    # Add a download button
    st.download_button(
        label="Download data as CSV",
        data=news_df.to_csv(index=False).encode('utf-8'),
        file_name='news_data.csv',
        mime='text/csv',
    )



def main():
    
    #st.title("Stock Price Dashboard and Prediction")
    
    menu = ['Home', 'Dashboard and Insights', 'News and Sentiment', 'Prediction model', 'StockSaavy']
    choice = st.sidebar.selectbox("Menu", menu, key="main_menu")
    
    if choice == 'Home':
        st.markdown('<h1>📊 Stock Price Dashboard and Prediction 💹 💵</h1>', unsafe_allow_html=True)
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

        # Add the audio player only on the Home page
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
        
        st.subheader("StockSavvy Services")

        st.markdown("""
        **Dashboard and Insights:** Access combined stock watchlists, historical data, and key insights.
        - Stakeholders: Investors, Traders, Analysts.
        
        **News and Sentiment Analysis:** Get the latest news and sentiment analysis for your favorite stocks.
        - Stakeholders: Investors, Market Analysts.
        
        **Prediction Model:** Predict next morning's opening prices using recent stock data.
        - Stakeholders: Analysts, Traders, Investors.
        
        **StockSaavy - Your Virtual Assistant:** 24/7 assistant for stock market queries, insights, and analysis.
        - Stakeholders: General Public, Advisors, Investors.
        """)
        st.markdown('<h5>**Disclaimer:** This prediction model should not be the sole basis for investment decisions. Conduct your own research before investing.</h5>',unsafe_allow_html=True)
    elif choice == 'Prediction model':
        st.markdown('<h2>🔮 Prediction Model</h2>', unsafe_allow_html=True)
    elif choice == 'Dashboard':
        st.markdown('<h2>📈 Dashboard</h2>', unsafe_allow_html=True)
        st.markdown('<h3>Line chart showing growth over last 12 months  and Current stock price</h2>', unsafe_allow_html=True)
    elif choice == 'Insights':
        pass
    elif choice == 'News and Sentiment':
        news_sentiment_page()
    elif choice == 'StockSaavy':
        stocksavvy_page()  
    elif choice == 'Dashboard and Insights':
        st.markdown('<h2>📈 Dashboard</h2>', unsafe_allow_html=True)
        st.markdown('<h3>Line chart showing growth over last 12 months  and Current stock price</h2>', unsafe_allow_html=True)
        # Example tickers to display
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "^GSPC"]
        stock_data = download_stock_data(tickers)
        ticker_df, history_dfs = transform_data(stock_data)   
        display_watchlist(ticker_df)
        st.divider()
        display_symbol_history(ticker_df, history_dfs)
        display_overview(ticker_df)
        insights_page()
    return choice
    

if __name__ == '__main__':
    choice = main()

if choice == 'Home':
    pass
elif choice == 'Prediction model':
    prediction_model_page()
#elif choice == 'Dashboard':
#    # Example tickers to display
#    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "^GSPC"]
#    stock_data = download_stock_data(tickers)
#    ticker_df, history_dfs = transform_data(stock_data)
#
#    display_watchlist(ticker_df)
#    st.divider()
#    display_symbol_history(ticker_df, history_dfs)
#    display_overview(ticker_df)

#elif choice == "Insights":
#    insights_page()