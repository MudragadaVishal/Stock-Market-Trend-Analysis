import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime
import pandas_ta as ta
from io import StringIO
import sys




STOCKS_AND_INDICES = {
    # Indian Indices
    "Nifty 50": "^NSEI", "Sensex": "^BSESN",
    # Indian Stocks
    "Reliance Industries": "RELIANCE.NS", "Tata Consultancy Services (TCS)": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS", "Infosys": "INFY.NS",
    "State Bank of India": "SBIN.NS", "Tata Motors": "TATAMOTORS.NS", "Bharti Airtel": "BHARTIARTL.NS", "ITC": "ITC.NS",
    # US Stocks
    "Apple Inc.": "AAPL", "Google (Alphabet)": "GOOGL", "Microsoft": "MSFT",
    "Amazon": "AMZN", "Tesla": "TSLA", "Meta Platforms (Facebook)": "META",
    # ETFs
    "Gold ETF (GOLDBEES)": "GOLDBEES.NS", "Nifty 50 ETF (NIFTYBEES)": "NIFTYBEES.NS", "Bank Nifty ETF (BANKBEES)": "BANKBEES.NS"
}



def get_stock_info(ticker_symbol):
    """Fetches key information and metrics for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        name = info.get('longName', ticker_symbol)
        currency = info.get('currency', 'USD')
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 'N/A'
        
        if isinstance(market_cap, (int, float)):
             market_cap = f"{market_cap / 1_000_000_000:.2f}B"
        
        info_dict = {
            "name": name, "currency": currency, "market_cap": market_cap,
            "pe_ratio": f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A",
            "dividend_yield": f"{dividend_yield:.2f}%" if dividend_yield else "N/A"
        }
        return info_dict
    except Exception as e:
        print(f"An error occurred while fetching stock info: {e}")
        return {"name": ticker_symbol, "currency": "USD", "market_cap": "N/A", "pe_ratio": "N/A", "dividend_yield": "N/A"}





def get_data_with_indicators(ticker_symbol, n_years):
    """Downloads historical data and calculates technical indicators."""
    start_date = datetime.datetime.now() - datetime.timedelta(days=n_years * 365)
    end_date = datetime.datetime.now()
    
    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = data.columns.str.lower()

    
    data.ta.sma(length=20, append=True)
    data.ta.sma(length=50, append=True)
    data.ta.sma(length=100, append=True)
    data.ta.sma(length=200, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    data.dropna(inplace=True)
    return data





def train_and_predict(data):
    """Trains an LSTM model and returns predictions for the next 8 days."""
    if data.empty: return None

    target_col = 'close'
    features = ['close', 'volume', 'SMA_20', 'RSI_14', 'MACDh_12_26_9']
    
    features_to_use = [f for f in features if f in data.columns]
    if len(features_to_use) < 2: return None
    dataset = data[features_to_use].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit_transform(data[[target_col]])



    
    
    training_data_len = int(np.ceil(len(dataset) * .8))
    train_data = scaled_data[0:int(training_data_len), :]
    
    x_train, y_train = [], []
    time_step = 60

    for i in range(time_step, len(train_data)):
        x_train.append(train_data[i-time_step:i, :])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential([
        LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        Dropout(0.2),
        LSTM(60, return_sequences=False),
        Dropout(0.2),
        Dense(60),
        Dense(30),
        Dense(15),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=0)
    


    
    string_buffer = StringIO()
    sys.stdout = string_buffer
    model.summary()
    sys.stdout = sys.__stdout__
    model_summary = string_buffer.getvalue()

    test_data = scaled_data[training_data_len - time_step:, :]
    x_test, y_test = [], data[target_col][training_data_len:].values
    
    for i in range(time_step, len(test_data)):
        x_test.append(test_data[i-time_step:i, :])
        
    x_test = np.array(x_test)
    
    predictions_scaled = model.predict(x_test)
    predictions = target_scaler.inverse_transform(predictions_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    accuracy = 100 * (1 - mape)

    train_df = data.iloc[:training_data_len]
    valid_df = data.iloc[training_data_len:].copy()
    valid_df['Predictions'] = predictions

    future_predictions = []
    current_sequence = scaled_data[-time_step:]
    current_sequence = np.reshape(current_sequence, (1, time_step, len(features_to_use)))




    for _ in range(8):
        next_day_prediction_scaled = model.predict(current_sequence)
        next_day_prediction = target_scaler.inverse_transform(next_day_prediction_scaled)[0][0]
        future_predictions.append(next_day_prediction)
        
        new_input_row = current_sequence[0, -1, :].copy()
        new_input_row[0] = next_day_prediction_scaled[0, 0]
        new_input_row = new_input_row.reshape(1, 1, len(features_to_use))
        
        current_sequence = np.append(current_sequence[:, 1:, :], new_input_row, axis=1)

    last_actual_price = y_test[-1] if len(y_test) > 0 else 0

    return (
        train_df, valid_df, rmse, mae, accuracy, last_actual_price, future_predictions,
        history, model_summary, features_to_use
    )
