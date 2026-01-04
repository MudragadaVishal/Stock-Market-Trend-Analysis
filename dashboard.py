import streamlit as st
import pandas as pd
import backend
import plotly.graph_objects as go
import time


st.set_page_config(page_title="Stock Market Trend Analysis Dashboard", page_icon="📈", layout="wide")


st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }

    /* --- NEW: Minimalist & Eye-Catching Sidebar --- */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #161A25;
        border-right: 2px solid #1E88E5;
    }
    /* Style headers within the sidebar for structure */
    [data-testid="stSidebar"] h3 {
        color: #FAFAFA;
        border-bottom: 1px solid #41485c;
        padding-bottom: 10px;
        margin-top: 20px;
        font-size: 1.25rem;
    }

    /* --- Main content fade-in animation --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-content {
        animation: fadeIn 0.8s ease-out;
    }

    /* --- Button Animation --- */
    @keyframes glow {
        0% { box-shadow: 0 0 3px #1E88E5; }
        50% { box-shadow: 0 0 15px #1E88E5; }
        100% { box-shadow: 0 0 3px #1E88E5; }
    }
    .stButton>button {
        color: #FAFAFA;
        background-color: #1E88E5;
        border: none;
        padding: 10px 20px;
        width: 100%;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin-top: 20px; /* Added margin for spacing */
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s;
        animation: glow 2.5s infinite ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        animation: none; /* Pause glow on hover */
    }

    /* --- Metric Card Hover Effect --- */
    .stMetric {
        background-color: #161A25;
        border: 1px solid #41485c;
        border-radius: 0.5rem;
        padding: 1rem;
        transition: all 0.3s ease-in-out;
    }
    .stMetric:hover {
        border: 1px solid #1E88E5;
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* --- Outlook & Summary Section --- */
    .outlook-container {
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        border-width: 2px;
        border-style: solid;
        height: 100%;
    }
    .summary-card {
        background-color: #161A25;
        border: 1px solid #41485c;
        border-left: 5px solid #1E88E5;
        border-radius: 0.5rem;
        padding: 1rem;
        height: 100%;
    }
    .outlook-positive {
        background-color: rgba(38, 166, 154, 0.1);
        border-color: #26A69A;
    }
    .outlook-negative {
        background-color: rgba(239, 83, 80, 0.1);
        border-color: #EF5350;
    }
    .outlook-neutral {
        background-color: rgba(158, 158, 158, 0.1);
        border-color: #9E9E9E;
    }
    .outlook-text {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_price_chart(data, valid_df, currency, selected_smas):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name='Price'))
    fig.add_trace(go.Scatter(x=valid_df.index, y=valid_df['Predictions'], mode='lines', name='Historical Prediction', line=dict(color='#FFD700', width=2, dash='dash')))
    
    sma_colors = {'20': '#00BFFF', '50': '#9370DB', '100': '#32CD32', '200': '#FF6347'}
    for sma in selected_smas:
        col_name = f'SMA_{sma}'
        if col_name in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data[col_name], mode='lines', name=f'SMA {sma}', line=dict(color=sma_colors[str(sma)], width=1.5)))
            
    fig.update_layout(
        title_text='Stock Price Analysis and Prediction', yaxis_title=f'Price ({currency})', xaxis_title='Date',
        xaxis_rangeslider_visible=False, legend_title_text='Legend', hovermode='x unified', height=500,
        margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="#0E1117", plot_bgcolor="#161A25",
        font=dict(color="#FAFAFA")
    )
    return fig

def create_indicator_chart(data, indicator):
    fig = go.Figure()
    if indicator == 'RSI':
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI_14'], mode='lines', name='RSI', line=dict(color='yellow')))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", annotation_position="bottom right")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="bottom right")
        fig.update_layout(title_text='Relative Strength Index (RSI)', yaxis_title='RSI')
    elif indicator == 'MACD':
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_12_26_9'], mode='lines', name='MACD Line', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MACDs_12_26_9'], mode='lines', name='Signal Line', line=dict(color='orange')))
        colors = ['#26A69A' if val >= 0 else '#EF5350' for val in data['MACDh_12_26_9']]
        fig.add_trace(go.Bar(x=data.index, y=data['MACDh_12_26_9'], name='Histogram', marker_color=colors))
        fig.update_layout(title_text='Moving Average Convergence Divergence (MACD)', yaxis_title='Value')
        
    fig.update_layout(
        hovermode='x unified', height=400, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="#0E1117", plot_bgcolor="#161A25", font=dict(color="#FAFAFA")
    )
    return fig


st.title("📈 Stock Market Trend Analysis Dashboard")


with st.sidebar:
    st.header("⚙️ Controls")
    selected_stock_name = st.selectbox("Select Stock, Index, or ETF", options=list(backend.STOCKS_AND_INDICES.keys()))
    n_years = st.slider("Years of historical data:", 1, 15, 3, help="Select the number of years of past data to train the model on.")

    st.header("📊 Overlays")
    selected_smas = st.multiselect(
        "Show Moving Averages (days)",
        options=[20, 50, 100, 200],
        default=[20, 50]
    )

    start_analysis = st.button("🚀 Run Analysis")
    
    st.markdown("---")
    
    with st.expander("ℹ️ About", expanded=False):
        st.info("This dashboard uses an LSTM neural network to forecast stock prices. Remember, this is for educational purposes and not financial advice.")


if start_analysis:
    with st.spinner(f"Fetching {n_years} years of historical data for **{selected_stock_name}**..."):
        ticker_symbol = backend.STOCKS_AND_INDICES[selected_stock_name]
        info = backend.get_stock_info(ticker_symbol)
        data = backend.get_data_with_indicators(ticker_symbol, n_years)

    if data.empty:
        st.error("Could not download data. Please check the ticker symbol or your internet connection.")
    else:
        with st.status("Training Predictive Model...", expanded=True) as status:
            st.write("Building LSTM model architecture...")
            time.sleep(1) 
            st.write("Starting model training on historical data. This may take a moment...")
            results = backend.train_and_predict(data)
            st.write("Finalizing predictions and analysis...")
            time.sleep(1)
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        if results:
            st.markdown('<div class="main-content">', unsafe_allow_html=True) 
            
            train_df, valid_df, rmse, mae, accuracy, last_price, future_predictions, history, model_summary, features_used = results
            
            st.success(f"Analysis complete for **{info['name']} ({ticker_symbol})**")

            with st.container():
                st.subheader("Key Metrics & Latest Price")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Market Cap", info['market_cap'])
                col2.metric("P/E Ratio", info['pe_ratio'])
                col3.metric("Dividend Yield", info['dividend_yield'])
                col4.metric("Last Close Price", f"{last_price:.2f} {info['currency']}")
            st.subheader("8-Day Price Forecast")
            last_date = data.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=8)
            
            row1_cols = st.columns(4)
            row2_cols = st.columns(4)
            all_cols = row1_cols + row2_cols
            for i, col in enumerate(all_cols):
                with col:
                    delta = future_predictions[i] - (future_predictions[i-1] if i > 0 else last_price)
                    st.metric(
                        label=f"Day {i+1} ({future_dates[i].strftime('%b %d')})",
                        value=f"{future_predictions[i]:.2f}",
                        delta=f"{delta:.2f}",
                    )
            
            net_change = future_predictions[-1] - last_price
            percent_change = (net_change / last_price) * 100 if last_price > 0 else 0
            outlook_class, outlook_icon = ("outlook-neutral", "⚪️")
            if percent_change > 1.5: outlook_class, outlook_icon = ("outlook-positive", "🟢")
            elif percent_change < -1.5: outlook_class, outlook_icon = ("outlook-negative", "🔴")

            st.subheader("Forecast Summary & Outlook")
            col_outlook, col_summary = st.columns(2)
            with col_outlook:
                st.markdown(f"""
                <div class="outlook-container {outlook_class}">
                    <span class="outlook-text">{outlook_icon} {outlook_class.replace('-', ' ').title()}</span>
                    <p>Based on the 8-day forecast ({percent_change:+.2f}%), the model suggests a potential {outlook_class.split('-')[1]} trend.</p>
                </div>
                """, unsafe_allow_html=True)
            with col_summary:
                st.markdown(f"""
                <div class="summary-card">
                    <h4>Prediction Summary</h4>
                    <p>The model forecasts a <strong>{outlook_class.split('-')[1]}</strong> trend, projecting a net price change of <strong>{net_change:+.2f} {info['currency']}</strong> ({percent_change:+.2f}%).</p>
                    <small>This prediction is derived from historical data and should be used as one of many tools for analysis.</small>
                </div>
                """, unsafe_allow_html=True)
            st.write("") 

            price_fig = create_price_chart(data, valid_df, info['currency'], selected_smas)
            st.plotly_chart(price_fig, use_container_width=True)

            st.subheader("Technical Indicator Analysis")
            tab1, tab2 = st.tabs(["Relative Strength Index (RSI)", "Moving Average Convergence Divergence (MACD)"])
            with tab1:
                st.write("RSI helps identify overbought (>70) or oversold (<30) conditions.")
                if 'RSI_14' in data.columns: st.plotly_chart(create_indicator_chart(data, 'RSI'), use_container_width=True)
            with tab2:
                st.write("A bullish signal occurs when the MACD line (blue) crosses above the Signal line (orange).")
                if 'MACD_12_26_9' in data.columns: st.plotly_chart(create_indicator_chart(data, 'MACD'), use_container_width=True)
            
            with st.expander("🔬 View Model Performance & Architecture"):
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                col_perf1.metric("Model Accuracy", f"{accuracy:.2f}%")
                col_perf2.metric("RMSE", f"{rmse:.2f}")
                col_perf3.metric("MAE", f"{mae:.2f}")
                st.subheader("LSTM Model Architecture")
                st.code(model_summary, language='text')

            st.caption("Disclaimer: This is not financial advice. Predictions are based on historical data and are not guaranteed to be accurate.")
            st.markdown('</div>', unsafe_allow_html=True) 
        else:
            st.error("An error occurred during model training. The data might not have the required features.")
else:
    st.info("Select your desired stock and settings in the sidebar, then click 'Run Analysis' to begin.")

