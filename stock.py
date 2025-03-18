import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

# Set page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Initialize yfinance with pandas_datareader
yf.pdr_override()

# Function to get stock data
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        return hist, info, financials, balance_sheet, cash_flow
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None, None, None

# Function to calculate technical indicators
def calculate_indicators(df):
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Upper_BB'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

# Function to calculate fundamental ratios
def calculate_fundamental_ratios(info, financials, balance_sheet):
    ratios = {}
    
    # Get key financial metrics if available
    try:
        # P/E ratio
        ratios['P/E Ratio'] = info.get('trailingPE', info.get('forwardPE', 'N/A'))
        
        # P/B ratio
        ratios['P/B Ratio'] = info.get('priceToBook', 'N/A')
        
        # Dividend Yield
        ratios['Dividend Yield (%)'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 'N/A'
        
        # Debt to Equity
        if 'totalDebt' in info and 'totalStockholderEquity' in info and info['totalStockholderEquity'] != 0:
            ratios['Debt to Equity'] = info['totalDebt'] / info['totalStockholderEquity']
        else:
            ratios['Debt to Equity'] = 'N/A'
        
        # ROE
        ratios['ROE (%)'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 'N/A'
        
        # ROA
        ratios['ROA (%)'] = info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 'N/A'
        
        # Current Ratio
        if 'totalCurrentAssets' in info and 'totalCurrentLiabilities' in info and info['totalCurrentLiabilities'] != 0:
            ratios['Current Ratio'] = info['totalCurrentAssets'] / info['totalCurrentLiabilities']
        else:
            ratios['Current Ratio'] = 'N/A'
        
        # Profit Margin
        ratios['Profit Margin (%)'] = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 'N/A'
        
        # EPS
        ratios['EPS'] = info.get('trailingEPS', 'N/A')
        
        # Market Cap
        ratios['Market Cap'] = info.get('marketCap', 'N/A')
        
        # 52-Week High
        ratios['52-Week High'] = info.get('fiftyTwoWeekHigh', 'N/A')
        
        # 52-Week Low
        ratios['52-Week Low'] = info.get('fiftyTwoWeekLow', 'N/A')
        
    except Exception as e:
        st.warning(f"Some ratios couldn't be calculated: {e}")
    
    return ratios

# Function to analyze stock performance
def analyze_stock(hist, ratios):
    analysis = {}
    
    # Recent performance
    try:
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            price_year_ago = hist['Close'].iloc[0]
            yearly_return = ((current_price / price_year_ago) - 1) * 100
            analysis['Current Price'] = f"${current_price:.2f}"
            analysis['1-Year Return'] = f"{yearly_return:.2f}%"
            
            # Volatility
            returns = hist['Close'].pct_change().dropna()
            analysis['Volatility (Daily)'] = f"{returns.std() * 100:.2f}%"
            analysis['Volatility (Annualized)'] = f"{returns.std() * np.sqrt(252) * 100:.2f}%"
            
            # Trend analysis
            if hist['Close'].iloc[-1] > hist['MA50'].iloc[-1] and hist['MA50'].iloc[-1] > hist['MA200'].iloc[-1]:
                analysis['Trend'] = "Bullish (Price > MA50 > MA200)"
            elif hist['Close'].iloc[-1] < hist['MA50'].iloc[-1] and hist['MA50'].iloc[-1] < hist['MA200'].iloc[-1]:
                analysis['Trend'] = "Bearish (Price < MA50 < MA200)"
            else:
                analysis['Trend'] = "Mixed/Neutral"
            
            # RSI analysis
            latest_rsi = hist['RSI'].iloc[-1]
            if latest_rsi > 70:
                analysis['RSI'] = f"{latest_rsi:.2f} (Overbought)"
            elif latest_rsi < 30:
                analysis['RSI'] = f"{latest_rsi:.2f} (Oversold)"
            else:
                analysis['RSI'] = f"{latest_rsi:.2f} (Neutral)"
            
            # MACD analysis
            if hist['MACD'].iloc[-1] > hist['Signal'].iloc[-1]:
                analysis['MACD'] = "Bullish Signal"
            else:
                analysis['MACD'] = "Bearish Signal"
    except Exception as e:
        st.warning(f"Error in performance analysis: {e}")
    
    # Fundamental analysis
    try:
        # P/E analysis
        if isinstance(ratios['P/E Ratio'], (int, float)):
            if ratios['P/E Ratio'] < 15:
                analysis['P/E Analysis'] = f"{ratios['P/E Ratio']:.2f} (Potentially Undervalued)"
            elif ratios['P/E Ratio'] > 30:
                analysis['P/E Analysis'] = f"{ratios['P/E Ratio']:.2f} (Potentially Overvalued)"
            else:
                analysis['P/E Analysis'] = f"{ratios['P/E Ratio']:.2f} (Fair Value)"
        else:
            analysis['P/E Analysis'] = "N/A"
        
        # P/B analysis
        if isinstance(ratios['P/B Ratio'], (int, float)):
            if ratios['P/B Ratio'] < 1:
                analysis['P/B Analysis'] = f"{ratios['P/B Ratio']:.2f} (Potentially Undervalued)"
            elif ratios['P/B Ratio'] > 3:
                analysis['P/B Analysis'] = f"{ratios['P/B Ratio']:.2f} (Potentially Overvalued)"
            else:
                analysis['P/B Analysis'] = f"{ratios['P/B Ratio']:.2f} (Fair Value)"
        else:
            analysis['P/B Analysis'] = "N/A"
        
        # Dividend analysis
        if isinstance(ratios['Dividend Yield (%)'], (int, float)):
            if ratios['Dividend Yield (%)'] > 4:
                analysis['Dividend Analysis'] = f"{ratios['Dividend Yield (%)']:.2f}% (High Yield)"
            elif ratios['Dividend Yield (%)'] > 0:
                analysis['Dividend Analysis'] = f"{ratios['Dividend Yield (%)']:.2f}% (Pays Dividend)"
            else:
                analysis['Dividend Analysis'] = "No Dividend"
        else:
            analysis['Dividend Analysis'] = "N/A"
        
    except Exception as e:
        st.warning(f"Error in fundamental analysis: {e}")
    
    return analysis

# Function to generate investment suggestion
def generate_investment_suggestion(analysis, ratios):
    score = 0
    max_score = 0
    reasons = []
    
    # Technical indicators
    if 'Trend' in analysis:
        max_score += 1
        if 'Bullish' in analysis['Trend']:
            score += 1
            reasons.append("Positive price trend")
        elif 'Bearish' in analysis['Trend']:
            reasons.append("Negative price trend")
    
    if 'RSI' in analysis:
        max_score += 1
        if 'Oversold' in analysis['RSI']:
            score += 1
            reasons.append("Oversold (potential buying opportunity)")
        elif 'Overbought' in analysis['RSI']:
            reasons.append("Overbought (potential selling opportunity)")
    
    if 'MACD' in analysis:
        max_score += 1
        if 'Bullish' in analysis['MACD']:
            score += 1
            reasons.append("Positive MACD crossover")
        else:
            reasons.append("Negative MACD crossover")
    
    # Fundamental indicators
    if 'P/E Analysis' in analysis:
        max_score += 1
        if 'Undervalued' in analysis['P/E Analysis']:
            score += 1
            reasons.append("P/E ratio suggests undervaluation")
        elif 'Overvalued' in analysis['P/E Analysis']:
            reasons.append("P/E ratio suggests overvaluation")
    
    if 'P/B Analysis' in analysis:
        max_score += 1
        if 'Undervalued' in analysis['P/B Analysis']:
            score += 1
            reasons.append("P/B ratio suggests undervaluation")
        elif 'Overvalued' in analysis['P/B Analysis']:
            reasons.append("P/B ratio suggests overvaluation")
    
    if 'Dividend Analysis' in analysis:
        max_score += 1
        if 'High Yield' in analysis['Dividend Analysis']:
            score += 1
            reasons.append("High dividend yield")
        elif 'Pays Dividend' in analysis['Dividend Analysis']:
            score += 0.5
            reasons.append("Pays regular dividends")
    
    # ROE and ROA
    if isinstance(ratios.get('ROE (%)', 'N/A'), (int, float)):
        max_score += 1
        if ratios['ROE (%)'] > 15:
            score += 1
            reasons.append(f"Strong ROE ({ratios['ROE (%)']:.2f}%)")
        elif ratios['ROE (%)'] <= 5:
            reasons.append(f"Weak ROE ({ratios['ROE (%)']:.2f}%)")
    
    # Calculate final score as percentage
    final_score = (score / max_score) * 100 if max_score > 0 else 0
    
    # Generate recommendation
    if final_score >= 70:
        recommendation = "Strong Buy"
    elif final_score >= 60:
        recommendation = "Buy"
    elif final_score >= 40:
        recommendation = "Hold"
    elif final_score >= 30:
        recommendation = "Reduce"
    else:
        recommendation = "Sell"
    
    return recommendation, final_score, reasons

# Main app
st.title("Stock Analysis Dashboard")

# Sidebar
st.sidebar.header("Stock Selection")
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", "AAPL")
period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
selected_period = st.sidebar.selectbox("Select Time Period", period_options, index=3)

# For comparison
st.sidebar.header("Compare with another stock")
enable_comparison = st.sidebar.checkbox("Enable Comparison")
comparison_ticker = None
if enable_comparison:
    comparison_ticker = st.sidebar.text_input("Enter Comparison Stock Ticker", "MSFT")

# Load data
if ticker_input:
    hist, info, financials, balance_sheet, cash_flow = get_stock_data(ticker_input, selected_period)
    
    if hist is not None and info is not None:
        # Calculate indicators
        hist = calculate_indicators(hist)
        
        # Calculate fundamental ratios
        ratios = calculate_fundamental_ratios(info, financials, balance_sheet)
        
        # Analyze stock
        analysis = analyze_stock(hist, ratios)
        
        # Generate investment suggestion
        recommendation, score, reasons = generate_investment_suggestion(analysis, ratios)
        
        # Company information
        st.header(f"{info.get('longName', ticker_input)} ({ticker_input})")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Business Summary:** {info.get('longBusinessSummary', 'N/A')[:500]}...")
        with col2:
            st.metric("Current Price", 
                      f"${hist['Close'].iloc[-1]:.2f}", 
                      f"{(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100:.2f}%")
            st.metric("Recommendation", recommendation, f"Score: {score:.2f}%")
        
        # Price chart
        st.subheader("Price History")
        
        # Prepare data for Plotly
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=ticker_input
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20', line=dict(color='blue', width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA200'], name='MA200', line=dict(color='red', width=1)))
        
        # Add comparison ticker if enabled
        if enable_comparison and comparison_ticker:
            comp_hist, comp_info, _, _, _ = get_stock_data(comparison_ticker, selected_period)
            if comp_hist is not None:
                # Normalize both series to start at 100 for better comparison
                hist_normalized = hist['Close'] / hist['Close'].iloc[0] * 100
                comp_normalized = comp_hist['Close'] / comp_hist['Close'].iloc[0] * 100
                
                fig.add_trace(go.Scatter(
                    x=comp_hist.index,
                    y=comp_normalized,
                    name=f"{comparison_ticker} (Normalized)",
                    line=dict(color='purple', width=2, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist_normalized,
                    name=f"{ticker_input} (Normalized)",
                    line=dict(color='green', width=2, dash='dash')
                ))
        
        # Update layout
        fig.update_layout(
            title=f"{ticker_input} Stock Price ({selected_period})",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            legend_title="Legend",
            template="plotly_white",
            height=600
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators
        st.subheader("Technical Indicators")
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI'))
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=[70] * len(hist.index), name='Overbought (70)', line=dict(color='red', width=1, dash='dash')))
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=[30] * len(hist.index), name='Oversold (30)', line=dict(color='green', width=1, dash='dash')))
            fig_rsi.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI", template="plotly_white")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD Chart
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD'))
            fig_macd.add_trace(go.Scatter(x=hist.index, y=hist['Signal'], name='Signal'))
            fig_macd.add_trace(go.Bar(x=hist.index, y=hist['MACD'] - hist['Signal'], name='Histogram'))
            fig_macd.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value", template="plotly_white")
            st.plotly_chart(fig_macd, use_container_width=True)
        
        # Bollinger Bands
        st.subheader("Bollinger Bands")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close Price'))
        fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Upper_BB'], name='Upper Band', line=dict(color='red', width=1)))
        fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='20-Day MA', line=dict(color='orange', width=1)))
        fig_bb.add_trace(go.Scatter(x=hist.index, y=hist['Lower_BB'], name='Lower Band', line=dict(color='green', width=1)))
        fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price ($)", template="plotly_white")
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Show key metrics
        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("P/E Ratio", f"{ratios['P/E Ratio']:.2f}" if isinstance(ratios['P/E Ratio'], (int, float)) else "N/A")
            st.metric("P/B Ratio", f"{ratios['P/B Ratio']:.2f}" if isinstance(ratios['P/B Ratio'], (int, float)) else "N/A")
            st.metric("Dividend Yield", f"{ratios['Dividend Yield (%)']:.2f}%" if isinstance(ratios['Dividend Yield (%)'], (int, float)) else "N/A")
        
        with col2:
            st.metric("ROE", f"{ratios['ROE (%)']:.2f}%" if isinstance(ratios['ROE (%)'], (int, float)) else "N/A")
            st.metric("ROA", f"{ratios['ROA (%)']:.2f}%" if isinstance(ratios['ROA (%)'], (int, float)) else "N/A")
            st.metric("Profit Margin", f"{ratios['Profit Margin (%)']:.2f}%" if isinstance(ratios['Profit Margin (%)'], (int, float)) else "N/A")
        
        with col3:
            st.metric("EPS", f"${ratios['EPS']:.2f}" if isinstance(ratios['EPS'], (int, float)) else "N/A")
            st.metric("Market Cap", f"${ratios['Market Cap'] / 1e9:.2f}B" if isinstance(ratios['Market Cap'], (int, float)) else "N/A")
            st.metric("Debt to Equity", f"{ratios['Debt to Equity']:.2f}" if isinstance(ratios['Debt to Equity'], (int, float)) else "N/A")
        
        # Analysis summary
        st.subheader("Analysis Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Analysis**")
            for key in ['Trend', 'RSI', 'MACD']:
                if key in analysis:
                    st.write(f"- **{key}:** {analysis[key]}")
        
        with col2:
            st.write("**Fundamental Analysis**")
            for key in ['P/E Analysis', 'P/B Analysis', 'Dividend Analysis']:
                if key in analysis:
                    st.write(f"- **{key}:** {analysis[key]}")
        
        # Investment recommendation
        st.subheader("Investment Recommendation")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create a gauge chart for the score
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Score: {recommendation}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "red"},
                        {'range': [30, 40], 'color': "orange"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 70], 'color': "lightgreen"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Key Factors:**")
            for reason in reasons:
                st.write(f"- {reason}")
        
        # Comparison with another stock if enabled
        if enable_comparison and comparison_ticker:
            st.subheader(f"Comparison: {ticker_input} vs {comparison_ticker}")
            
            comp_hist, comp_info, comp_financials, comp_balance_sheet, comp_cash_flow = get_stock_data(comparison_ticker, selected_period)
            
            if comp_hist is not None and comp_info is not None:
                # Calculate indicators
                comp_hist = calculate_indicators(comp_hist)
                
                # Calculate fundamental ratios
                comp_ratios = calculate_fundamental_ratios(comp_info, comp_financials, comp_balance_sheet)
                
                # Create comparison table
                comparison_data = {
                    'Metric': ['P/E Ratio', 'P/B Ratio', 'Dividend Yield (%)', 'ROE (%)', 'ROA (%)', 'Profit Margin (%)', 'EPS'],
                    f'{ticker_input}': [
                        f"{ratios['P/E Ratio']:.2f}" if isinstance(ratios['P/E Ratio'], (int, float)) else "N/A",
                        f"{ratios['P/B Ratio']:.2f}" if isinstance(ratios['P/B Ratio'], (int, float)) else "N/A",
                        f"{ratios['Dividend Yield (%)']:.2f}%" if isinstance(ratios['Dividend Yield (%)'], (int, float)) else "N/A",
                        f"{ratios['ROE (%)']:.2f}%" if isinstance(ratios['ROE (%)'], (int, float)) else "N/A",
                        f"{ratios['ROA (%)']:.2f}%" if isinstance(ratios['ROA (%)'], (int, float)) else "N/A",
                        f"{ratios['Profit Margin (%)']:.2f}%" if isinstance(ratios['Profit Margin (%)'], (int, float)) else "N/A",
                        f"${ratios['EPS']:.2f}" if isinstance(ratios['EPS'], (int, float)) else "N/A"
                    ],
                    f'{comparison_ticker}': [
                        f"{comp_ratios['P/E Ratio']:.2f}" if isinstance(comp_ratios['P/E Ratio'], (int, float)) else "N/A",
                        f"{comp_ratios['P/B Ratio']:.2f}" if isinstance(comp_ratios['P/B Ratio'], (int, float)) else "N/A",
                        f"{comp_ratios['Dividend Yield (%)']:.2f}%" if isinstance(comp_ratios['Dividend Yield (%)'], (int, float)) else "N/A",
                        f"{comp_ratios['ROE (%)']:.2f}%" if isinstance(comp_ratios['ROE (%)'], (int, float)) else "N/A",
                        f"{comp_ratios['ROA (%)']:.2f}%" if isinstance(comp_ratios['ROA (%)'], (int, float)) else "N/A",
                        f"{comp_ratios['Profit Margin (%)']:.2f}%" if isinstance(comp_ratios['Profit Margin (%)'], (int, float)) else "N/A",
                        f"${comp_ratios['EPS']:.2f}" if isinstance(comp_ratios['EPS'], (int, float)) else "N/A"
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                st.table(comparison_df)
                
                # Performance comparison
                hist_ret = hist['Close'].pct_change().dropna()
                comp_hist_ret = comp_hist['Close'].pct_change().dropna()
                
                # Calculate cumulative returns
                hist_cum_ret = (1 + hist_ret).cumprod() - 1
                comp_cum_ret = (1 + comp_hist_ret).cumprod() - 1
                
                # Create common date range
                common_dates = hist_cum_ret.index.intersection(comp_cum_ret.index)
                
                if len(common_dates) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=common_dates, y=hist_cum_ret[common_dates] * 100, name=f'{ticker_input} Return (%)'))
                    fig.add_trace(go.Scatter(x=common_dates, y=comp_cum_ret[common_dates] * 100, name=f'{comparison_ticker} Return (%)'))
                    fig.update_layout(
                        title=f"Cumulative Return Comparison ({selected_period})",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk-Return Comparison
                    st.subheader("Risk-Return Comparison")
                    
                    # Calculate annualized returns
                    trading_days = 252
                    ticker_annual_return = hist_ret.mean() * trading_days * 100
                    comp_annual_return = comp_hist_ret.mean() * trading_days * 100
                    
                    # Calculate annualized volatility (risk)
                    ticker_annual_risk = hist_ret.std() * np.sqrt(trading_days) * 100
                    comp_annual_risk = comp_hist_ret.std() * np.sqrt(trading_days) * 100
                    
                    # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
                    risk_free_rate = 0.02
                    ticker_sharpe = (ticker_annual_return/100 - risk_free_rate) / (ticker_annual_risk/100)
                    comp_sharpe = (comp_annual_return/100 - risk_free_rate) / (comp_annual_risk/100)
                    
                    # Create comparison table
                    risk_return_data = {
                        'Metric': ['Annualized Return (%)', 'Annualized Risk (%)', 'Sharpe Ratio'],
                        f'{ticker_input}': [
                            f"{ticker_annual_return:.2f}%",
                            f"{ticker_annual_risk:.2f}%",
                            f"{ticker_sharpe:.2f}"
                        ],
                        f'{comparison_ticker}': [
                            f"{comp_annual_return:.2f}%",
                            f"{comp_annual_risk:.2f}%",
                            f"{comp_sharpe:.2f}"
                        ]
                    }
                    
                    risk_return_df = pd.DataFrame(risk_return_data)
                    st.table(risk_return_df)
                
                    # Plot risk-return scatter
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=[ticker_annual_risk],
                        y=[ticker_annual_return],
                        mode='markers+text',
                        marker=dict(size=15, color='blue'),
                        text=[ticker_input],
                        textposition="top center",
                        name=ticker_input
                    ))
                    fig.add_trace(go.Scatter(
                        x=[comp_annual_risk],
                        y=[comp_annual_return],
                        mode='markers+text',
                        marker=dict(size=15, color='red'),
                        text=[comparison_ticker],
                        textposition="top center",
                        name=comparison_ticker
                    ))
                    fig.update_layout(
                        title="Risk vs. Return",
                        xaxis_title="Risk (Annual Volatility %)",
                        yaxis_title="Return (Annual %)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create correlation matrix
                    returns_df = pd.DataFrame({
                        ticker_input: hist_ret,
                        comparison_ticker: comp_hist_ret
                    }).dropna()
                    
                    correlation = returns_df.corr().iloc[0, 1]
                    st.write(f"**Correlation:** {correlation:.4f}")
                    
                    # Create a heatmap for correlation
                    corr_matrix = returns_df.corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title="Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Volume analysis
        st.subheader("Volume Analysis")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume'))
        # Add 20-day average volume
        hist['Volume_MA20'] = hist['Volume'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(x=hist.index, y=hist['Volume_MA20'], name='20-Day Avg Volume', line=dict(color='red', width=2)))
        fig.update_layout(
            title=f"{ticker_input} Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical Events Analysis
        st.subheader("Historical Events Impact")
        st.write("Analyze how major market events affected the stock:")
        
        # Find significant price movements
        hist['Daily_Return'] = hist['Close'].pct_change() * 100
        significant_moves = hist[abs(hist['Daily_Return']) > 5].sort_values('Daily_Return', ascending=False)
        
        if not significant_moves.empty:
            st.write("**Significant Price Movements (>5% in a day):**")
            significant_moves_display = pd.DataFrame({
                'Date': significant_moves.index,
                'Price Change (%)': significant_moves['Daily_Return'].round(2),
                'Close Price': significant_moves['Close'].round(2),
                'Volume': significant_moves['Volume']
            })
            st.dataframe(significant_moves_display)
        else:
            st.write("No significant price movements (>5% in a day) found in the selected period.")
        
        # Financial Health Analysis
        if financials is not None and not financials.empty:
            st.subheader("Financial Health Analysis")
            
            # Display recent quarterly financials
            try:
                # Transpose for better readability
                quarterly_financials = financials.T
                
                # Format the columns to be more readable
                quarterly_financials = quarterly_financials / 1e6  # Convert to millions
                
                # Display the financials
                st.write("**Quarterly Financials (in $ millions):**")
                st.dataframe(quarterly_financials.round(2).head(10))
                
                # Plot revenue and net income trends
                if 'Total Revenue' in quarterly_financials.columns and 'Net Income' in quarterly_financials.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=quarterly_financials.index,
                        y=quarterly_financials['Total Revenue'],
                        name='Revenue'
                    ))
                    fig.add_trace(go.Scatter(
                        x=quarterly_financials.index,
                        y=quarterly_financials['Net Income'],
                        name='Net Income',
                        line=dict(color='red', width=2)
                    ))
                    fig.update_layout(
                        title="Revenue and Net Income Trends",
                        xaxis_title="Quarter",
                        yaxis_title="$ Millions",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display financials: {e}")
        
        # News Sentiment Analysis
        st.subheader("Recent News")
        try:
            news = info.get('news', [])
            if news and len(news) > 0:
                for i, news_item in enumerate(news[:5]):  # Show top 5 news
                    st.write(f"**{news_item.get('title', 'No Title')}**")
                    st.write(f"Source: {news_item.get('publisher', 'Unknown')} | Date: {news_item.get('providerPublishTime', 'Unknown')}")
                    st.write(news_item.get('summary', 'No summary available'))
                    st.write("---")
            else:
                st.write("No recent news available.")
        except Exception as e:
            st.write("Could not retrieve news data.")
        
        # Disclaimer
        st.sidebar.markdown("---")
        st.sidebar.write("**Disclaimer:** This tool is for informational purposes only. It is not financial advice.")
    else:
        st.error(f"Could not retrieve data for {ticker_input}. Please check the ticker symbol and try again.")