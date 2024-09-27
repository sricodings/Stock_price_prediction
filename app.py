import streamlit as st
import sqlite3
import hashlib
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time



def create_tables():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            feedback TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, role='user'):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (username, hashed_password, role))
        conn.commit()
        st.success("You have successfully registered!")
    except sqlite3.IntegrityError:
        st.error("Username already exists! Please choose a different username.")
    conn.close()

def check_user_credentials(username, password):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result

def get_feedback():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT * FROM feedback")
    feedback = c.fetchall()
    conn.close()
    return feedback

def save_feedback(username, feedback):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("INSERT INTO feedback (username, feedback) VALUES (?, ?)", (username, feedback))
    conn.commit()
    conn.close()

def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Open'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def create_lstm_model(X_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prediction_page(asset_type, ticker):
    st.subheader(f"{asset_type} Price Prediction")

    period = st.selectbox(
        "Select period for data download:",
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    )

    interval = st.selectbox(
        "Select data interval:",
        ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    )
    with st.spinner(f"Fetching data for {ticker}..."):
        data = yf.download(tickers=ticker, period=period, interval=interval)

        if data.empty:
            st.write("No data found.")
        else:
            data['7-day MA'] = data['Open'].rolling(window=7*24).mean()

            X, y, scaler = prepare_data(data)
            X_train = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = create_lstm_model(X_train)
            model.fit(X_train, y, epochs=10, batch_size=32, verbose=1)

            user_input_date = st.date_input("Enter the prediction date", value=pd.to_datetime('today'))
            prediction_date = pd.to_datetime(user_input_date)
            last_date = data.index[-1].to_pydatetime().replace(tzinfo=None)

            hours_until_prediction = int((prediction_date - last_date).total_seconds() // 3600)

            if hours_until_prediction < 0:
                st.write("The prediction date is in the past. Please enter a future date.")
            else:
                last_60_hours = data['Open'].values[-60:].reshape(-1, 1)
                last_60_hours_scaled = scaler.transform(last_60_hours)
                last_60_hours_scaled = np.reshape(last_60_hours_scaled, (1, last_60_hours_scaled.shape[0], 1))

                future_dates = [last_date + timedelta(hours=i) for i in range(1, hours_until_prediction + 1)]
                future_predictions = []

                for _ in range(hours_until_prediction):
                    predicted_price_scaled = model.predict(last_60_hours_scaled)
                    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0, 0]
                    future_predictions.append(predicted_price)

                    last_60_hours = np.append(last_60_hours[1:], [[predicted_price]], axis=0)
                    last_60_hours_scaled = scaler.transform(last_60_hours)
                    last_60_hours_scaled = np.reshape(last_60_hours_scaled, (1, last_60_hours_scaled.shape[0], 1))

                future_predictions_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })

                selected_predicted_price = future_predictions_df.iloc[-1]['Predicted Price']
                last_known_price = data['Open'].iloc[-1]
                advice = "Buy" if selected_predicted_price > last_known_price else "Sell"

                st.write(f"Predicted price for {prediction_date.date()}: ${selected_predicted_price:.2f}")
                st.write(f"Current price: ${last_known_price:.2f}")
                st.write(f"Advice: {advice}")

                chart_type = st.radio("Select Chart Type", ("Line Chart", "Candlestick Chart"))

                if chart_type == "Line Chart":
                    trace1 = go.Scatter(
                        x=data.index,
                        y=data['Open'],
                        mode='lines',
                        name='Open Price'
                    )

                    trace2 = go.Scatter(
                        x=data.index,
                        y=data['7-day MA'] if '7-day MA' in data.columns else pd.Series([np.nan] * len(data)),
                        mode='lines',
                        name='7-Day Moving Average'
                    )

                    trace3 = go.Scatter(
                        x=future_predictions_df['Date'],
                        y=future_predictions_df['Predicted Price'],
                        mode='lines',
                        name='Future Predictions',
                        line=dict(color='red', width=2)
                    )

                    layout = go.Layout(
                        title=f'{ticker} Price Prediction',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Open Price (USD)', fixedrange=False),
                        template='plotly_dark',
                        width=1400,
                        height=700
                    )

                    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)


                else:
                    candlestick = go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Candlestick'
                    )

                    trace_pred = go.Scatter(
                        x=future_predictions_df['Date'],
                        y=future_predictions_df['Predicted Price'],
                        mode='lines',
                        name='Future Predictions',
                        line=dict(color='red', width=2)
                    )

                    layout = go.Layout(
                        title=f'{ticker} Candlestick Chart & Predictions',
                        xaxis=dict(title='Date'),
                        yaxis=dict(title='Price (USD)', fixedrange=False),
                        template='plotly_dark',
                        width=1400,
                        height=700
                    )

                    fig = go.Figure(data=[candlestick, trace_pred], layout=layout)


                st.plotly_chart(fig)

def require_login(func):
    def wrapper(*args, **kwargs):
        if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
            st.warning("Please log in to access this page.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

def admin_login():
    st.subheader("Admin Login")
    username = st.text_input("Username", help="Enter your admin username.")
    password = st.text_input("Password", type="password", help="Enter your admin password.")
    if st.button("Login"):
        if check_user_credentials(username, password):
            user = check_user_credentials(username, password)
            if user and user[3] == 'admin':
                st.session_state['admin_logged_in'] = True
                st.success("Admin logged in successfully!")
            else:
                st.error("Invalid admin credentials or insufficient privileges.")
        else:
            st.error("Invalid credentials.")

@require_login
def admin_page():
    st.subheader("Admin Dashboard")
    
    if 'admin_logged_in' in st.session_state and st.session_state['admin_logged_in']:
        st.write("Welcome, Admin!")
        feedback = get_feedback()
        
        if feedback:
            st.subheader("User Feedback")
            for fb in feedback:
                st.write(f"User: {fb[1]} - Feedback: {fb[2]}")
        else:
            st.write("No feedback available.")
    else:
        st.warning("You need to be an admin to access this page.")

def feedback_page():
    st.subheader("User Feedback")
    username = st.text_input("Username", help="Enter your username.")
    feedback = st.text_area("Feedback", help="Provide your feedback here.")
    if st.button("Submit Feedback"):
        if username and feedback:
            save_feedback(username, feedback)
            st.success("Feedback submitted successfully!")
        else:
            st.error("Please enter both username and feedback.")


def fetch_dynamic_news_content(urls):
    content_list = []
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        for url in urls:
            driver.get(url)
            time.sleep(5)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            content_items = soup.find_all('div', class_='news-analysis-v2_content__z0iLP w-full text-xs sm:flex-1')
            
            for item in content_items:
                title_tag = item.find('a')
                if title_tag:
                    title = title_tag.text.strip()
                    link = title_tag['href']
                    if not link.startswith('http'):
                        link = 'https://www.investing.com' + link
                    content_list.append({'title': title, 'link': link})
        
        driver.quit()
        
    except Exception as e:
        st.error(f"Error fetching content: {e}")
    
    return content_list

def display_news_content(news_content):
    st.subheader("Latest Financial News")
    
    if news_content:
        current_section = ""
        for item in news_content:
            title = item['title']
            link = item['link']
            
            st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 15px; background-color: #f9f9f9;">
                    <h3 style="color: #2b8a3e;">{title}</h3>
                    <a href="{link}" style="text-decoration: none; color: #1a73e8; font-weight: bold;" target="_blank">Read more</a>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No content found.")

def content_page():
    urls = [
        'https://www.investing.com/news/stock-market-news',
        'https://www.investing.com/news/cryptocurrency-news',
        'https://www.investing.com/news/forex-news'
    ]
    news_content = fetch_dynamic_news_content(urls)
    display_news_content(news_content)

def main():
    create_tables()

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    with st.sidebar:
        selected = option_menu(
        menu_title=None,
        options=["Home", "Forex", "Stocks", "Coins", "News", "Register", "Login", "Feedback", "Admin"],
        icons=["house", "currency-exchange", "bar-chart", "coin", "newspaper", "person-plus", "person", "envelope", "shield"],
        menu_icon="cast",
        default_index=0
    )

    if selected == "Home":
        st.write("Welcome to the Financial Prediction App!")

    elif selected == "Forex":
        if st.session_state['logged_in']:
            forex_ticker = st.selectbox("Select Forex Pair", ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'])
            prediction_page('Forex', forex_ticker)
        else:
            st.warning("Please log in to access this page.")

    elif selected == "Stocks":
        if st.session_state['logged_in']:
            stock_ticker = st.selectbox("Select Stock", ['AAPL', 'GOOGL', 'MSFT'])
            prediction_page('Stocks', stock_ticker)
        else:
            st.warning("Please log in to access this page.")

    elif selected == "Coins":
        if st.session_state['logged_in']:
            coin_ticker = st.selectbox("Select Coin", ['BTC-USD', 'ETH-USD', 'LTC-USD','BNB-USD', 'SOL-USD', 'AVAX-USD', 'NEAR-USD', 'APT-USD'])
            prediction_page('Coins', coin_ticker)
        else:
            st.warning("Please log in to access this page.")

    elif selected == "News":
        content_page()

    elif selected == "Register":
        st.subheader("Register")
        username = st.text_input("Username", help="Enter your desired username.")
        password = st.text_input("Password", type="password", help="Choose a strong password.")
        role = "user"
        if st.button("Register"):
            register_user(username, password, role)

    elif selected == "Login":
        st.subheader("Login")
        username = st.text_input("Username", help="Enter your username.")
        password = st.text_input("Password", type="password", help="Enter your password.")
        if st.button("Login"):
            if check_user_credentials(username, password):
                st.session_state['logged_in'] = True
                st.success("You are logged in!")
            else:
                st.error("Invalid credentials, please try again.")

    elif selected == "Feedback":
        feedback_page()

    elif selected == "Admin":
        if st.session_state['admin_logged_in']:
            admin_page()
        else:
            admin_login()

if __name__ == "__main__":
    main()