import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

# Function to calculate RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 1. Fetch historical data from Yahoo Finance
symbol = "SPY"  # Example: SPY (S&P 500 ETF), change to your preferred ticker
stock = yf.Ticker(symbol)
# Get data as far back as possible (Yahoo Finance limits vary by asset)
data = stock.history(period="max", interval="1d")  # Daily candles
data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()

# 2. Calculate RSI and define baseline strategy signals
data["RSI"] = calculate_rsi(data["Close"])
data["Returns"] = data["Close"].pct_change()

# Baseline strategy: Buy when RSI < 30, Sell when RSI > 70
data["Signal"] = np.where(data["RSI"] < 30, 1, 0)  # 1 = Buy
data["Signal"] = np.where(data["RSI"] > 70, -1, data["Signal"])  # -1 = Sell, 0 = Hold
data["Target"] = data["Signal"].shift(-1)  # Next day's action (what we predict)

# Drop NaN values (from RSI calculation and shifting)
data = data.dropna()

# Simulate baseline strategy performance
print("Baseline Strategy (RSI < 30 Buy, RSI > 70 Sell):")
buy_signals = len(data[data["Signal"] == 1])
sell_signals = len(data[data["Signal"] == -1])
print(f"Buy Signals: {buy_signals}, Sell Signals: {sell_signals}")

# 3. Prepare data for ML
features = data[["Close", "Volume", "RSI"]]  # Features to train on
target = data["Target"]  # Buy (1), Sell (-1), or Hold (0)

# Split into training and testing sets (80% train, 20% test, no shuffle for time series)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy:.2f}")

# 4. Predict the next buy/sell action
# Get the latest available data (last row)
latest_data = features.tail(1)
next_prediction = model.predict(latest_data)[0]

# Interpret the prediction
action = {1: "Buy", -1: "Sell", 0: "Hold"}
print(f"Predicted Next Action (based on latest RSI: {latest_data['RSI'].values[0]:.2f}): {action[next_prediction]}")

# Optional: Show latest data for context
print("\nLatest Data:")
print(latest_data)

# Simulate applying it to "today" (assuming latest data is up to yesterday)
today = datetime.datetime.now().strftime("%Y-%m-%d")
print(f"\nAs of {today}, based on historical data, the model suggests: {action[next_prediction]}")