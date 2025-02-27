# Trading App with RSI and Machine Learning

This project is an automated trading application built in Python. It uses historical stock data from Yahoo Finance to implement a simple RSI (Relative Strength Index) trading strategy and enhances it with machine learning using scikit-learn. The app fetches daily candle data, calculates RSI, simulates a baseline strategy (buy when RSI < 30, sell when RSI > 70), and trains a Random Forest model to predict future buy/sell/hold actions.

## Features

- **Data Source**: Pulls historical OHLCV data from Yahoo Finance for any ticker (e.g., SPY, AAPL).
- **RSI Strategy**: Baseline signals: Buy when RSI drops below 30, Sell when RSI exceeds 70.
- **Machine Learning**: Uses scikit-learn’s Random Forest Classifier to learn and predict trading signals based on price, volume, and RSI.
- **Prediction**: Outputs the next suggested action (Buy, Sell, or Hold) based on the latest data.

## Prerequisites

- **Python 3.6+**: Ensure Python is installed on your system.
- **Git**: For cloning and managing the repository.
- **macOS/Linux/Windows**: Tested on macOS, but should work cross-platform.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:yourusername/trading-app.git
cd trading-app
```

### 2. Create and activate VE

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip list
```

### Run Script

```bash
python3 main.py
```

### Sample Output

```bash
Baseline Strategy (RSI < 30 Buy, RSI > 70 Sell):
Buy Signals: 45, Sell Signals: 32
Model Accuracy on Test Data: 0.85
Predicted Next Action (based on latest RSI: 45.32): Hold
```

### Deactivate environment

```bash
deactivate
```
