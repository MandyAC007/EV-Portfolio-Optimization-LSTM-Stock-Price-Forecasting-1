# EV-Portfolio-Optimization-LSTM-Stock-Price-Forecasting-1
Overview This project combines quantitative portfolio optimization and deep learning-based time series forecasting to guide investment strategies in the Electric Vehicle (EV) sector. We analyze four key EV stocks — Tesla, BYD, NIO, XPeng — using historical price data, technical indicators, and machine learning models.

EV-Portfolio-LSTM/

├── data/                  # CSV datasets
├── plots/                 # Generated charts
├── scripts/               # Python scripts
├── README.md              # Project documentation

## Installation ⚙️ 
pip install -r requirements.txt

## Run ▶️
python portfolio_optimization.py
python lstm_forecast.py


# EV Portfolio Optimization & LSTM Stock Price Forecasting

## Overview

This project combines quantitative portfolio optimization and deep learning-based time series forecasting to guide investment strategies in the Electric Vehicle (EV) sector.
We analyze four key EV stocks — Tesla, BYD, NIO, XPeng — using historical price data, technical indicators, and machine learning models.

## Objectives

Optimize a minimum variance EV stock portfolio using historical returns and covariance matrices.
Assess allocation stability with bootstrapping.
Predict the next 60 trading days (~3 months) of stock prices using LSTM neural networks.
Visualize trends with custom-colored plots for clarity.

## Data Sources

Historical price datasets for:
Tesla
BYD
NIO
XPeng
Enhanced with technical indicators:
Simple Moving Average (SMA)
Relative Strength Index (RSI)
Moving Average Convergence Divergence (MACD)
Cleaned, aligned, and merged into a unified DataFrame of daily returns.

## Methodology

# Part 1: Portfolio Optimization

# 1. Data Processing

# Load & clean CSVs
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Calculate daily returns
df['Return'] = df['Close'].pct_change()

# 2. Optimization

   cov_matrix = np.cov(returns_matrix, rowvar=False)
   inv_cov = np.linalg.inv(cov_matrix)
   ones = np.ones(returns_matrix.shape[1])
   optimal_weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)

# 3. Interpretation

def interpret_weight(row):
    if row['Optimal Weight'] > 0:
        return f"Allocate {row['Optimal Weight']:.2%} to {row['Stock']} (long)"
    elif row['Optimal Weight'] < 0:
        return f"Short {abs(row['Optimal Weight']):.2%} of {row['Stock']}"
    else:
        return f"No allocation to {row['Stock']}"

# 4. Bootstrapping

for _ in range(500):
    sample_idx = np.random.choice(n_obs, size=n_obs, replace=True)
    sample = returns_matrix[sample_idx]
    cov_sample = np.cov(sample, rowvar=False)
    inv_sample = np.linalg.inv(cov_sample)
    w = inv_sample @ ones / (ones.T @ inv_sample @ ones)
    boot_weights.append(w)


for _ in range(500):
    sample_idx = np.random.choice(n_obs, size=n_obs, replace=True)
    sample = returns_matrix[sample_idx]
    cov_sample = np.cov(sample, rowvar=False)
    inv_sample = np.linalg.inv(cov_sample)
    w = inv_sample @ ones / (ones.T @ inv_sample @ ones)
    boot_weights.append(w)




# Part 2: Price Forecasting with LSTM

# 1. Sequence Preparation

from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, feature_col='Price', window_size=5):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[[feature_col]])
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
    return np.array(X), np.array(y), scaler

# 2. Model Training

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=50, batch_size=4, verbose=1)

# 3. Forecasting

last_sequence = X[-1]
predictions = []
for _ in range(60):
    pred = model.predict(last_sequence.reshape(1, 5, 1), verbose=0)
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0]).reshape(5, 1)
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


## Results

Optimal Portfolio Weights
Stock Weight Interpretation
1，Tesla ~25% Moderate long position

2，BYD ~71% Strong long position

3，NIO ~7.8% Small hedge

4，XPeng ~-4.7% Short position for hedging

## Visual Samples

Bootstrapped Weight Distributions (Custom colors for each stock — Tesla: darkgreen, BYD: lightgreen, NIO: yellowgreen, XPeng: purple)

LSTM Forecast Plots

60-day price prediction for each stock
Historical prices plotted in black for contrast
Tools & Libraries
Python: pandas, numpy, matplotlib

Machine Learning: tensorflow / keras

Data Processing: scikit-learn

Visualization: Custom color coding

## Future Improvements

Add Sharpe ratio & efficient frontier analysis

Explore multivariate LSTM for joint stock predictions
