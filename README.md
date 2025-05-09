# Stock Price Prediction Using Sentiment and LSTM

This project predicts short-term stock prices using historical time-series data and financial news sentiment. It combines traditional stock indicators with NLP-based sentiment scores and leverages LSTM neural networks to forecast stock prices for the next 5 days and 1 day.

---

## Project Overview

- **Input:** Ticker symbol (e.g., `AAPL`)
- **Output:** Predicted `close` prices for the next 5 days and 1 day
- **Model:** LSTM-based models (with and without sentiment features)
- **Sentiment Analysis:** FinBERT (finance-specific BERT model) - using Hugging Face pipeline
- **News Source:** Finnhub API (past 1-year company news)
- **Storage:** Supabase (PostgreSQL backend)
- **Processing:** Async Python pipeline for data ingestion and preprocessing

---

## Features & Pipeline

### Data Collection (`app.py`)

- Fetches 2 years of stock price data from Alpha Vantage
- Fetches 1 year of company news articles from Finnhub
- Calculates:
  - Average daily sentiment
  - Lagged sentiment
  - Sentiment volatility
  - Article count (log-transformed)
- Stores cleaned data in Supabase for efficient reuse

### Sentiment Analysis

- Uses FinBERT via Hugging Face Transformers
- Batch inference for speed and efficiency
- Aggregates daily sentiment scores

### Model Training (`model_training.ipynb`)

- Two LSTM models trained:
  - **Price-Only LSTM**: trained on OHLCV data
  - **Sentiment LSTM**: trained on price + engineered sentiment features
- 80/10/10 data split (train/val/test)- 5 day prediction
  = 80/20 data split (train/test) - 1 day prediction
- Uses EarlyStopping on validation MAE

---

## Evaluation

- Multi-step prediction: forecasts 5-day closing prices
- Metrics: MAE, MSE
- Visualizations: Actual vs Predicted prices, Model history (Loss and MAE)

---

## Tech Stack

- Python (Pandas, NumPy, Asyncio, Aiohttp)
- Hugging Face Transformers
- Keras/TensorFlow for LSTM modeling
- Supabase for data storage
- Alpha Vantage & Finnhub APIs

---

## File Structure

```
app.py               # Async data ingestion & sentiment pipeline
model_training.ipynb # LSTM model training and evaluation
.env                 # Stores API keys and Supabase credentials
```

---

## To Run

1. Set up `.env` file with your API and Supabase credentials
2. Run `app.py` to ingest and store data
3. Train models via `model_training.ipynb`
4. Use trained model to predict future prices from processed test data

---

## Future Steps

1. Process data for multiple stock tickers
2. Improve Model performance
3. Reduce latency issues during sentiment analysis
4. Create Webapp
