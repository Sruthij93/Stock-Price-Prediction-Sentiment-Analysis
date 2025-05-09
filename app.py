from load_dotenv import load_dotenv
load_dotenv()

import pandas as pd
import os
import urllib.request, json
import datetime as dt
# from alpha_vantage.news import News
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from newsapi import NewsApiClient
import finnhub
from supabase import create_client
import asyncio
import aiohttp
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

url= os.environ.get("SUPABASE_URL")
key= os.environ.get("SUPABASE_KEY")
supabase= create_client(url, key)



API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
finnhub_client = finnhub.Client(api_key=finnhub_api_key)

# Model and tokenizer (FinBERT) for sentiment analysis
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# function to fetch stock data based on the symbol entered
async def get_stock_data(session, symbol):
    
    # create JSON file with all the stock market data for AAPL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(symbol,API_KEY)  

    async with session.get(url_string) as response: 
        if response.status != 200:
            raise Exception(f"Error fetching stock data for {symbol}: {response.status}")
        
        # get the response body
        data = await response.text()
        data = json.loads(data)
        # key = "Time Series (Daily)" only needs to be used
        data = data['Time Series (Daily)']

        # create a pandas dataframe with the data
        df = pd.DataFrame.from_dict(data, orient = 'index')
        df.reset_index(inplace = True)
        df.rename(columns={'index': 'date', '1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'}, inplace=True)
        
        
        df['date'] = pd.to_datetime(df['date'])
        
        # get past 2 years data
        two_years_ago = dt.datetime.today() - dt.timedelta(days=2*365)
        df = df[df['date'] > two_years_ago]

        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(int)

        # Sort the dataframe by date
        df.sort_values(by='date', ascending=True, inplace=True)

        df['ticker'] = symbol.upper()


        # print(df.head(5))

        return df
    

# function to insert stock data into the Supabase database
async def insert_stock_data(df):
    
    if df.empty:
        print("No data to insert.")
        return
    else:
        df = df.copy()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # convert data to dict to insert into the database
    data_dict = df.to_dict(orient='records')
    try:
        response = supabase.table('stock_prices').upsert(data_dict).execute()
    except Exception as e:
        print(f"Error inserting stock data: {str(e)}")


async def save_stock_data(symbol, df):
    """Update stock data in the Supabase database"""
    # check if df is empty
    if df.empty:
        print(f"No data to insert for {symbol}.")
        return
    
    # convert the df columns to str type
    merged_df = df.copy() 
    # merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
    merged_df['date'] = merged_df['date'].astype(str)

    # fill nan values with 0
    merged_df.fillna(0, inplace=True)
    merged_df['num_articles'] = merged_df['num_articles'].fillna(0).astype(int)

    # Select only columns that exist in the table
    columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 
               'avg_sentiment', 'num_articles', 'sentiment_ma3', 'sentiment_lag1',
               'log_article_count', 'sentiment_volatility', 'sentiment_close_corr']
    
    # Filter to only columns that exist in the DataFrame
    existing_columns = [col for col in columns if col in merged_df.columns]
    if not existing_columns:
        print("No matching columns found in merged_df")
        return
    
    data_dict = merged_df[existing_columns].to_dict(orient='records')

    supabase.table('stock_prices').upsert(data_dict).execute()
    print(f"Data inserted successfully for {symbol}")

async def save_news_data(symbol, news_df):
    """Save news data to Supabase"""
    if news_df.empty:
        print(f"No news data to insert for {symbol}.")
        return

    # Convert the 'datetime' column to string format
    # news_df['date'] = news_df['date'].dt.strftime('%Y-%m-%d')

    # rename date column to published_date
    news_df.rename(columns={'date': 'published_date', 'summary': 'content', 'sentiment_score': 'sentiment'}, inplace=True)
    news_df['published_date'] = news_df['published_date'].astype(str)
    # filter out the columns needed for the database
    existing_columns = ['id', 'ticker', 'published_date', 'headline', 'content', 'sentiment']

    # Convert data to dict to insert into the database
    data_dict = news_df[existing_columns].to_dict(orient='records')

    supabase.table('news_articles').upsert(data_dict).execute()
    print(f"News data inserted successfully for {symbol}")


async def save_to_supabase(symbol, news_df, merged_df):
    """Save all dataframes to Supabase"""
    # Save news articles
    await save_news_data(symbol, news_df)
    
    # Save merged stock data with sentiment
    await save_stock_data(symbol, merged_df)
    
    print("Data saved to Supabase successfully")
    
        

# function to fetch news for a date chunk asynchronously
async def fetch_news_chunk(session, symbol, from_str, to_str):
    
    url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_str}&to={to_str}&token={finnhub_api_key}'
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.json()
            print(f"Error {response.status} for {from_str} to {to_str} for {symbol}")
            return []
    except Exception as e:
        print(f"Request failed: {str(e)}")
        return []

# function that calls fetch_news_chunk asynchronously 
async def fetch_full_year_news_async(session, symbol):
    
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365)
    all_articles = []
    
    
    # Generate date chunks
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + dt.timedelta(days=5), end_date)
        from_str = current_start.strftime('%Y-%m-%d')
        to_str = current_end.strftime('%Y-%m-%d')
        # date_chunks.append((from_str, to_str))
    
        chunk = await fetch_news_chunk(session, symbol, from_str, to_str)
        
        if isinstance(chunk, list):
            all_articles.extend(chunk)
            print(f"{symbol}: Got {len(chunk)} articles from {from_str} to {to_str}")
            if len(chunk) >= 1000:
                print(f"Warning: Possible data truncation in chunk")
        
        current_start = current_end + dt.timedelta(days=1)

    df = pd.DataFrame(all_articles).drop_duplicates('id')
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df[df['datetime'].between('1970-01-01', '2262-04-11')]
    df['ticker'] = symbol.upper()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    # df = df.groupby(['ticker', 'date']).head(10).reset_index(drop=True)
    # print(df.head(3))
    # print(df.iloc[0]['headline'])
    df = df[['ticker', 'id', 'datetime', 'headline', 'url', 'summary']].sort_values('datetime')
    return df

# function to analyze sentiment in batches
def analyze_sentiment_batch(text_list):
    
    # try:
    print("Analyzing sentiment in batches")
    results = sentiment_pipeline(text_list, padding = True, truncation=True, batch_size=16)
    scores = [
        r['score'] if r['label'] == 'positive' else -r['score']
        for r in results
    ]
    return scores
    # except Exception as e:
    #     print(f"Sentiment batch error: {str(e)}")
    #     return [0.0 for _ in text_list]

async def process_news_sentiment(news_df):
    """Analyze sentiment in batches"""
    if news_df.empty:
        return []
    
    texts = (news_df['headline'].fillna('') + '. ' + news_df['summary'].fillna('')).tolist()
    scores = analyze_sentiment_batch(texts)
    return scores

        
async def process_ticker(symbol):
    async with aiohttp.ClientSession() as session:
        # fetch stock and news concurrently
        stock_task = get_stock_data(session, symbol)
        news_task = fetch_full_year_news_async(session, symbol)
        stock_df, news_df = await asyncio.gather(stock_task, news_task)

        # run sentiment analysis
        sentiments = await process_news_sentiment(news_df)

        # Add sentiments back to news_df
        news_df['sentiment_score'] = sentiments

        # group daily sentiments and calculate avg sentiment for the day
        news_df['date'] = pd.to_datetime(news_df['datetime']).dt.date
        
        daily_sentiment = news_df.groupby(['ticker', 'date'])['sentiment_score'].mean().reset_index()

        # add a column for number of articles per day
        daily_sentiment['num_articles'] = news_df.groupby(['ticker', 'date'])['date'].count().values

        daily_sentiment.rename(columns={'sentiment_score': 'avg_sentiment'}, inplace=True)

        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date

        # merge stock prices with daily average sentiment
        merged_df = pd.merge(stock_df, daily_sentiment, on=['ticker', 'date'], how='left')
        merged_df['avg_sentiment']= merged_df['avg_sentiment'].fillna(0.0)  # Fill missing sentiment as neutral
        merged_df['num_articles'] = merged_df['num_articles'].fillna(0)

        # Adding more features related to sentiment (lagged sentiment, moving averages, etc.)
        # Moving average
        merged_df['sentiment_ma3'] = merged_df['avg_sentiment'].rolling(window=3).mean()
        # lagged sentiment to capture delayed effects/reaction
        merged_df['sentiment_lag1'] = merged_df['avg_sentiment'].shift(1)
        # news volume
        merged_df['log_article_count'] = np.log(merged_df['num_articles'] + 1)  # Adding 1 to avoid log(0)
        # sentiment volatility- measure of how much sentiment varies over time
        merged_df['sentiment_volatility'] = merged_df['avg_sentiment'].rolling(window=7).std()
        # sentiment-closing price correlation
        merged_df['sentiment_close_corr'] = merged_df['avg_sentiment'].rolling(5).corr(merged_df['close'])


        return stock_df, news_df, merged_df


async def main():
    tickers = ['AAPL']
    tasks = [process_ticker(t) for t in tickers]
    results = await asyncio.gather(*tasks)

    

    for symbol, (stock_df, news_df, merged_df) in zip(tickers, results):
        print(f"{symbol}: {len(stock_df)} stock rows, {len(news_df)} news rows")

        # Save data to Supabase
        # await save_to_supabase(symbol, news_df, merged_df)
        
        print(news_df)
        # ✅ Check if sentiment scores are working
        print(f"Sample merged data for {symbol}:")
        print(merged_df)

        # save to csv
        merged_df.to_csv(f"{symbol}_merged_5.csv", index=False)

if __name__ == "__main__":
    asyncio.run(main())
