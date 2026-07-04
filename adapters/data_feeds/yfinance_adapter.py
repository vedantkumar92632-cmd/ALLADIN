import datetime
from typing import List, Union
import pandas as pd
import yfinance as yf
from core.logger import get_logger

logger = get_logger("adapters.data_feeds.yfinance_adapter")

def fetch_ohlcv(
    tickers: Union[str, List[str]], 
    start_date: Union[str, datetime.date], 
    end_date: Union[str, datetime.date]
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance for the specified tickers and date range.
    For Indian markets, NSE tickers should have the '.NS' suffix (e.g. 'RELIANCE.NS').
    
    Returns a unified long-format DataFrame with columns:
    ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = list(tickers)

    # Format dates
    start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
    end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
    
    logger.info(f"Fetching OHLCV data from yfinance for {len(ticker_list)} tickers from {start_str} to {end_str}")
    
    try:
        # Download data
        df = yf.download(
            tickers=ticker_list,
            start=start_str,
            end=end_str,
            group_by="ticker",
            auto_adjust=False,  # Keep raw OHLCV
            prepost=False,
            threads=True
        )
        
        if df.empty:
            logger.warning(f"No data returned from yfinance for tickers: {ticker_list}")
            return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
        records = []
        
        # yfinance structure varies by ticker list size. Make it robust to MultiIndex structures.
        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            if isinstance(df.columns, pd.MultiIndex):
                # Try levels to find matching ticker level
                if ticker in df.columns.levels[0]:
                    df_ticker = df.xs(ticker, axis=1, level=0, drop_level=True)
                elif ticker in df.columns.levels[1]:
                    df_ticker = df.xs(ticker, axis=1, level=1, drop_level=True)
                else:
                    df_ticker = df
            else:
                df_ticker = df
                
            for idx, row in df_ticker.dropna(subset=['Close']).iterrows():
                records.append({
                    'Date': idx,
                    'Ticker': ticker,
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': float(row['Volume'])
                })
        else:
            # MultiIndex columns for multiple tickers
            for ticker in ticker_list:
                if ticker in df.columns.levels[0]:
                    df_ticker = df.xs(ticker, axis=1, level=0, drop_level=True)
                elif ticker in df.columns.levels[1]:
                    df_ticker = df.xs(ticker, axis=1, level=1, drop_level=True)
                else:
                    logger.warning(f"Ticker {ticker} not found in columns of downloaded data")
                    continue
                    
                for idx, row in df_ticker.dropna(subset=['Close']).iterrows():
                    records.append({
                        'Date': idx,
                        'Ticker': ticker,
                        'Open': float(row['Open']),
                        'High': float(row['High']),
                        'Low': float(row['Low']),
                        'Close': float(row['Close']),
                        'Volume': float(row['Volume'])
                    })
                    
        result_df = pd.DataFrame(records)
        
        if not result_df.empty:
            result_df['Date'] = pd.to_datetime(result_df['Date'])
            
        logger.info(f"Successfully fetched {len(result_df)} data rows.")
        return result_df
        
    except Exception as e:
        logger.error(f"Error fetching data from yfinance: {e}", exc_info=True)
        return pd.DataFrame(columns=['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'])
