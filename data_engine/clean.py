import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
from input.market_data.ohlcv import OHLCVBar
from core.time_utils import is_trading_day, IST
from core.logger import get_logger

logger = get_logger("data_engine.clean")

def clean_ohlcv_data(bars: List[OHLCVBar]) -> List[OHLCVBar]:
    """
    Clean the ingested OHLCV bars by performing:
    1. Deduplication by ticker and timestamp
    2. Corporate Action Adjustment (using adj_close/close ratio)
    3. Outlier Rejection (reverting price spike filter)
    4. Gap Filling (forward filling missing trading days)
    
    Returns a clean, sorted, gap-filled list of OHLCVBar objects.
    """
    if not bars:
        return []

    # 1. Convert to DataFrame for vector operations
    records = []
    for bar in bars:
        records.append({
            'ticker': bar.ticker,
            'timestamp': bar.timestamp,
            'open': bar.open_price,
            'high': bar.high_price,
            'low': bar.low_price,
            'close': bar.close_price,
            'volume': bar.volume,
            'vwap': bar.vwap,
            'trade_count': bar.trade_count,
            'adj_close': bar.adj_close
        })
    df = pd.DataFrame(records)

    # Deduplicate by sorting and keeping last duplicate
    df.sort_values(by=['ticker', 'timestamp'], inplace=True)
    df.drop_duplicates(subset=['ticker', 'timestamp'], keep='last', inplace=True)

    # 2. Corporate Action Adjustment
    adjusted_count = 0
    for idx, row in df.iterrows():
        close = row['close']
        adj_close = row['adj_close']
        if adj_close is not None and not pd.isna(adj_close) and close > 0:
            ratio = adj_close / close
            # If the ratio deviates significantly from 1.0, adjust prices and volume
            if not np.isclose(ratio, 1.0, atol=1e-4):
                df.at[idx, 'open'] = row['open'] * ratio
                df.at[idx, 'high'] = row['high'] * ratio
                df.at[idx, 'low'] = row['low'] * ratio
                df.at[idx, 'close'] = adj_close
                # Scale volume inversely to keep trade value constant
                df.at[idx, 'volume'] = row['volume'] / ratio
                adjusted_count += 1
                
    if adjusted_count > 0:
        logger.info(f"Adjusted {adjusted_count} bars for corporate actions.")

    # 3 & 4. Outlier Rejection & Gap Filling per ticker
    cleaned_dfs = []
    
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values(by='timestamp').copy()
        
        # Outlier Rejection (reverting price spike check)
        if len(group) >= 3:
            close_prices = group['close'].values.copy()
            for i in range(1, len(close_prices) - 1):
                ret_t = (close_prices[i] / close_prices[i-1]) - 1.0
                ret_next = (close_prices[i+1] / close_prices[i]) - 1.0
                
                # Check for revert spike: return > 50% followed by return < -30% (or vice versa)
                if (ret_t > 0.50 and ret_next < -0.30) or (ret_t < -0.50 and ret_next > 0.50):
                    logger.warning(f"Outlier spike detected for {ticker} at {group.iloc[i]['timestamp']}. Replacing with previous close.")
                    # Replace current day's price with previous day's close
                    group.iloc[i, group.columns.get_loc('open')] = close_prices[i-1]
                    group.iloc[i, group.columns.get_loc('high')] = close_prices[i-1]
                    group.iloc[i, group.columns.get_loc('low')] = close_prices[i-1]
                    group.iloc[i, group.columns.get_loc('close')] = close_prices[i-1]
                    group.iloc[i, group.columns.get_loc('volume')] = 0.0
                    close_prices[i] = close_prices[i-1]

        # Gap Filling
        min_date = group['timestamp'].min().date()
        max_date = group['timestamp'].max().date()
        
        # Generate expected trading days range
        expected_dates = []
        curr = min_date
        while curr <= max_date:
            if is_trading_day(curr):
                expected_dates.append(curr)
            curr += datetime.timedelta(days=1)
            
        expected_timestamps = [datetime.datetime.combine(d, datetime.time(15, 30, 0), tzinfo=IST) for d in expected_dates]
        
        # Reindex to insert missing trading days
        group.set_index('timestamp', inplace=True)
        group = group.reindex(expected_timestamps)
        group.index.name = 'timestamp'
        group.reset_index(inplace=True)
        
        # Fill missing values
        group['ticker'] = group['ticker'].ffill().bfill()
        
        gap_mask = group['close'].isna()
        if gap_mask.any():
            group['close'] = group['close'].ffill().bfill()
            
            # For filled gaps, Open/High/Low are set to Close, and volume is 0
            group.loc[gap_mask, 'open'] = group.loc[gap_mask, 'close']
            group.loc[gap_mask, 'high'] = group.loc[gap_mask, 'close']
            group.loc[gap_mask, 'low'] = group.loc[gap_mask, 'close']
            group.loc[gap_mask, 'volume'] = 0.0
            group.loc[gap_mask, 'trade_count'] = 0
            
            logger.info(f"Filled {gap_mask.sum()} gaps for {ticker}.")
            
        cleaned_dfs.append(group)
        
    if cleaned_dfs:
        final_df = pd.concat(cleaned_dfs, ignore_index=True)
    else:
        final_df = df

    # Convert back to List[OHLCVBar]
    cleaned_bars = []
    for _, row in final_df.iterrows():
        cleaned_bars.append(OHLCVBar(
            ticker=row['ticker'],
            timestamp=row['timestamp'],
            open_price=float(row['open']),
            high_price=float(row['high']),
            low_price=float(row['low']),
            close_price=float(row['close']),
            volume=float(row['volume']),
            vwap=float(row['vwap']) if not pd.isna(row['vwap']) else None,
            trade_count=int(row['trade_count']) if not pd.isna(row['trade_count']) else None,
            adj_close=float(row['adj_close']) if not pd.isna(row['adj_close']) else None
        ))
        
    return cleaned_bars
