import datetime
from zoneinfo import ZoneInfo
from typing import List
import pandas as pd
from input.market_data.ohlcv import OHLCVBar
from core.time_utils import IST, to_ist
from core.logger import get_logger

logger = get_logger("data_engine.ingest")

def ingest_raw_data(df: pd.DataFrame) -> List[OHLCVBar]:
    """
    Ingest and normalize raw OHLCV DataFrame from the data feed adapter.
    
    Converts timestamps to timezone-aware Asia/Kolkata datetimes at market close (15:30:00 IST)
    for daily data, validates the schema, and returns a list of validated OHLCVBar objects.
    """
    if df.empty:
        logger.warning("Empty DataFrame passed to ingestion.")
        return []
        
    validated_bars = []
    
    for idx, row in df.iterrows():
        try:
            raw_date = row['Date']
            
            # Handle conversion from pandas Timestamp, standard date, or datetime
            if isinstance(raw_date, pd.Timestamp):
                raw_date = raw_date.to_pydatetime()
                
            if isinstance(raw_date, datetime.date) and not isinstance(raw_date, datetime.datetime):
                dt_ist = datetime.datetime.combine(raw_date, datetime.time(15, 30, 0), tzinfo=IST)
            elif isinstance(raw_date, datetime.datetime):
                if raw_date.tzinfo is None:
                    # Naive datetime: assume IST and adjust midnight to 15:30:00 IST
                    if raw_date.time() == datetime.time(0, 0, 0):
                        dt_ist = datetime.datetime.combine(raw_date.date(), datetime.time(15, 30, 0), tzinfo=IST)
                    else:
                        dt_ist = raw_date.replace(tzinfo=IST)
                else:
                    # Aware datetime: convert to IST and adjust midnight to 15:30:00 IST
                    dt_ist = to_ist(raw_date)
                    if dt_ist.time() == datetime.time(0, 0, 0):
                        dt_ist = datetime.datetime.combine(dt_ist.date(), datetime.time(15, 30, 0), tzinfo=IST)
            else:
                logger.warning(f"Unsupported date format at row {idx}: {raw_date}")
                continue
            
            # Map fields according to Section 3 Part 4
            bar = OHLCVBar(
                ticker=str(row['Ticker']),
                timestamp=dt_ist,
                open_price=float(row['Open']),
                high_price=float(row['High']),
                low_price=float(row['Low']),
                close_price=float(row['Close']),
                volume=float(row['Volume']),
                vwap=float(row['VWAP']) if 'VWAP' in row and not pd.isna(row['VWAP']) else None,
                trade_count=int(row['TradeCount']) if 'TradeCount' in row and not pd.isna(row['TradeCount']) else None,
                adj_close=float(row['AdjClose']) if 'AdjClose' in row and not pd.isna(row['AdjClose']) else None
            )
            
            if bar.validate():
                validated_bars.append(bar)
            else:
                logger.warning(f"OHLCV bar validation failed for ticker {bar.ticker} at {bar.timestamp}")
                
        except Exception as e:
            logger.warning(f"Row {idx} ingestion failed with error: {e}")
            continue
            
    logger.info(f"Ingested and validated {len(validated_bars)} / {len(df)} rows.")
    return validated_bars
