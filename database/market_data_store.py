import sqlite3
from typing import List, Optional
from input.market_data.ohlcv import OHLCVBar
from database.connection import get_connection
from core.logger import get_logger

logger = get_logger("database.market_data_store")

def create_ohlcv_table():
    """Create the ohlcv_bars table in market.db if it doesn't exist."""
    conn = get_connection("market")
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                vwap REAL,
                trade_count INTEGER,
                PRIMARY KEY (ticker, timestamp)
            );
        """)
        conn.commit()
        logger.debug("Checked/created ohlcv_bars table in market.db.")
    except sqlite3.Error as e:
        logger.error(f"Failed to create ohlcv_bars table: {e}")
        raise
    finally:
        conn.close()

def save_ohlcv_bars(bars: List[OHLCVBar]):
    """Insert or replace OHLCV bars in market.db."""
    if not bars:
        return
        
    create_ohlcv_table()
    
    conn = get_connection("market")
    try:
        cursor = conn.cursor()
        # Use INSERT OR REPLACE to overwrite existing data for the same ticker/timestamp
        cursor.executemany("""
            INSERT OR REPLACE INTO ohlcv_bars 
            (ticker, timestamp, open, high, low, close, volume, vwap, trade_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            (
                bar.ticker,
                bar.timestamp.isoformat(),
                bar.open_price,
                bar.high_price,
                bar.low_price,
                bar.close_price,
                bar.volume,
                bar.vwap,
                bar.trade_count
            ) for bar in bars
        ])
        conn.commit()
        logger.info(f"Saved {len(bars)} OHLCV bars to market.db.")
    except sqlite3.Error as e:
        logger.error(f"Failed to save OHLCV bars to database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def load_ohlcv_bars(ticker: str, start_date: str, end_date: str) -> List[dict]:
    """Load OHLCV bars from market.db for a given ticker and ISO date range."""
    # Convert date strings to start and end ISO format timestamps
    start_ts = f"{start_date}T00:00:00"
    end_ts = f"{end_date}T23:59:59"
    
    conn = get_connection("market")
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, timestamp, open, high, low, close, volume, vwap, trade_count
            FROM ohlcv_bars
            WHERE ticker = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC;
        """, (ticker, start_ts, end_ts))
        
        rows = cursor.fetchall()
        bars = []
        for r in rows:
            bars.append({
                'ticker': r[0],
                'timestamp': r[1],
                'open': r[2],
                'high': r[3],
                'low': r[4],
                'close': r[5],
                'volume': r[6],
                'vwap': r[7],
                'trade_count': r[8]
            })
        return bars
    except sqlite3.Error as e:
        logger.error(f"Failed to load OHLCV bars from database: {e}")
        raise
    finally:
        conn.close()
