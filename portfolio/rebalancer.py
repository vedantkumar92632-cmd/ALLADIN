"""
Portfolio rebalancer (L5.S004).

Decides when to actually trade the gap between current and target weights.
Don't rebalance on every tiny drift — use a buffer system.

Per Part 3A/Part 6:
- Adapted from pysystemtrade's buffer concept (A5/D2) — the *idea*, not
  the code, since pysystemtrade is a futures system and this is equities.
- For NSE: monthly rebalancing (HRP has 3x higher turnover than risk parity,
  weekly rebalancing generates excessive STT + brokerage costs).

Buffer logic:
- Only rebalance a position if weight drift exceeds buffer threshold
- Buffer = max(min_buffer, target_weight * relative_buffer)
- This prevents excessive trading on small positions
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from core.logger import get_logger

logger = get_logger("portfolio.rebalancer")

# Default buffer parameters
DEFAULT_MIN_BUFFER = 0.005       # 0.5% absolute minimum buffer
DEFAULT_RELATIVE_BUFFER = 0.20   # 20% of target weight as buffer


def compute_rebalance_trades(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    prices: Dict[str, float],
    min_buffer: float = DEFAULT_MIN_BUFFER,
    relative_buffer: float = DEFAULT_RELATIVE_BUFFER,
    min_trade_value: float = 1000.0,
) -> List[Dict]:
    """
    Compute trades needed to rebalance, applying buffer logic.
    
    Only generates trades for positions where weight drift exceeds buffer.
    
    Args:
        current_weights: Dict of {ticker: current_weight}.
        target_weights: Dict of {ticker: target_weight}.
        portfolio_value: Total portfolio NAV.
        prices: Dict of {ticker: current_price}.
        min_buffer: Minimum absolute weight buffer (default 0.5%).
        relative_buffer: Buffer as fraction of target weight (default 20%).
        min_trade_value: Minimum trade value in INR to execute.
        
    Returns:
        List of trade dicts with keys: ticker, action, shares, value, reason.
    """
    trades = []
    all_tickers = set(list(current_weights.keys()) + list(target_weights.keys()))
    
    for ticker in all_tickers:
        current_w = current_weights.get(ticker, 0.0)
        target_w = target_weights.get(ticker, 0.0)
        drift = target_w - current_w
        
        # Compute buffer for this position
        buffer = max(min_buffer, target_w * relative_buffer)
        
        if abs(drift) <= buffer:
            continue  # Within buffer — no trade needed
        
        # Compute trade
        price = prices.get(ticker, 0.0)
        if price <= 0:
            logger.warning(f"No valid price for {ticker} — skipping.")
            continue
        
        trade_value = abs(drift) * portfolio_value
        if trade_value < min_trade_value:
            continue  # Too small to be worth trading
        
        shares = int(trade_value / price)
        if shares == 0:
            continue
        
        action = "BUY" if drift > 0 else "SELL"
        
        trades.append({
            'ticker': ticker,
            'action': action,
            'shares': shares,
            'value': shares * price,
            'current_weight': current_w,
            'target_weight': target_w,
            'drift': drift,
            'buffer': buffer,
            'reason': f"drift {drift:+.4f} exceeds buffer {buffer:.4f}",
        })
    
    logger.info(
        f"Rebalancer: {len(trades)} trades needed out of {len(all_tickers)} positions"
    )
    return trades


def should_rebalance(
    last_rebalance_date: str,
    current_date: str,
    frequency_days: int = 20,
) -> bool:
    """
    Check if it's time to rebalance (monthly = 20 trading days for NSE).
    
    Args:
        last_rebalance_date: ISO date string of last rebalance.
        current_date: ISO date string of current date.
        frequency_days: Rebalance frequency in trading days.
        
    Returns:
        True if enough time has passed since last rebalance.
    """
    from datetime import datetime
    
    last = datetime.fromisoformat(last_rebalance_date)
    current = datetime.fromisoformat(current_date)
    
    days_elapsed = (current - last).days
    
    # Approximate trading days (5/7 of calendar days)
    trading_days_elapsed = int(days_elapsed * 5 / 7)
    
    return trading_days_elapsed >= frequency_days
