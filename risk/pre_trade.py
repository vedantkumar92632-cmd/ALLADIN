"""
ATR-based dynamic stop-loss calculation (L4.S004).

Adapts stop distance to current volatility rather than using a fixed percentage.
Directly prevents stop-hunting in high-volatility regimes.

Per Part 3A:
- Period: 14 days (standard) or 20 days for less reactive stops
- Multiplier: 2.0-3.0x ATR for NSE stocks (wider than US due to higher vol)
- NSE has circuit breakers at +/-10%/+/-20% — factor in when setting stops

Resources:
- QuantInsti: ATR-based stop-loss implementation, Indian market examples
"""
import numpy as np
import pandas as pd
from typing import Optional
from core.logger import get_logger

logger = get_logger("risk.pre_trade")


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = rolling mean of True Range over `period` days.
    
    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ATR period (14 standard, 20 for less reactive).
        
    Returns:
        Series of ATR values.
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr


def atr_stop_distance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    multiplier: float = 2.5,
    period: int = 14,
) -> pd.Series:
    """
    Compute ATR-based stop distance per position.
    
    Stop distance = multiplier * ATR.
    For NSE stocks, use multiplier 2.0-3.0 (wider than US due to higher vol).
    
    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        multiplier: ATR multiplier (2.0-3.0 for NSE). Default 2.5.
        period: ATR period.
        
    Returns:
        Series of stop distances in price units.
    """
    atr = compute_atr(high, low, close, period)
    stop_dist = multiplier * atr
    return stop_dist


def atr_stop_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    direction: str = "LONG",
    multiplier: float = 2.5,
    period: int = 14,
) -> pd.Series:
    """
    Compute ATR-based stop price.
    
    LONG: stop = close - multiplier * ATR
    SHORT: stop = close + multiplier * ATR
    
    NSE circuit breakers at +/-10%/+/-20% are factored in:
    stop price is clamped to not exceed circuit breaker limits.
    """
    stop_dist = atr_stop_distance(high, low, close, multiplier, period)
    
    if direction.upper() == "LONG":
        stop = close - stop_dist
        # NSE lower circuit = close * 0.80 (20% circuit breaker)
        lower_circuit = close * 0.80
        stop = stop.clip(lower=lower_circuit)
    else:
        stop = close + stop_dist
        # NSE upper circuit = close * 1.20
        upper_circuit = close * 1.20
        stop = stop.clip(upper=upper_circuit)
    
    return stop


def pre_trade_check(
    current_price: float,
    atr_value: float,
    multiplier: float = 2.5,
    max_position_size_pct: float = 0.05,
    portfolio_value: float = 100000.0,
) -> dict:
    """
    Pre-trade risk check for a single position.
    
    Returns:
        Dict with stop_price, stop_distance, max_shares, risk_per_share.
    """
    if current_price <= 0:
        logger.error('Invalid price')
        return {'approved': False, 'stop_distance': 0.0, 'stop_price_long': 0.0, 'stop_price_short': 0.0, 'risk_per_share': 0.0, 'max_shares': 0, 'max_loss': 0.0, 'max_loss_pct': 0.0}

    stop_distance = multiplier * atr_value
    risk_per_share = stop_distance
    
    # Max position value
    max_position_value = portfolio_value * max_position_size_pct
    max_shares = int(max_position_value / current_price)
    
    # Max loss if stopped out
    max_loss = max_shares * risk_per_share
    
    result = {
        'stop_distance': stop_distance,
        'stop_price_long': current_price - stop_distance,
        'stop_price_short': current_price + stop_distance,
        'risk_per_share': risk_per_share,
        'max_shares': max_shares,
        'max_loss': max_loss,
        'max_loss_pct': max_loss / portfolio_value if portfolio_value > 0 else 0,
    }
    
    logger.info(
        f"Pre-trade: price={current_price:.2f}, ATR={atr_value:.2f}, "
        f"stop_dist={stop_distance:.2f}, max_shares={max_shares}"
    )
    return result


class TrailingATRStop:
    """
    Trailing ATR-based stop-loss manager (L4.S004 / P2).
    Tracks peak price achieved for LONG (or trough for SHORT) and moves stop monotonically.
    Never moves stop loss level backwards.
    """
    def __init__(self, multiplier: float = 2.5):
        self.multiplier = multiplier
        self._peak_prices = {}
        self._stop_levels = {}

    def update_and_check(
        self,
        ticker: str,
        current_price: float,
        atr_value: float,
        direction: str = "LONG",
    ):
        """
        Update peak price and recalculate trailing stop level.
        
        Returns:
            Tuple of (current_stop_level: float, is_stopped_out: bool).
        """
        if atr_value <= 0 or current_price <= 0:
            return self._stop_levels.get(ticker, 0.0), False

        stop_dist = self.multiplier * atr_value

        if ticker not in self._peak_prices:
            self._peak_prices[ticker] = current_price
            initial_stop = current_price - stop_dist if direction.upper() == "LONG" else current_price + stop_dist
            self._stop_levels[ticker] = initial_stop
            return initial_stop, False

        if direction.upper() == "LONG":
            if current_price > self._peak_prices[ticker]:
                self._peak_prices[ticker] = current_price
                new_stop = current_price - stop_dist
                self._stop_levels[ticker] = max(self._stop_levels[ticker], new_stop)
            is_stopped = current_price <= self._stop_levels[ticker]
        else:
            if current_price < self._peak_prices[ticker]:
                self._peak_prices[ticker] = current_price
                new_stop = current_price + stop_dist
                self._stop_levels[ticker] = min(self._stop_levels[ticker], new_stop)
            is_stopped = current_price >= self._stop_levels[ticker]

        return self._stop_levels[ticker], is_stopped

    def reset(self, ticker: str):
        """Reset trailing stop state when a position is closed."""
        self._peak_prices.pop(ticker, None)
        self._stop_levels.pop(ticker, None)

