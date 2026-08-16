"""
Half-Kelly position sizing (L4.S002).

Per Part 3A: Full Kelly is mathematically optimal but practically causes
catastrophic drawdowns. Half-Kelly is the accepted institutional compromise.

Failure modes to avoid:
1. Using backtest win rate instead of live-calibrated forward edge — always
   overstates Kelly size. Apply 30% shrinkage (Thorp's recommendation).
2. Ignoring trade correlation — NSE stocks are highly correlated during
   market stress, effectively increasing position size beyond what half-Kelly
   implies for a single stock.

Cap at 5% of portfolio per position regardless of Kelly output.

Note: Position sizing belongs in portfolio/constructor, NOT in risk/
(per Part 5 folder split).
"""
import numpy as np
from typing import Optional
from core.logger import get_logger

logger = get_logger("portfolio.constructor")

# Absolute cap per Part 3A: never more than 5% per position
MAX_POSITION_PCT = 0.05


def half_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    shrinkage: float = 0.70,
    max_pct: float = MAX_POSITION_PCT,
) -> float:
    """
    Compute Half-Kelly position size as a fraction of portfolio.
    
    Per Part 3A:
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        odds = avg_win / avg_loss
        f* = edge / (odds * avg_loss)
        f_half_kelly = 0.5 * f* * shrinkage
    
    Args:
        win_rate: Historical win rate (0 to 1). Should be out-of-sample.
        avg_win: Average winning trade return (positive, e.g. 0.02 = 2%).
        avg_loss: Average losing trade return (positive magnitude, e.g. 0.015).
        shrinkage: Shrinkage factor (0.7 = 30% haircut for backtest overfitting).
        max_pct: Maximum position size as fraction of portfolio.
        
    Returns:
        Position size as fraction of portfolio (0 to max_pct).
    """
    if avg_loss <= 0 or avg_win <= 0:
        logger.warning("Invalid avg_win or avg_loss — returning 0.")
        return 0.0
    
    if not 0 < win_rate < 1:
        logger.warning(f"Win rate {win_rate} out of (0,1) — returning 0.")
        return 0.0
    
    # Expected edge per trade
    edge = win_rate * avg_win - (1 - win_rate) * avg_loss
    
    if edge <= 0:
        logger.info(f"Negative edge ({edge:.4f}) — no position warranted.")
        return 0.0
    
    # Kelly fraction
    odds = avg_win / avg_loss
    f_star = edge / (odds * avg_loss)
    
    # Half-Kelly with shrinkage
    f_half = 0.5 * f_star * shrinkage
    
    # Cap at maximum
    f_final = min(max(f_half, 0.0), max_pct)
    
    logger.info(
        f"Half-Kelly: edge={edge:.4f}, odds={odds:.2f}, "
        f"f*={f_star:.4f}, f_half={f_half:.4f}, capped={f_final:.4f}"
    )
    return f_final


def position_size_shares(
    kelly_fraction: float,
    portfolio_value: float,
    share_price: float,
) -> int:
    """
    Convert Kelly fraction to number of shares.
    
    Args:
        kelly_fraction: Output of half_kelly().
        portfolio_value: Total portfolio NAV.
        share_price: Current share price.
        
    Returns:
        Number of shares to buy (integer, rounded down).
    """
    if share_price <= 0 or kelly_fraction <= 0:
        return 0
    
    position_value = kelly_fraction * portfolio_value
    shares = int(position_value / share_price)
    return shares
