"""
Filtering (L6.S004) — removes low-liquidity, high-spread, blacklisted,
near-earnings tickers per Part 5's exact list.
"""
from typing import Dict, List, Set, Optional
from core.logger import get_logger

logger = get_logger("decision.filtering")

# Default blacklist (can be loaded from config)
DEFAULT_BLACKLIST: Set[str] = set()

# Minimum average daily volume (in shares) to be tradeable
MIN_AVG_VOLUME = 100000

# Maximum bid-ask spread (bps) to be tradeable
MAX_SPREAD_BPS = 50


def filter_universe(
    ranked_tickers: list,
    volumes: Optional[Dict[str, float]] = None,
    spreads: Optional[Dict[str, float]] = None,
    blacklist: Optional[Set[str]] = None,
    earnings_dates: Optional[Dict[str, str]] = None,
    current_date: Optional[str] = None,
    earnings_buffer_days: int = 3,
    min_volume: float = MIN_AVG_VOLUME,
    max_spread_bps: float = MAX_SPREAD_BPS,
) -> list:
    """
    Filter ranked tickers, removing untradeable ones.
    
    Removes:
    1. Low liquidity (avg volume below threshold)
    2. High spread (bid-ask spread above threshold)
    3. Blacklisted tickers
    4. Near-earnings (within buffer_days of earnings)
    
    Args:
        ranked_tickers: List of dicts from ranking.rank_by_risk_adjusted().
        volumes: {ticker: avg_daily_volume}.
        spreads: {ticker: spread_in_bps}.
        blacklist: Set of blacklisted tickers.
        earnings_dates: {ticker: 'YYYY-MM-DD'} of next earnings.
        current_date: Current date as 'YYYY-MM-DD'.
        
    Returns:
        Filtered list (same format as input, with removal reasons logged).
    """
    bl = blacklist or DEFAULT_BLACKLIST
    filtered = []
    removed_count = 0
    
    for entry in ranked_tickers:
        ticker = entry['ticker']
        removed = False
        reason = ""
        
        # Check blacklist
        if ticker in bl:
            removed = True
            reason = "blacklisted"
        
        # Check volume
        if not removed and volumes and ticker in volumes:
            if volumes[ticker] < min_volume:
                removed = True
                reason = f"low_volume ({volumes[ticker]:.0f} < {min_volume:.0f})"
        
        # Check spread
        if not removed and spreads and ticker in spreads:
            if spreads[ticker] > max_spread_bps:
                removed = True
                reason = f"high_spread ({spreads[ticker]:.1f}bps > {max_spread_bps:.1f}bps)"
        
        # Check near-earnings
        if not removed and earnings_dates and current_date and ticker in earnings_dates:
            from datetime import datetime
            try:
                earn_date = datetime.fromisoformat(earnings_dates[ticker])
                curr_date = datetime.fromisoformat(current_date)
                days_to_earnings = abs((earn_date - curr_date).days)
                if days_to_earnings <= earnings_buffer_days:
                    removed = True
                    reason = f"near_earnings ({days_to_earnings}d away)"
            except ValueError as e:
                logger.warning(f'Failed to parse earnings date for {ticker}: {e}')
        
        if removed:
            removed_count += 1
            logger.debug(f"Filtered out {ticker}: {reason}")
        else:
            filtered.append(entry)
    
    logger.info(f"Filtering: {len(filtered)} tickers kept, {removed_count} removed.")
    return filtered
