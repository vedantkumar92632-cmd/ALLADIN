from typing import List
from input.market_data.ohlcv import OHLCVBar
from core.logger import get_logger

logger = get_logger("data_engine.compute")

def compute_bars(bars: List[OHLCVBar]) -> List[OHLCVBar]:
    """
    OHLCV aggregation logic.
    
    For now, this is a thin pass-through returning the bars as-is, since yfinance
    already downloads pre-aggregated EOD OHLCV bars.
    
    If a tick-level data source is added later, this module should be expanded
    to aggregate raw tick trade events into custom time, volume, or dollar bars.
    """
    logger.debug(f"Passing through {len(bars)} pre-aggregated bars.")
    return bars
