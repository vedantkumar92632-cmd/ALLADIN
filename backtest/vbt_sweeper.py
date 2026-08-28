"""
VectorBT Parameter Sweeper (P3 / L7).

Fast R&D parameter sweep utility using vectorbt for rapid strategy exploration.
Does NOT replace the production event-driven backtest engine — strictly for R&D.
"""
import numpy as np
import pandas as pd
import vectorbt as vbt
from typing import Dict, Any, List, Tuple
from core.logger import get_logger

logger = get_logger("backtest.vbt_sweeper")


def sweep_ma_crossover(
    prices: pd.Series,
    fast_windows: List[int] = [10, 20, 30],
    slow_windows: List[int] = [50, 100, 200],
) -> pd.DataFrame:
    """
    Fast parameter sweep for moving average crossover parameters using vectorbt.
    
    Args:
        prices: Series of asset closing prices.
        fast_windows: List of fast MA windows to test.
        slow_windows: List of slow MA windows to test.
        
    Returns:
        DataFrame summarizing Sharpe ratios across parameter combinations.
    """
    fast_ma, slow_ma = vbt.MA.run_combs(
        prices,
        window=fast_windows + slow_windows,
        r=2,
        short_names=['fast', 'slow']
    )
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    portfolio = vbt.Portfolio.from_signals(prices, entries, exits, freq='1D')
    
    sharpes = portfolio.sharpe_ratio()
    returns = portfolio.total_return()
    drawdowns = portfolio.max_drawdown()
    
    results = pd.DataFrame({
        'sharpe_ratio': sharpes,
        'total_return': returns,
        'max_drawdown': drawdowns,
    })
    
    logger.info(f"VectorBT Sweep: Evaluated parameter combinations. Best Sharpe={sharpes.max():.2f}")
    return results
