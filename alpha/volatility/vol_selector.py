"""
AIC-based volatility model selector.

Per Part 3B Domain 3: fit both EGARCH and GJR-GARCH per asset, keep whichever
wins by AIC. Indian-market research: EGARCH wins on BSE Sensex, GJR-GARCH wins
on NSE Nifty — the winner varies by asset, so this must be automated.

This is a standalone selection function, NOT buried inside either model file.
"""
import pandas as pd
from typing import Tuple
from core.logger import get_logger
from alpha.volatility.egarch_model import EGARCHModel
from alpha.volatility.gjrgarch_model import GJRGARCHModel
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.volatility.vol_selector")


def select_best_vol_model(
    returns: pd.Series,
    dist: str = "t"
) -> Tuple[BaseAlpha, str, float, float]:
    """
    Fit both EGARCH(1,1) and GJR-GARCH(1,1) on the given returns,
    select the winner by AIC (lower is better).
    
    Args:
        returns: Series of asset returns (percentage scale recommended).
        dist: Error distribution. Default 't' (Student-t). Never use 'normal'.
        
    Returns:
        Tuple of (winner_model, winner_name, egarch_aic, gjr_aic)
    """
    returns_df = pd.DataFrame({'returns': returns})
    
    # Fit EGARCH
    egarch = EGARCHModel(dist=dist)
    egarch_aic = float('inf')
    try:
        egarch.fit(returns_df)
        egarch_aic = egarch.aic
    except Exception as e:
        logger.warning(f"EGARCH fit failed: {e}")
    
    # Fit GJR-GARCH
    gjr = GJRGARCHModel(dist=dist)
    gjr_aic = float('inf')
    try:
        gjr.fit(returns_df)
        gjr_aic = gjr.aic
    except Exception as e:
        logger.warning(f"GJR-GARCH fit failed: {e}")
        
    if egarch_aic == float('inf') and gjr_aic == float('inf'):
        logger.error("Both vol models failed to fit.")
        return None, None, float('inf'), float('inf')
    
    # Select by AIC (lower is better)
    if egarch_aic < gjr_aic:
        winner = egarch
        winner_name = "EGARCH"
    else:
        winner = gjr
        winner_name = "GJR-GARCH"
    
    logger.info(
        f"Vol model selection: EGARCH AIC={egarch_aic:.2f}, GJR-GARCH AIC={gjr_aic:.2f} -> winner: {winner_name}"
    )
    
    return winner, winner_name, egarch_aic, gjr_aic
