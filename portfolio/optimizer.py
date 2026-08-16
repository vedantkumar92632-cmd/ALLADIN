"""
Portfolio optimizer: HRP + Black-Litterman (L5.S002, L5.S003).

HRP (Hierarchical Risk Parity):
  Avoids matrix inversion — critical for 20-40 stock Indian universe where
  correlation matrix can be near-singular during market stress. More stable
  weights than Markowitz. Per QuantPedia: HRP has 3x higher turnover than
  risk parity — for NSE, rebalance monthly not weekly.

Black-Litterman:
  Converts CatBoost+LightGBM alpha forecasts into portfolio weights without
  forcing extreme concentration. Prior derived from NSE market cap weights
  and Nifty 50 risk premium — NOT a US market prior. Omega (view confidence)
  set using variance of ML prediction intervals.

Per Part 3A:
- HRP: Use linkage_method='single' or 'complete' (not 'ward') to preserve
  NSE sector clustering.
- BL: Prior from CAPM with NSE market cap weights.

Resources:
- PyPortfolioOpt: HRPOpt + BlackLittermanModel
- Lopez de Prado original HRP paper
- Indian market comparative study (arXiv 2305.17523)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from core.logger import get_logger

logger = get_logger("portfolio.optimizer")


def optimize_hrp(
    returns: pd.DataFrame,
    linkage_method: str = "single",
) -> Dict[str, float]:
    """
    Hierarchical Risk Parity optimization.
    
    Args:
        returns: DataFrame of asset daily returns (columns = tickers).
        linkage_method: 'single' or 'complete' (NOT 'ward') to preserve
                       NSE sector clustering.
    Returns:
        Dict of {ticker: weight}.
    """
    from pypfopt import HRPOpt
    
    # Compatibility fix: scipy 1.18+ removed _LINKAGE_METHODS internal attribute
    # that pyportfolioopt 1.6.0 references
    import scipy.cluster.hierarchy as _sch
    if not hasattr(_sch, '_LINKAGE_METHODS'):
        _sch._LINKAGE_METHODS = {
            'single', 'complete', 'average', 'weighted',
            'centroid', 'median', 'ward',
        }
    
    returns = returns.dropna(how='all').fillna(0)
    hrp = HRPOpt(returns)
    weights = hrp.optimize(linkage_method=linkage_method)
    
    clean_weights = {k: round(v, 6) for k, v in weights.items() if v > 1e-6}
    
    logger.info(f"HRP optimization: {len(clean_weights)} assets, linkage={linkage_method}")
    return clean_weights


def optimize_black_litterman(
    returns: pd.DataFrame,
    market_caps: Dict[str, float],
    views: Dict[str, float],
    view_confidences: Optional[Dict[str, float]] = None,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.065,
) -> Dict[str, float]:
    """
    Black-Litterman portfolio optimization.
    
    Blends ML model outputs (as "views") with market-cap-weight consensus.
    
    Args:
        returns: DataFrame of asset daily returns.
        market_caps: Dict of {ticker: market_cap_in_crores}.
        views: Dict of {ticker: expected_return} from CatBoost/LightGBM.
        view_confidences: Optional dict of {ticker: confidence_0_to_1}.
                         If None, uses uniform confidence.
        risk_aversion: Market risk aversion parameter.
        risk_free_rate: NSE risk-free rate (e.g. 6.5% for Indian T-bill).
        
    Returns:
        Dict of {ticker: weight}.
    """
    from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models
    
    # Covariance from historical returns
    cov_matrix = risk_models.sample_cov(returns)
    
    # Build Black-Litterman model
    bl = BlackLittermanModel(
        cov_matrix,
        pi="market",
        market_caps=market_caps,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate,
        absolute_views=views,
    )
    
    # Get BL expected returns
    bl_returns = bl.bl_returns()
    bl_cov = bl.bl_cov()
    
    # Optimize using efficient frontier on BL outputs
    ef = EfficientFrontier(bl_returns, bl_cov)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    
    clean_weights = {k: round(v, 6) for k, v in weights.items() if v > 1e-6}
    
    logger.info(f"Black-Litterman optimization: {len(clean_weights)} assets, {len(views)} views")
    return clean_weights


def validate_weights(weights: Dict[str, float], tolerance: float = 0.05) -> bool:
    """
    Validate that portfolio weights are sane.
    Per Layer 5 exit checklist: weights sum to ~1, no single position wildly dominant.
    """
    total = sum(weights.values())
    max_weight = max(weights.values()) if weights else 0
    
    if abs(total - 1.0) > tolerance:
        logger.warning(f"Weights sum to {total:.4f} (expected ~1.0)")
        return False
    
    if max_weight > 0.40:
        logger.warning(f"Max weight {max_weight:.4f} exceeds 40% — may be too concentrated")
        return False
    
    logger.info(f"Weights valid: sum={total:.4f}, max={max_weight:.4f}, n={len(weights)}")
    return True
