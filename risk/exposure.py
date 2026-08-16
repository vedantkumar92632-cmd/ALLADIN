"""
Risk exposure calculations: Historical CVaR and Parametric VaR.

L4.S001 — Historical CVaR (Conditional Value at Risk)
  Strictly superior to plain VaR — tells you the AVERAGE loss in the worst
  X% of scenarios, not just the threshold. Use 99% for Indian markets
  (fatter tails than US equities). 252-day rolling window.

L4.S003 — Parametric VaR via EGARCH
  Forward-looking VaR using EGARCH conditional volatility. Nearly free since
  it reuses EGARCH output. When historical CVaR and parametric VaR diverge,
  it's an early warning signal.

Per Part 3A: CVaR should be computed on PORTFOLIO return distribution,
not individual stock returns.
"""
import numpy as np
import pandas as pd
from typing import Optional
from core.logger import get_logger

logger = get_logger("risk.exposure")


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.99,
    lookback: int = 252,
) -> float:
    """
    Compute Historical CVaR (Expected Shortfall) on portfolio returns.
    
    CVaR = average of returns below the VaR threshold.
    Use 99% confidence for Indian markets (fat tails).
    
    Args:
        returns: Series of portfolio daily returns (decimal, e.g. -0.02 = -2%).
        confidence: Confidence level (0.99 = 99% CVaR). Use 0.99 for NSE.
        lookback: Rolling window in trading days (252 = 1 year).
        
    Returns:
        CVaR as a negative number (e.g. -0.035 means 3.5% expected tail loss).
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    # Use last `lookback` returns
    recent = returns.iloc[-lookback:] if len(returns) > lookback else returns
    
    if len(recent) < 20:
        logger.warning(f"CVaR: Only {len(recent)} observations — result may be unreliable.")
    
    # VaR threshold
    alpha = 1 - confidence  # e.g. 0.01 for 99%
    var_threshold = np.percentile(recent, alpha * 100)
    
    # CVaR = mean of returns worse than VaR
    tail_returns = recent[recent <= var_threshold]
    
    if len(tail_returns) == 0:
        logger.warning("CVaR: No returns below VaR threshold — returning VaR as CVaR.")
        return float(var_threshold)
    
    cvar = float(tail_returns.mean())
    
    logger.info(
        f"CVaR({confidence*100:.0f}%, {lookback}d): {cvar:.4f} "
        f"(VaR={var_threshold:.4f}, tail_obs={len(tail_returns)})"
    )
    return cvar


def historical_var(
    returns: pd.Series,
    confidence: float = 0.99,
    lookback: int = 252,
) -> float:
    """
    Compute Historical VaR.
    
    Returns:
        VaR as a negative number (e.g. -0.025 means 2.5% worst-case loss).
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return 0.0

    recent = returns.iloc[-lookback:] if len(returns) > lookback else returns
    alpha = 1 - confidence
    return float(np.percentile(recent, alpha * 100))


def parametric_var(
    conditional_vol: float,
    confidence: float = 0.99,
    horizon: int = 1,
    mean_return: float = 0.0,
) -> float:
    """
    Compute Parametric VaR using EGARCH conditional volatility.
    
    Uses EGARCH sigma as the volatility input (forward-looking).
    Assumes Student-t distribution (consistent with EGARCH fit).
    
    Args:
        conditional_vol: EGARCH conditional volatility (sigma, not sigma^2).
        confidence: Confidence level (0.99 for 99%).
        horizon: Forecast horizon in days.
        mean_return: Expected return (usually ~0 for daily).
        
    Returns:
        VaR as a negative number.
    """
    from scipy.stats import norm
    
    # z-score for the given confidence level
    z = norm.ppf(1 - confidence)  # e.g. -2.326 for 99%
    
    # Scale volatility by sqrt(horizon) for multi-day
    vol_scaled = conditional_vol * np.sqrt(horizon)
    
    var = mean_return + z * vol_scaled
    
    logger.info(f"Parametric VaR({confidence*100:.0f}%, {horizon}d): {var:.4f} (sigma={conditional_vol:.4f})")
    return float(var)


def cvar_parametric(
    conditional_vol: float,
    confidence: float = 0.99,
    horizon: int = 1,
    mean_return: float = 0.0,
) -> float:
    """
    Parametric CVaR (Expected Shortfall) using Gaussian approximation.
    
    ES = mu - sigma * phi(z) / (1 - confidence)
    where phi is the standard normal PDF and z is the VaR quantile.
    """
    from scipy.stats import norm
    
    z = norm.ppf(1 - confidence)
    vol_scaled = conditional_vol * np.sqrt(horizon)
    
    es = mean_return - vol_scaled * norm.pdf(z) / (1 - confidence)
    return float(es)
