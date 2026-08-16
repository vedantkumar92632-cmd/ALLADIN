"""
Signal score aggregation (L6.S001).

Aggregates Layer 3 alpha outputs into one composite score per ticker.
This is glue logic specific to this architecture — combines CatBoost,
LightGBM, regime, volatility, and NLP signals into a single score.
"""
import numpy as np
from typing import Dict, List, Optional
from core.logger import get_logger

logger = get_logger("decision.signal_score")

# Default weights for each signal domain
DEFAULT_DOMAIN_WEIGHTS = {
    'ml_alpha': 0.30,        # CatBoost + LightGBM ensemble
    'regime': 0.15,          # SJM + BOCPD regime state
    'volatility': 0.15,      # EGARCH/GJR-GARCH vol regime
    'mean_reversion': 0.10,  # OU + Kalman z-scores
    'sentiment': 0.10,       # FinBERT + VADER
    'momentum': 0.20,        # Price momentum (from features)
}


def compute_signal_score(
    ticker: str,
    signals: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute composite signal score for a single ticker.
    
    Args:
        ticker: NSE ticker symbol.
        signals: Dict mapping domain name to signal value (each in [-1, +1]).
        weights: Optional custom domain weights. Defaults to DEFAULT_DOMAIN_WEIGHTS.
        
    Returns:
        Composite score in [-1, +1].
    """
    w = weights or DEFAULT_DOMAIN_WEIGHTS
    
    score = 0.0
    total_weight = 0.0
    
    for domain, signal_value in signals.items():
        if signal_value is None or (isinstance(signal_value, float) and np.isnan(signal_value)):
            continue  # Skip NaN/None signals
        domain_weight = w.get(domain, 0.0)
        score += signal_value * domain_weight
        total_weight += domain_weight
    
    # Normalize by actual weight used (handles missing domains gracefully)
    if total_weight > 0:
        score = score / total_weight
    
    # Clamp to [-1, +1]
    score = max(-1.0, min(1.0, score))
    
    return score


def compute_signal_scores_batch(
    ticker_signals: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute composite scores for all tickers.
    
    Args:
        ticker_signals: {ticker: {domain: signal_value}}.
        
    Returns:
        {ticker: composite_score}.
    """
    scores = {}
    for ticker, signals in ticker_signals.items():
        scores[ticker] = compute_signal_score(ticker, signals, weights)
    
    logger.info(f"Computed signal scores for {len(scores)} tickers.")
    return scores
