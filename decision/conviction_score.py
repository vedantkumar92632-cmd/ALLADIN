"""
Conviction score (L6.S002) — 0-100 scale, per Part 5.

Maps composite signal scores to a conviction level that accounts for
signal agreement, regime context, and model confidence.
"""
import numpy as np
from typing import Dict, Optional
from core.logger import get_logger

logger = get_logger("decision.conviction_score")


def compute_conviction(
    composite_score: float,
    n_agreeing_models: int = 1,
    total_models: int = 1,
    regime_multiplier: float = 1.0,
) -> int:
    """
    Convert composite signal score to conviction (0-100 scale).
    
    Factors:
    1. Base: |composite_score| * 100
    2. Agreement bonus: more models agreeing = higher conviction
    3. Regime adjustment: low-vol regimes boost conviction
    
    Args:
        composite_score: Signal score in [-1, +1].
        n_agreeing_models: Number of alpha models that agree on direction.
        total_models: Total number of alpha models.
        regime_multiplier: 1.0 = normal, <1.0 = bearish/high-vol penalty.
        
    Returns:
        Conviction score (0-100).
    """
    if composite_score is None or np.isnan(composite_score):
        return 0

    # Base conviction from signal strength
    base = abs(composite_score) * 70  # Max 70 from signal alone
    
    # Agreement bonus (max 30 additional points)
    if total_models > 0:
        agreement_ratio = n_agreeing_models / total_models
        agreement_bonus = agreement_ratio * 30
    else:
        agreement_bonus = 0
    
    raw_conviction = (base + agreement_bonus) * regime_multiplier
    
    # Clamp to 0-100
    conviction = max(0, min(100, int(round(raw_conviction))))
    
    return conviction


def compute_conviction_batch(
    scores: Dict[str, float],
    model_agreement: Optional[Dict[str, tuple]] = None,
    regime_multiplier: float = 1.0,
) -> Dict[str, int]:
    """
    Compute conviction for all tickers.
    
    Args:
        scores: {ticker: composite_score}.
        model_agreement: Optional {ticker: (n_agreeing, total_models)}.
        
    Returns:
        {ticker: conviction_0_to_100}.
    """
    convictions = {}
    for ticker, score in scores.items():
        if model_agreement and ticker in model_agreement:
            n_agree, total = model_agreement[ticker]
        else:
            n_agree, total = 1, 1
        
        convictions[ticker] = compute_conviction(
            score, n_agree, total, regime_multiplier
        )
    
    logger.info(f"Computed conviction scores for {len(convictions)} tickers.")
    return convictions
