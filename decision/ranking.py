"""
Ranking (L6.S003) — ranks by risk-adjusted expected value.

Uses Layer 4 risk outputs, not raw score, for ranking.
"""
from typing import Dict, Optional
from core.logger import get_logger

logger = get_logger("decision.ranking")


def rank_by_risk_adjusted(
    scores: Dict[str, float],
    convictions: Dict[str, int],
    risk_scores: Optional[Dict[str, float]] = None,
) -> list:
    """
    Rank tickers by risk-adjusted expected value.
    
    Risk-adjusted score = composite_score * conviction * (1 - risk_penalty).
    
    Args:
        scores: {ticker: composite_score}.
        convictions: {ticker: conviction_0_to_100}.
        risk_scores: Optional {ticker: risk_penalty_0_to_1}. Higher = riskier.
        
    Returns:
        List of (ticker, adjusted_score, conviction) sorted descending.
    """
    ranked = []
    
    for ticker in scores:
        score = scores[ticker]
        conviction = convictions.get(ticker, 50)
        risk_penalty = risk_scores.get(ticker, 0.0) if risk_scores else 0.0
        
        import math
        if math.isnan(score) or math.isnan(conviction) or math.isnan(risk_penalty):
            continue
            
        # Risk-adjusted expected value
        adjusted = score * (conviction / 100.0) * (1.0 - risk_penalty)
        
        ranked.append({
            'ticker': ticker,
            'raw_score': score,
            'conviction': conviction,
            'risk_penalty': risk_penalty,
            'adjusted_score': adjusted,
        })
    
    # Sort by absolute adjusted score descending (strongest signal first)
    ranked.sort(key=lambda x: abs(x['adjusted_score']), reverse=True)
    
    logger.info(f"Ranked {len(ranked)} tickers by risk-adjusted expected value.")
    return ranked
