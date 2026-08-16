"""
Prioritization (L6.S005) — final ordered trade list with urgency tagging.

This is the last step in the decision pipeline. Takes the filtered,
ranked tickers and assigns urgency levels for execution.
"""
from typing import List, Dict
from core.logger import get_logger

logger = get_logger("decision.prioritization")

# Urgency thresholds
URGENCY_HIGH = 70      # Conviction >= 70 = urgent
URGENCY_MEDIUM = 40    # Conviction 40-69 = standard
# Below 40 = low urgency


def prioritize(
    filtered_ranked: list,
    max_positions: int = 20,
) -> List[Dict]:
    """
    Final ordered trade list with urgency tagging.
    
    Args:
        filtered_ranked: Output of filtering.filter_universe().
        max_positions: Maximum number of positions to recommend.
        
    Returns:
        List of trade recommendations with urgency.
    """
    trade_list = []
    
    for i, entry in enumerate(filtered_ranked[:max_positions]):
        conviction = entry.get('conviction', 50)
        adj_score = entry.get('adjusted_score', 0)
        
        # Assign urgency
        if conviction >= URGENCY_HIGH:
            urgency = "HIGH"
        elif conviction >= URGENCY_MEDIUM:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Direction from score
        if adj_score > 0:
            direction = "BUY"
        elif adj_score < 0:
            direction = "SELL"
        else:
            direction = "HOLD"
        
        trade_list.append({
            'rank': i + 1,
            'ticker': entry['ticker'],
            'direction': direction,
            'adjusted_score': adj_score,
            'conviction': conviction,
            'urgency': urgency,
            'raw_score': entry.get('raw_score', 0),
            'risk_penalty': entry.get('risk_penalty', 0),
        })
    
    high = sum(1 for t in trade_list if t['urgency'] == 'HIGH')
    med = sum(1 for t in trade_list if t['urgency'] == 'MEDIUM')
    low = sum(1 for t in trade_list if t['urgency'] == 'LOW')
    
    logger.info(
        f"Prioritized {len(trade_list)} trades: "
        f"{high} HIGH, {med} MEDIUM, {low} LOW urgency"
    )
    return trade_list
