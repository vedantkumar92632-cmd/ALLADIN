"""
Almgren-Chriss execution cost estimator (L8.S005).

Per Part 3B Domain 11: In this system, Almgren-Chriss is a COST ESTIMATOR,
not a live execution algorithm. The Indian broker API handles actual order
slicing. This module estimates market impact cost before order placement.
"""
import numpy as np
from core.logger import get_logger

logger = get_logger("execution.execution_algos")


def almgren_chriss_cost(
    order_size: int,
    avg_daily_volume: float,
    daily_volatility: float,
    urgency: float = 0.5,
    temporary_impact: float = 0.01,
    permanent_impact: float = 0.001,
) -> dict:
    """
    Estimate execution cost using Almgren-Chriss model.
    
    Args:
        order_size: Number of shares to trade.
        avg_daily_volume: Average daily volume of the instrument.
        daily_volatility: Daily return volatility (e.g. 0.02 = 2%).
        urgency: Trading urgency (0 = patient, 1 = immediate).
                Higher urgency = more market impact but less timing risk.
        temporary_impact: Temporary impact parameter.
        permanent_impact: Permanent impact parameter.
        
    Returns:
        Dict with estimated costs: temporary_cost, permanent_cost,
        timing_risk, total_cost_bps.
    """
    participation_rate = order_size / max(avg_daily_volume, 1)
    
    # Temporary impact: proportional to participation rate and urgency
    temp_cost = temporary_impact * daily_volatility * (participation_rate ** 0.5) * (1 + urgency)
    
    # Permanent impact: proportional to total order size
    perm_cost = permanent_impact * daily_volatility * participation_rate
    
    # Timing risk: variance of cost if we spread the order over time
    n_slices = max(1, int(1.0 / max(urgency, 0.01)))  # More slices = less urgent
    timing_risk = daily_volatility * np.sqrt(participation_rate / n_slices)
    
    total_cost = temp_cost + perm_cost
    total_cost_bps = total_cost * 10000
    
    result = {
        'temporary_cost': round(temp_cost, 6),
        'permanent_cost': round(perm_cost, 6),
        'timing_risk': round(timing_risk, 6),
        'total_cost': round(total_cost, 6),
        'total_cost_bps': round(total_cost_bps, 2),
        'participation_rate': round(participation_rate, 4),
        'recommended_slices': n_slices,
    }
    
    logger.info(
        f"A-C cost estimate: {order_size} shares, "
        f"participation={participation_rate:.2%}, "
        f"total_cost={total_cost_bps:.1f}bps"
    )
    return result
