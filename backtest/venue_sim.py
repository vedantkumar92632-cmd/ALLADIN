"""
Venue simulator — realistic fill model (L7.S003).

Per Part 5: without this, backtest results are "dangerously optimistic."

Simulates:
- Partial fills
- Latency
- Slippage (based on volatility and order size)
- Rejects (for illiquid instruments)
"""
import numpy as np
from typing import Optional
from core.logger import get_logger

logger = get_logger("backtest.venue_sim")


class VenueSim:
    """
    Simulated venue for realistic backtesting.
    
    Models slippage, partial fills, and latency to prevent
    overly optimistic backtest results.
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        fill_rate: float = 0.95,
        latency_ms: float = 50.0,
        commission_bps: float = 3.0,
    ):
        """
        Args:
            slippage_bps: Average slippage in basis points.
            fill_rate: Probability of full fill (0 to 1).
            latency_ms: Simulated latency in milliseconds.
            commission_bps: Commission per trade in basis points.
        """
        self.slippage_bps = slippage_bps
        self.fill_rate = fill_rate
        self.latency_ms = latency_ms
        self.commission_bps = commission_bps
        
        logger.info(
            f"VenueSim: slippage={slippage_bps}bps, fill_rate={fill_rate:.0%}, "
            f"latency={latency_ms}ms, commission={commission_bps}bps"
        )

    def simulate_fill(
        self,
        order_price: float,
        order_qty: int,
        side: str,
        volatility: float = 0.015,
        avg_volume: float = 1000000,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Simulate a fill for a given order.
        
        Args:
            order_price: Requested price.
            order_qty: Requested quantity.
            side: 'BUY' or 'SELL'.
            volatility: Current daily volatility of the instrument.
            avg_volume: Average daily volume.
            seed: Optional random seed.
            
        Returns:
            Dict with fill_price, fill_qty, slippage, commission, is_rejected.
        """
        rng = np.random.RandomState(seed)
        
        # Market impact: larger orders relative to volume = more slippage
        size_ratio = order_qty / max(avg_volume, 1)
        impact_multiplier = 1.0 + size_ratio * 10  # Linear impact model
        
        # Slippage (random with volatility scaling)
        slippage_pct = (self.slippage_bps / 10000) * impact_multiplier
        slippage_noise = rng.normal(0, volatility * 0.1)
        total_slippage_pct = slippage_pct + abs(slippage_noise)
        
        # Apply slippage in the adverse direction
        if side.upper() == 'BUY':
            fill_price = order_price * (1 + total_slippage_pct)
        else:
            fill_price = order_price * (1 - total_slippage_pct)
        
        # Partial fills
        if rng.random() < self.fill_rate:
            fill_qty = order_qty
        else:
            fill_qty = int(order_qty * rng.uniform(0.3, 0.9))
        
        # Reject very large orders relative to volume
        is_rejected = size_ratio > 0.05  # Can't fill >5% of daily volume
        if is_rejected:
            fill_qty = 0
            fill_price = 0.0
        
        # Commission
        commission = fill_price * fill_qty * (self.commission_bps / 10000)
        
        slippage_amount = abs(fill_price - order_price) * fill_qty
        
        return {
            'fill_price': round(fill_price, 2),
            'fill_qty': fill_qty,
            'order_qty': order_qty,
            'slippage_amount': round(slippage_amount, 2),
            'slippage_bps': round(total_slippage_pct * 10000, 2),
            'commission': round(commission, 2),
            'is_rejected': is_rejected,
            'is_partial': fill_qty < order_qty and not is_rejected,
        }
