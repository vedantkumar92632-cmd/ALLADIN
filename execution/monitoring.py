"""
Slippage and Fill Quality Monitor (L8.S006).

Per Part 2: without this you can't detect broker routing degrading
your live returns vs. backtest assumptions.

Tracks:
- Actual vs expected fill prices
- Slippage trends over time
- Fill rate degradation
"""
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from core.logger import get_logger

logger = get_logger("execution.monitoring")


class FillQualityMonitor:
    """
    Monitors fill quality to detect broker routing degradation.
    
    If live slippage consistently exceeds backtest assumptions,
    the system's edge is being eroded.
    """

    def __init__(
        self,
        expected_slippage_bps: float = 5.0,
        alert_threshold_bps: float = 15.0,
        window_size: int = 50,
    ):
        """
        Args:
            expected_slippage_bps: Backtest slippage assumption.
            alert_threshold_bps: Alert if rolling avg exceeds this.
            window_size: Rolling window for average slippage.
        """
        self.expected_slippage_bps = expected_slippage_bps
        self.alert_threshold_bps = alert_threshold_bps
        self.window_size = window_size
        
        self._fill_records: List[dict] = []
        self._alerts: List[dict] = []
        
        logger.info(
            f"FillQualityMonitor: expected={expected_slippage_bps}bps, "
            f"alert_threshold={alert_threshold_bps}bps"
        )

    def record_fill(
        self,
        ticker: str,
        side: str,
        expected_price: float,
        actual_price: float,
        quantity: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a fill and check for quality degradation.
        
        Args:
            ticker: Instrument ticker.
            side: BUY or SELL.
            expected_price: Price at signal generation time.
            actual_price: Actual fill price from broker.
            quantity: Filled quantity.
            timestamp: Fill timestamp.
        """
        ts = timestamp or datetime.now()
        
        # Slippage: positive = adverse (paid more to buy / received less to sell)
        if side.upper() == "BUY":
            slippage_pct = (actual_price - expected_price) / expected_price
        else:
            slippage_pct = (expected_price - actual_price) / expected_price
        
        slippage_bps = slippage_pct * 10000
        
        record = {
            'ticker': ticker,
            'side': side,
            'expected_price': expected_price,
            'actual_price': actual_price,
            'quantity': quantity,
            'slippage_bps': round(slippage_bps, 2),
            'slippage_amount': round(abs(actual_price - expected_price) * quantity, 2),
            'timestamp': ts.isoformat(),
        }
        self._fill_records.append(record)
        
        # Check rolling average
        recent = self._fill_records[-self.window_size:]
        avg_slippage = np.mean([r['slippage_bps'] for r in recent])
        
        if avg_slippage > self.alert_threshold_bps:
            alert = {
                'type': 'slippage_degradation',
                'avg_slippage_bps': round(avg_slippage, 2),
                'threshold_bps': self.alert_threshold_bps,
                'n_fills': len(recent),
                'timestamp': ts.isoformat(),
            }
            self._alerts.append(alert)
            logger.warning(
                f"FILL QUALITY ALERT: avg slippage {avg_slippage:.1f}bps "
                f"exceeds threshold {self.alert_threshold_bps}bps "
                f"(expected {self.expected_slippage_bps}bps)"
            )

    def get_stats(self) -> dict:
        """Get fill quality statistics."""
        if not self._fill_records:
            return {'n_fills': 0}
        
        slippages = [r['slippage_bps'] for r in self._fill_records]
        total_cost = sum(r['slippage_amount'] for r in self._fill_records)
        
        return {
            'n_fills': len(self._fill_records),
            'avg_slippage_bps': round(np.mean(slippages), 2),
            'median_slippage_bps': round(np.median(slippages), 2),
            'max_slippage_bps': round(max(slippages), 2),
            'min_slippage_bps': round(min(slippages), 2),
            'total_slippage_cost': round(total_cost, 2),
            'n_alerts': len(self._alerts),
            'exceeds_backtest_assumption': np.mean(slippages) > self.expected_slippage_bps,
        }

    def get_alerts(self) -> List[dict]:
        return self._alerts
