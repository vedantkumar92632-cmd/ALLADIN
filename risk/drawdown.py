"""
Continuous drawdown monitor (L4.S005).

Per Part 1: a drawdown breach must HALT NEW ENTRIES AUTOMATICALLY IN CODE,
not be a dashboard number a human has to notice. Wire this to the message bus
so a breach publishes a HaltEvent that other components subscribe to.

Monitors:
- Daily drawdown (default: 2% threshold)
- Weekly drawdown (default: 5% threshold)
- Portfolio peak-to-trough drawdown (default: 10% threshold)
"""
import numpy as np
import pandas as pd
from typing import Optional
from core.logger import get_logger
import engine.message_bus as _mb_module
from engine.event_types import HaltEvent
from config.risk_settings import load_risk_settings

logger = get_logger("risk.drawdown")


class DrawdownMonitor:
    """
    Continuous drawdown monitor that publishes HaltEvent on breach.
    
    Tracks daily, weekly, and absolute portfolio drawdown.
    On breach, publishes HaltEvent to message bus — all subscribers
    (order manager, strategy layer) must stop new entries immediately.
    """

    def __init__(
        self,
        max_daily_dd: Optional[float] = None,
        max_weekly_dd: Optional[float] = None,
        max_portfolio_dd: Optional[float] = None,
        bus=None,
    ):
        """
        Args:
            max_daily_dd: Max daily drawdown (positive float, e.g. 0.02 = 2%).
            max_weekly_dd: Max weekly drawdown.
            max_portfolio_dd: Max peak-to-trough portfolio drawdown.
        """
        risk = load_risk_settings()
        self.max_daily_dd = max_daily_dd or risk.max_daily_drawdown_pct
        self.max_weekly_dd = max_weekly_dd or risk.max_weekly_drawdown_pct
        self.max_portfolio_dd = max_portfolio_dd or risk.max_portfolio_drawdown_pct
        
        self._bus = bus  # If None, uses the global message_bus at publish time
        self._portfolio_peak = 0.0
        self._daily_returns = []
        self._weekly_returns = []
        self._is_halted = False
        self._halt_reason = ""
        
        logger.info(
            f"DrawdownMonitor initialized: daily={self.max_daily_dd:.1%}, "
            f"weekly={self.max_weekly_dd:.1%}, portfolio={self.max_portfolio_dd:.1%}"
        )

    def update(self, portfolio_value: float, daily_return: float) -> bool:
        """
        Update the monitor with today's data. Check all drawdown thresholds.
        
        Args:
            portfolio_value: Current portfolio NAV.
            daily_return: Today's portfolio return (decimal, e.g. -0.02 = -2%).
            
        Returns:
            True if a halt was triggered, False otherwise.
        """
        self._daily_returns.append(daily_return)
        self._daily_returns = self._daily_returns[-504:]
        self._weekly_returns.append(daily_return)
        
        # Keep only last 5 days for weekly
        if len(self._weekly_returns) > 5:
            self._weekly_returns = self._weekly_returns[-5:]
        
        # Track portfolio peak
        if portfolio_value > self._portfolio_peak:
            self._portfolio_peak = portfolio_value
        
        # Check daily drawdown
        if abs(daily_return) > 0 and daily_return < 0:
            if abs(daily_return) >= self.max_daily_dd:
                self._trigger_halt(
                    reason="daily_drawdown",
                    threshold=self.max_daily_dd,
                    current_value=abs(daily_return),
                )
                return True
        
        # Check weekly drawdown (cumulative over last 5 days)
        if len(self._weekly_returns) >= 2:
            weekly_cum = np.prod([1 + r for r in self._weekly_returns]) - 1
            if weekly_cum < 0 and abs(weekly_cum) >= self.max_weekly_dd:
                self._trigger_halt(
                    reason="weekly_drawdown",
                    threshold=self.max_weekly_dd,
                    current_value=abs(weekly_cum),
                )
                return True
        
        # Check portfolio peak-to-trough drawdown
        if self._portfolio_peak > 0:
            portfolio_dd = (self._portfolio_peak - portfolio_value) / self._portfolio_peak
            if portfolio_dd >= self.max_portfolio_dd:
                self._trigger_halt(
                    reason="portfolio_drawdown",
                    threshold=self.max_portfolio_dd,
                    current_value=portfolio_dd,
                )
                return True
        
        return False

    def _trigger_halt(self, reason: str, threshold: float, current_value: float):
        """Publish HaltEvent on the message bus."""
        self._is_halted = True
        self._halt_reason = reason
        
        halt_event = HaltEvent(
            reason=reason,
            severity="HALT",
            threshold=threshold,
            current_value=current_value,
        )
        
        logger.warning(
            f"DRAWDOWN BREACH: {reason} = {current_value:.4f} "
            f"(threshold: {threshold:.4f}). Publishing HaltEvent."
        )
        
        bus = self._bus if self._bus is not None else _mb_module.message_bus
        bus.publish(halt_event)

    @property
    def is_halted(self) -> bool:
        """Whether trading is currently halted due to drawdown breach."""
        return self._is_halted

    def reset_halt(self):
        """
        Manually reset the halt state.
        Per Part 1: this should require human intervention, not auto-reset.
        """
        if self._is_halted:
            logger.warning(f"Halt manually reset (was: {self._halt_reason}).")
            self._is_halted = False
            self._halt_reason = ""

    @staticmethod
    def compute_max_drawdown(returns: pd.Series) -> float:
        """
        Compute maximum drawdown from a return series.
        
        Returns:
            Max drawdown as a positive float (e.g. 0.15 = 15% drawdown).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (running_max - cumulative) / running_max
        return float(drawdown.max())
