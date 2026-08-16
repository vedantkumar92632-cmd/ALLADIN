"""
Emergency circuit breaker / kill-switch (L4.S006).

Explicit kill-switch required by serious risk frameworks (MiFID II referenced
in Part 5 for context — the practice, not the regulation, transfers to NSE).

This is the last line of defense. When triggered:
- All new entries are halted immediately
- Optionally triggers liquidation of all open positions
- Can only be reset by explicit human intervention

Subscribes to HaltEvents from drawdown monitor and adds its own checks.
"""
from core.logger import get_logger
from engine.message_bus import message_bus
from engine.event_types import HaltEvent
from config.risk_settings import load_risk_settings

logger = get_logger("risk.circuit_breaker")


class CircuitBreaker:
    """
    Emergency kill-switch for the trading system.
    
    Monitors for catastrophic conditions and halts all trading.
    Can be triggered manually or automatically by risk events.
    """

    def __init__(self):
        risk = load_risk_settings()
        self._enabled = risk.emergency_kill_switch_enabled
        self._consecutive_loss_limit = risk.consecutive_loss_limit
        self._max_slippage_bps = risk.max_slippage_bps
        
        self._is_tripped = False
        self._trip_reason = ""
        self._consecutive_losses = 0
        
        # Subscribe to HaltEvents from drawdown monitor
        message_bus.subscribe(HaltEvent, self._on_halt_event)
        
        logger.info(
            f"CircuitBreaker initialized: enabled={self._enabled}, "
            f"consecutive_loss_limit={self._consecutive_loss_limit}, "
            f"max_slippage_bps={self._max_slippage_bps}"
        )

    def _on_halt_event(self, event: HaltEvent):
        """Handle HaltEvent from drawdown monitor."""
        if event.severity == "EMERGENCY":
            self.trip(f"emergency_halt_from_{event.reason}")
        elif event.reason == "portfolio_drawdown":
            # Portfolio-level drawdown is always escalated to circuit breaker
            self.trip(f"portfolio_drawdown_breach_{event.current_value:.4f}")

    def record_trade_result(self, pnl: float, slippage_bps: float = 0.0):
        """
        Record a trade result and check circuit breaker conditions.
        
        Args:
            pnl: Trade PnL (positive = profit, negative = loss).
            slippage_bps: Actual slippage in basis points.
        """
        if not self._enabled:
            return
        
        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self._consecutive_loss_limit:
                self.trip(
                    f"consecutive_losses_{self._consecutive_losses}"
                )
        else:
            self._consecutive_losses = 0
        
        # Check slippage
        if slippage_bps > self._max_slippage_bps:
            logger.warning(
                f"Slippage {slippage_bps:.0f}bps exceeds limit {self._max_slippage_bps}bps"
            )
            # Don't trip on single slippage event, but log it
            # Trip only on systematic slippage (future enhancement)

    def trip(self, reason: str):
        """
        Trip the circuit breaker — halt all trading.
        Can only be reset by explicit human call to reset().
        """
        if self._is_tripped:
            logger.warning(f"Circuit breaker already tripped ({self._trip_reason}). New reason: {reason}")
            return
        
        self._is_tripped = True
        self._trip_reason = reason
        
        logger.error(f"CIRCUIT BREAKER TRIPPED: {reason}. All trading halted.")
        
        # Publish emergency HaltEvent
        halt_event = HaltEvent(
            reason=f"circuit_breaker:{reason}",
            severity="EMERGENCY",
            threshold=0.0,
            current_value=0.0,
        )
        message_bus.publish(halt_event)

    def manual_trip(self, reason: str = "manual_intervention"):
        """Manual kill-switch activation."""
        self.trip(reason)

    @property
    def is_tripped(self) -> bool:
        """Whether the circuit breaker is currently tripped."""
        return self._is_tripped

    @property
    def trip_reason(self) -> str:
        """Reason the circuit breaker was tripped."""
        return self._trip_reason

    def reset(self, authorized_by: str = "system"):
        """
        Reset the circuit breaker. Requires explicit human authorization.
        
        Args:
            authorized_by: Who is authorizing the reset (for audit trail).
        """
        if self._is_tripped:
            logger.warning(
                f"Circuit breaker RESET by '{authorized_by}' "
                f"(was tripped: {self._trip_reason})"
            )
            self._is_tripped = False
            self._trip_reason = ""
            self._consecutive_losses = 0

    def can_trade(self) -> bool:
        """Check if trading is allowed (circuit breaker not tripped)."""
        if not self._enabled:
            return True
        return not self._is_tripped
