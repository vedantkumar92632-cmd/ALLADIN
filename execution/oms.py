"""
Order Management System (L8.S003).

Tracks order state locally. This is the internal book of record — what
the system BELIEVES about its orders and positions.

Must be reconciled against broker state via the reconciler (L8.S004).
"""
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
from core.logger import get_logger

logger = get_logger("execution.oms")


class OrderState(Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class InternalOrder:
    """Internal representation of an order."""
    
    def __init__(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float = 0.0,
        stop_price: float = 0.0,
    ):
        self.order_id = ""
        self.broker_order_id = ""
        self.ticker = ticker
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.state = OrderState.CREATED
        self.filled_qty = 0
        self.avg_fill_price = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'broker_order_id': self.broker_order_id,
            'ticker': self.ticker,
            'side': self.side,
            'quantity': self.quantity,
            'order_type': self.order_type,
            'state': self.state.value,
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
        }


class OrderManager:
    """
    Internal order management system.
    
    Maintains the system's belief about orders and positions.
    This must be reconciled against broker state before any decisions.
    """

    def __init__(self):
        self._orders: Dict[str, InternalOrder] = {}
        self._positions: Dict[str, int] = {}
        self._position_avg_prices: Dict[str, float] = {}
        self._order_counter = 0
        self._is_halted = False

    def create_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: float = 0.0,
    ) -> Optional[InternalOrder]:
        """
        Create a new internal order.
        
        Returns None if trading is halted.
        """
        if self._is_halted:
            logger.warning(f"OMS: Order rejected — trading is halted. {side} {quantity} {ticker}")
            return None
        
        self._order_counter += 1
        order = InternalOrder(ticker, side, quantity, order_type, limit_price)
        order.order_id = f"INT-{self._order_counter:06d}"
        
        self._orders[order.order_id] = order
        
        logger.info(f"OMS: Created {order.order_id} {side} {quantity} {ticker} @ {order_type}")
        return order

    def mark_submitted(self, order_id: str, broker_order_id: str):
        """Mark order as submitted to broker."""
        if order_id in self._orders:
            self._orders[order_id].state = OrderState.SUBMITTED
            self._orders[order_id].broker_order_id = broker_order_id
            self._orders[order_id].updated_at = datetime.now()

    def mark_filled(self, order_id: str, filled_qty: int, avg_price: float):
        """Mark order as filled and update positions."""
        if order_id not in self._orders:
            return
        
        order = self._orders[order_id]
        # Compute delta from previous fill qty to handle cumulative updates
        prev_filled = order.filled_qty
        delta_qty = filled_qty - prev_filled
        
        order.filled_qty = filled_qty
        order.avg_fill_price = avg_price
        order.updated_at = datetime.now()
        
        if filled_qty >= order.quantity:
            order.state = OrderState.FILLED
        else:
            order.state = OrderState.PARTIAL
        
        # Update positions by delta only
        if delta_qty != 0:
            current = self._positions.get(order.ticker, 0)
            if order.side == "BUY":
                new_pos = current + delta_qty
            else:
                new_pos = current - delta_qty
            
            # Weighted average price
            if order.side == "BUY" and new_pos > 0:
                prev_value = self._position_avg_prices.get(order.ticker, avg_price) * current
                new_value = prev_value + delta_qty * avg_price
                self._position_avg_prices[order.ticker] = new_value / new_pos
            else:
                self._position_avg_prices[order.ticker] = avg_price
            
            self._positions[order.ticker] = new_pos
        
        logger.info(
            f"OMS: Filled {order_id} delta={delta_qty}@{avg_price:.2f}, "
            f"position {order.ticker}={self._positions.get(order.ticker, 0)}"
        )

    def halt_trading(self):
        """Halt all new order creation."""
        self._is_halted = True
        logger.warning("OMS: Trading HALTED")

    def resume_trading(self):
        """Resume order creation."""
        self._is_halted = False
        logger.info("OMS: Trading RESUMED")

    def get_position(self, ticker: str) -> int:
        """Get current position for a ticker."""
        return self._positions.get(ticker, 0)

    def get_all_positions(self) -> Dict[str, int]:
        """Get all non-zero positions."""
        return {k: v for k, v in self._positions.items() if v != 0}

    def get_order(self, order_id: str) -> Optional[InternalOrder]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[InternalOrder]:
        return [o for o in self._orders.values()
                if o.state in (OrderState.CREATED, OrderState.SUBMITTED, OrderState.PARTIAL)]
