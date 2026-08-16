"""
OpenAlgo adapter (L8.S001) — Paper-trading connection.

Per Part 3B Domain 15: OpenAlgo is tier 1 for NSE (30+ Indian brokers,
unified API, built by an NSE practitioner).

IMPORTANT: This adapter connects ONLY to paper trading mode. Do NOT
connect to a live funded account at this step. That decision requires
explicit human sign-off per Part 1.

The adapter wraps OpenAlgo's REST API with methods that match the
internal order management interface.

Status: REFERENCE_PATTERN — OpenAlgo is used as a unified interface
concept. The actual HTTP calls would need OpenAlgo running locally.
This module provides the abstraction layer.
"""
from typing import Optional, Dict
from enum import Enum
from core.logger import get_logger

logger = get_logger("adapters.brokers.openalgo")


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"         # Stop-loss
    SL_M = "SL-M"     # Stop-loss market


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OpenAlgoAdapter:
    """
    Adapter for OpenAlgo unified broker API.
    
    Paper-trading mode only. Real capital requires human sign-off.
    """

    def __init__(
        self,
        api_host: str = "http://127.0.0.1:5000",
        api_key: str = "",
        paper_mode: bool = True,
    ):
        """
        Args:
            api_host: OpenAlgo API host URL.
            api_key: API key for authentication.
            paper_mode: MUST be True. Real trading requires human override.
        """
        if not paper_mode:
            raise ValueError(
                "OpenAlgoAdapter: paper_mode=False is BLOCKED. "
                "Real capital trading requires human authorization. "
                "See MASTER_REFERENCE.md Layer 8."
            )
        
        self.api_host = api_host
        self.api_key = api_key
        self.paper_mode = paper_mode
        self._connected = False
        self._orders: Dict[str, dict] = {}
        self._order_counter = 0
        
        logger.info(f"OpenAlgoAdapter initialized (paper_mode={paper_mode})")

    def connect(self) -> bool:
        """Establish connection to OpenAlgo API."""
        # In paper mode, we simulate connection
        self._connected = True
        logger.info("OpenAlgoAdapter: Connected (paper mode)")
        return True

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float = 0.0,
        trigger_price: float = 0.0,
        exchange: str = "NSE",
        product: str = "CNC",
    ) -> str:
        """
        Place an order through OpenAlgo.
        
        Args:
            symbol: NSE symbol (e.g. 'RELIANCE').
            side: BUY or SELL.
            quantity: Number of shares.
            order_type: MARKET, LIMIT, SL, SL-M.
            price: Limit price (for LIMIT orders).
            trigger_price: Trigger price (for SL orders).
            exchange: Exchange code (default NSE).
            product: Product type (CNC=delivery, MIS=intraday).
            
        Returns:
            Order ID string.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"
        
        side_val = side.value if hasattr(side, 'value') else str(side)
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side_val,
            'quantity': quantity,
            'order_type': order_type.value,
            'price': price,
            'trigger_price': trigger_price,
            'exchange': exchange,
            'product': product,
            'status': OrderStatus.FILLED.value,  # Paper mode: instant fill
            'filled_qty': quantity,
            'avg_price': price if price > 0 else 0.0,
        }
        
        self._orders[order_id] = order
        
        logger.info(
            f"Order placed: {order_id} {side.value} {quantity} {symbol} "
            f"@ {order_type.value} (paper mode)"
        )
        return order_id

    def get_order_status(self, order_id: str) -> dict:
        """Get status of an order."""
        return self._orders.get(order_id, {'status': 'UNKNOWN'})

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id in self._orders:
            self._orders[order_id]['status'] = OrderStatus.CANCELLED.value
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False

    def get_positions(self) -> list:
        """Get current positions."""
        positions = {}
        for order in self._orders.values():
            if order['status'] == OrderStatus.FILLED.value:
                sym = order['symbol']
                qty = order['filled_qty']
                if order['side'] == 'BUY':
                    positions[sym] = positions.get(sym, 0) + qty
                else:
                    positions[sym] = positions.get(sym, 0) - qty
        
        return [{'symbol': k, 'quantity': v} for k, v in positions.items() if v != 0]

    def disconnect(self):
        """Disconnect from OpenAlgo API."""
        self._connected = False
        logger.info("OpenAlgoAdapter: Disconnected")
