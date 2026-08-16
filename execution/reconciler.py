"""
Reconciler (L8.S004).

Per Part 5: constantly checks whether the system's internal belief about
orders/positions matches what the broker actually reports. Build this BEFORE
going live with any real capital.

Position mismatches are one of the most dangerous failure modes in
algorithmic trading — they can cause doubled positions, phantom fills,
or undetected losses.
"""
from typing import Dict, List, Tuple
from core.logger import get_logger
from execution.oms import OrderManager

logger = get_logger("execution.reconciler")


class PositionReconciler:
    """
    Reconciles internal OMS state against broker-reported positions.
    
    Any mismatch triggers an alert and optionally halts trading.
    """

    def __init__(self, oms: OrderManager, halt_on_mismatch: bool = True):
        self.oms = oms
        self.halt_on_mismatch = halt_on_mismatch
        self._mismatches: List[dict] = []

    def reconcile(
        self,
        broker_positions: Dict[str, int],
    ) -> Tuple[bool, List[dict]]:
        """
        Compare internal positions against broker positions.
        
        Args:
            broker_positions: {ticker: quantity} from broker API.
            
        Returns:
            (is_reconciled, list_of_mismatches).
        """
        internal_positions = self.oms.get_all_positions()
        mismatches = []
        
        # All tickers from both sources
        all_tickers = set(list(internal_positions.keys()) + list(broker_positions.keys()))
        
        for ticker in all_tickers:
            internal_qty = internal_positions.get(ticker, 0)
            broker_qty = broker_positions.get(ticker, 0)
            
            if internal_qty != broker_qty:
                mismatch = {
                    'ticker': ticker,
                    'internal_qty': internal_qty,
                    'broker_qty': broker_qty,
                    'difference': broker_qty - internal_qty,
                }
                mismatches.append(mismatch)
                logger.error(
                    f"POSITION MISMATCH: {ticker} internal={internal_qty}, "
                    f"broker={broker_qty}, diff={broker_qty - internal_qty}"
                )
        
        self._mismatches = mismatches
        is_reconciled = len(mismatches) == 0
        
        if is_reconciled:
            logger.info("Reconciliation: All positions match.")
        else:
            logger.error(f"Reconciliation FAILED: {len(mismatches)} mismatches found!")
            if self.halt_on_mismatch:
                self.oms.halt_trading()
                logger.error("Trading HALTED due to position mismatch.")
        
        return is_reconciled, mismatches

    def get_last_mismatches(self) -> List[dict]:
        return self._mismatches
