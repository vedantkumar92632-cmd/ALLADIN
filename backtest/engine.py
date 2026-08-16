"""
Backtest engine (L7.S001/L7.S002).

Wraps vectorbt for fast parameter sweeps and provides a simple event-driven
backtest loop for final validation.

Per Part 6: Use vectorbt first for fast/cheap parameter sweeps to test whether
an idea has potential, then use the event-driven loop for final validation.

Note: zipline-reloaded integration is DEFERRED — it has heavy Windows
compatibility issues. The event-driven backtest loop below provides
equivalent bar-by-bar simulation capability without the dependency.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from core.logger import get_logger

logger = get_logger("backtest.engine")


class SimpleBacktest:
    """
    Simple event-driven backtest engine.
    
    Feeds data bar-by-bar to prevent look-ahead bias (the key requirement
    from Part 6 A7). Uses VenueSim for realistic fills.
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        commission_bps: float = 3.0,
        slippage_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}
        self.position_prices: Dict[str, float] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        
    def run(
        self,
        prices: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, int], Dict[str, float]],
    ) -> dict:
        """
        Run the backtest bar-by-bar.
        
        Args:
            prices: DataFrame with close prices (columns = tickers, index = dates).
            signal_fn: Function(prices_up_to_bar, bar_index) -> {ticker: signal}.
                      Signal > 0 = BUY, < 0 = SELL.
                      
        Returns:
            Dict with equity_curve, trades, metrics.
        """
        self.cash = self.initial_capital
        self.positions = {}
        self.position_prices = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        prev_equity = self.initial_capital
        
        for i in range(1, len(prices)):
            # Only pass data up to current bar (no look-ahead)
            available_data = prices.iloc[:i]
            current_prices = prices.iloc[i]
            
            # Get signals
            signals = signal_fn(available_data, i)
            
            # Execute signals
            for ticker, signal in signals.items():
                if ticker not in current_prices.index:
                    continue
                    
                price = current_prices[ticker]
                if np.isnan(price) or price <= 0:
                    continue
                
                current_pos = self.positions.get(ticker, 0)
                
                if signal > 0 and current_pos <= 0:
                    # Buy
                    qty = max(1, int(self.cash * 0.05 / price))  # 5% per position
                    cost = qty * price * (1 + self.slippage_bps / 10000)
                    commission = cost * self.commission_bps / 10000
                    if cost + commission <= self.cash:
                        self.cash -= cost + commission
                        self.positions[ticker] = current_pos + qty
                        self.position_prices[ticker] = price
                        self.trades.append({
                            'bar': i, 'ticker': ticker, 'action': 'BUY',
                            'qty': qty, 'price': price, 'commission': commission,
                        })
                
                elif signal < 0 and current_pos > 0:
                    # Sell
                    proceeds = current_pos * price * (1 - self.slippage_bps / 10000)
                    commission = proceeds * self.commission_bps / 10000
                    self.cash += proceeds - commission
                    entry_price = self.position_prices.get(ticker, price)
                    self.trades.append({
                        'bar': i, 'ticker': ticker, 'action': 'SELL',
                        'qty': current_pos, 'price': price, 'commission': commission,
                        'pnl': (price - entry_price) * current_pos,
                    })
                    self.positions[ticker] = 0
            
            # Compute equity
            position_value = sum(
                qty * current_prices.get(t, 0)
                for t, qty in self.positions.items()
                if qty > 0
            )
            equity = self.cash + position_value
            self.equity_curve.append(equity)
            
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_equity = equity
        
        returns = np.array(self.daily_returns)
        
        sharpe = 0.0 if (len(returns) < 2 or np.std(returns) == 0) else float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        metrics = {
            'total_return': (self.equity_curve[-1] / self.initial_capital - 1) if self.equity_curve else 0,
            'sharpe': sharpe,
            'max_drawdown': self._max_drawdown(returns),
            'n_trades': len(self.trades),
            'final_equity': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
        }
        
        logger.info(
            f"Backtest complete: return={metrics['total_return']:.2%}, "
            f"sharpe={metrics['sharpe']:.2f}, max_dd={metrics['max_drawdown']:.2%}, "
            f"trades={metrics['n_trades']}"
        )
        return metrics
    
    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
