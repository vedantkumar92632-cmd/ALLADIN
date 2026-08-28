"""
ALLADIN V1 - Main Orchestrator
==============================

Wires all system components through the Kernel/MessageBus event-driven
architecture. Supports two modes:

    BACKTEST mode: Replays historical data, generates signals, executes
                   simulated trades, validates with CPCV/DSR/PSR.

    PAPER mode:    Fetches latest data, generates live signals, routes
                   through OMS in paper mode (no real capital).

Usage:
    cd c:\\Users\\vedan\\OneDrive\\Desktop\\alladin

    # Backtest mode (default):
    .venv\\Scripts\\python main.py --mode backtest --tickers RELIANCE.NS TCS.NS INFY.NS

    # Paper trading mode:
    .venv\\Scripts\\python main.py --mode paper --tickers RELIANCE.NS TCS.NS

    # Quick test with defaults:
    .venv\\Scripts\\python main.py

Per MASTER_REFERENCE Part 1: NO LIVE TRADING WITH REAL CAPITAL.
Founder decides go-live. This system is paper-only until then.
"""
import sys
import os
import argparse
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -- Core --
from core.logger import get_logger
from core.time_utils import IST, is_trading_day

# -- Engine --
from engine.kernel import kernel
from engine.message_bus import message_bus
from engine.event_types import (
    MarketDataEvent, SignalEvent, OrderEvent, FillEvent, HaltEvent
)
from engine.cache import cache

# -- Data Pipeline --
from adapters.data_feeds.yfinance_adapter import fetch_ohlcv
from data_engine.ingest import ingest_raw_data

# -- Alpha / Strategy Models --
from alpha.volatility.egarch_model import EGARCHModel
from alpha.volatility.gjrgarch_model import GJRGARCHModel
from alpha.volatility.vol_selector import select_best_vol_model
from alpha.mean_reversion.ou_process import OUProcess
from strategy.regime.sjm import SJMRegimeDetector
from strategy.regime.bocpd import BOCPD

# -- Decision Pipeline --
from decision.signal_score import compute_signal_score
from decision.conviction_score import compute_conviction
from decision.ranking import rank_by_risk_adjusted

# -- Risk --
from risk.exposure import historical_cvar, historical_var
from risk.drawdown import DrawdownMonitor
from risk.pre_trade import pre_trade_check

# -- Portfolio --
from portfolio.constructor import half_kelly

# -- Backtest --
from backtest.engine import SimpleBacktest
from backtest.validation.dsr_psr import probabilistic_sharpe_ratio, deflated_sharpe_ratio
from backtest.validation.cpcv import combinatorial_purged_cv
from backtest.validation.monte_carlo import monte_carlo_permutation_test
from backtest.validation.tearsheet import generate_tearsheet

# -- Execution --
from execution.oms import OrderManager

# -- Monitoring --
from monitoring.ic_decay import ICDecayTracker
from monitoring.correlation import SectorCorrelationEngine

logger = get_logger("main")


# =====================================================================
# ORCHESTRATOR CLASS
# =====================================================================
class AlladinOrchestrator:
    """
    Wires all ALLADIN V1 components through the MessageBus.

    Flow:
        Data -> MarketDataEvent -> Alpha Models -> SignalEvent
        -> Decision Pipeline -> Risk Check -> OrderEvent
        -> OMS -> FillEvent -> Monitoring
    """

    def __init__(self, tickers, lookback_days=365):
        self.tickers = tickers
        self.lookback_days = lookback_days
        self.is_halted = False

        # Components
        self.oms = OrderManager()
        self.drawdown_monitor = DrawdownMonitor()
        self.ic_tracker = ICDecayTracker()

        # Alpha models (fitted per ticker)
        self.vol_models = {}
        self.regime_model = None
        self.regime_labels = None

        # Data
        self.prices = None
        self.returns = None

        # Wire message bus handlers
        self._wire_event_handlers()

    def _wire_event_handlers(self):
        """Subscribe component handlers to the MessageBus."""
        message_bus.subscribe(HaltEvent, self._on_halt)
        message_bus.subscribe(FillEvent, self._on_fill)
        logger.info("Orchestrator event handlers wired to MessageBus.")

    def _on_halt(self, event: HaltEvent):
        """Handle risk halt events."""
        logger.warning(f"HALT EVENT: {event.reason} (severity={event.severity})")
        self.is_halted = True
        self.oms.halt_trading()

    def _on_fill(self, event: FillEvent):
        """Handle fill events for monitoring."""
        self.drawdown_monitor.update(event.fill_price)

    # =================================================================
    # DATA PIPELINE
    # =================================================================
    def fetch_and_prepare_data(self):
        """Step 1: Fetch OHLCV, ingest, build price/return matrices."""
        logger.info(f"Fetching data for {len(self.tickers)} tickers, {self.lookback_days}d lookback...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")

        raw = fetch_ohlcv(self.tickers, start_date, end_date)
        if raw.empty:
            raise RuntimeError("No data fetched. Check internet and tickers.")

        bars = ingest_raw_data(raw)
        logger.info(f"Ingested {len(bars)} bars.")

        # Build price matrix
        records = [{'date': b.timestamp, 'ticker': b.ticker, 'close': b.close_price,
                     'volume': b.volume} for b in bars]
        df = pd.DataFrame(records)
        self.prices = df.pivot(index='date', columns='ticker', values='close')
        self.prices = self.prices.sort_index().ffill().dropna()
        self.returns = self.prices.pct_change().dropna()

        logger.info(f"Price matrix: {self.prices.shape[0]} dates x {self.prices.shape[1]} tickers")
        return self.prices, self.returns

    # =================================================================
    # ALPHA MODEL FITTING
    # =================================================================
    def fit_alpha_models(self):
        """Step 2: Fit volatility models per ticker + regime model."""
        logger.info("Fitting alpha models...")

        # Fit best volatility model per ticker (Strategy 3 vs Strategy 4)
        for ticker in self.returns.columns:
            rets_scaled = self.returns[ticker] * 100
            try:
                best_model, best_name, _, _ = select_best_vol_model(rets_scaled)
                if best_model is not None:
                    self.vol_models[ticker] = best_model
                    logger.info(f"  {ticker}: {best_name} (AIC={best_model.aic:.1f})")
                else:
                    logger.warning(f"  {ticker}: Both vol models failed")
            except Exception as e:
                logger.warning(f"  {ticker}: Vol model failed - {e}")

        # Fit regime model (Strategy 7) on first ticker's features
        col = self.returns.columns[0]
        features = self.returns[[col]].copy()
        features['vol_20d'] = self.returns[col].rolling(20).std()
        features = features.dropna()

        self.regime_model = SJMRegimeDetector(n_components=2)
        self.regime_model.fit(features)
        self.regime_labels = pd.Series(
            self.regime_model.predict(features),
            index=features.index
        )
        regime_dist = self.regime_labels.value_counts().to_dict()
        logger.info(f"  Regime distribution: {regime_dist}")

    # =================================================================
    # SIGNAL GENERATION
    # =================================================================
    def generate_signals(self, bar_date):
        """
        Generate composite signals for all tickers at a given date.
        Combines volatility regime + regime detection into a composite score.
        """
        signals = {}

        for ticker in self.returns.columns:
            domain_signals = {}

            # Volatility signal: low vol = bullish, high vol = bearish
            if ticker in self.vol_models:
                model = self.vol_models[ticker]
                rets = self.returns[ticker].loc[:bar_date] * 100
                if len(rets) > 50:
                    vol = model.predict(rets)
                    current_vol = vol[-1] if len(vol) > 0 else 0
                    vol_mean = np.mean(vol)
                    vol_signal = -1.0 if current_vol > vol_mean * 1.2 else (
                        1.0 if current_vol < vol_mean * 0.8 else 0.0
                    )
                    domain_signals['volatility'] = vol_signal

            # Regime signal: regime 0 = bullish, regime 1 = bearish
            if bar_date in self.regime_labels.index:
                regime = self.regime_labels.loc[bar_date]
                domain_signals['regime'] = 1.0 if regime == 0 else -1.0

            # Momentum signal: 20-day return
            rets_slice = self.returns[ticker].loc[:bar_date]
            if len(rets_slice) >= 20:
                mom_20 = rets_slice.iloc[-20:].sum()
                domain_signals['momentum'] = max(-1.0, min(1.0, mom_20 * 10))

            # Composite score
            if domain_signals:
                score = compute_signal_score(ticker, domain_signals)
                signals[ticker] = score

        return signals

    # =================================================================
    # DECISION PIPELINE
    # =================================================================
    def run_decision_pipeline(self, signals, current_prices):
        """
        Signal -> Conviction -> Ranking -> Risk Check -> Orders.
        Returns list of trade decisions.
        """
        if self.is_halted:
            logger.warning("Trading halted. No new orders.")
            return []

        decisions = []
        for ticker, score in signals.items():
            # Conviction
            conviction = compute_conviction(score, n_agreeing_models=1, total_models=3)
            if abs(conviction) < 20:
                continue  # Below threshold

            # Risk check
            price = current_prices.get(ticker, 0)
            if price <= 0:
                continue

            direction = "BUY" if score > 0 else "SELL"
            decisions.append({
                'ticker': ticker,
                'score': score,
                'conviction': conviction,
                'direction': direction,
                'price': price,
            })

        # Rank by absolute score
        decisions.sort(key=lambda x: abs(x['score']), reverse=True)
        return decisions[:5]  # Top 5 trades max

    # =================================================================
    # BACKTEST MODE
    # =================================================================
    def run_backtest(self):
        """Run full backtest with validation."""
        print("\n" + "=" * 60)
        print("  ALLADIN V1 - BACKTEST MODE")
        print("=" * 60)

        # Step 1: Data
        print("\n[1/6] Fetching data...")
        self.fetch_and_prepare_data()
        print(f"      {self.prices.shape[0]} dates x {self.prices.shape[1]} tickers")

        # Step 2: Fit models
        print("\n[2/6] Fitting alpha models...")
        self.fit_alpha_models()
        print(f"      {len(self.vol_models)} vol models + 1 regime model fitted")

        # Step 3: Run backtest engine
        print("\n[3/6] Running backtest...")
        orchestrator = self

        def signal_fn(data, bar_idx):
            if bar_idx < 1 or bar_idx >= len(orchestrator.returns.index):
                return {}
            bar_date = data.index[-1]
            raw_signals = orchestrator.generate_signals(bar_date)
            # Convert to +1/-1/0
            return {t: (1 if s > 0.1 else (-1 if s < -0.1 else 0))
                    for t, s in raw_signals.items()}

        bt = SimpleBacktest(
            initial_capital=1_000_000,
            slippage_bps=5,
            commission_bps=3,
        )
        metrics = bt.run(self.prices, signal_fn)

        print(f"      Total Return:  {metrics['total_return']:.2%}")
        print(f"      Sharpe Ratio:  {metrics['sharpe']:.2f}")
        print(f"      Max Drawdown:  {metrics['max_drawdown']:.2%}")
        print(f"      Total Trades:  {metrics['n_trades']}")
        print(f"      Final Equity:  Rs {metrics['final_equity']:,.0f}")

        # Step 4: Validation
        print("\n[4/6] Validating...")
        daily_rets = np.array(bt.daily_returns)

        psr = probabilistic_sharpe_ratio(daily_rets)
        dsr = deflated_sharpe_ratio(daily_rets, n_trials=1)
        print(f"      PSR:  {psr:.4f} (>0.95 = statistically significant)")
        print(f"      DSR:  {dsr:.4f}")

        splits = combinatorial_purged_cv(
            n_samples=len(self.prices), n_groups=5,
            n_test_groups=2, purge_gap=5
        )
        print(f"      CPCV: {len(splits)} combinatorial splits generated")

        if len(daily_rets) > 20:
            mc = monte_carlo_permutation_test(daily_rets, n_simulations=1000)
            print(f"      Monte Carlo p-value: {mc['p_value']:.4f}")

        # Step 5: Risk metrics
        print("\n[5/6] Risk metrics...")
        var_95 = historical_var(pd.Series(daily_rets))
        cvar_95 = historical_cvar(pd.Series(daily_rets))
        print(f"      VaR  (95%): {var_95:.4%}")
        print(f"      CVaR (95%): {cvar_95:.4%}")

        # Step 6: Summary & Tearsheet
        print("\n[6/6] Strategy summary & tearsheet generation...")
        mc_p = mc['p_value'] if 'mc' in locals() and isinstance(mc, dict) else 1.0
        generate_tearsheet(
            metrics=metrics,
            psr=psr,
            dsr=dsr,
            mc_p_value=mc_p,
            var_95=var_95,
            cvar_95=cvar_95,
            output_filepath="tearsheet.md"
        )
        print("      Saved tearsheet.md")
        print(f"      Strategies used: Strategy 3/4 (Vol), Strategy 7 (Regime)")
        print(f"      See STRATEGY_REGISTRY.md for full mapping")

        print("\n" + "=" * 60)
        print("  BACKTEST COMPLETE")
        print("=" * 60)

        return metrics

    # =================================================================
    # PAPER TRADING MODE
    # =================================================================
    def run_paper_session(self):
        """Run one paper trading session (latest data -> signals -> orders)."""
        print("\n" + "=" * 60)
        print("  ALLADIN V1 - PAPER TRADING SESSION")
        print("=" * 60)

        # Step 1: Data
        print("\n[1/4] Fetching latest data...")
        self.fetch_and_prepare_data()

        # Step 2: Fit models on latest data
        print("\n[2/4] Fitting models on latest data...")
        self.fit_alpha_models()

        # Step 3: Generate signals for today
        print("\n[3/4] Generating signals...")
        latest_date = self.returns.index[-1]
        signals = self.generate_signals(latest_date)
        current_prices = self.prices.iloc[-1].to_dict()

        print(f"      Date: {latest_date}")
        for ticker, score in sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
            print(f"      {ticker}: score={score:+.3f} ({direction})")

        # Step 4: Decision pipeline
        print("\n[4/4] Decision pipeline...")
        decisions = self.run_decision_pipeline(signals, current_prices)

        if not decisions:
            print("      No actionable trades above conviction threshold.")
        else:
            print(f"      {len(decisions)} trade(s) generated (PAPER ONLY):")
            for d in decisions:
                print(f"        {d['direction']} {d['ticker']} @ Rs{d['price']:.2f} "
                      f"(conviction={d['conviction']}, score={d['score']:+.3f})")

            # Create paper orders via OMS
            for d in decisions:
                qty = max(1, int(50000 / d['price']))  # ~50K per position
                order = self.oms.create_order(
                    d['ticker'], d['direction'], qty, "MARKET", d['price']
                )
                if order:
                    print(f"        -> Paper order: {order.order_id}")

        # Risk snapshot
        print(f"\n      Portfolio VaR (95%): {historical_var(pd.Series(self.returns.mean())):.4%}")
        print(f"      System halted: {self.is_halted}")

        print("\n" + "=" * 60)
        print("  PAPER SESSION COMPLETE")
        print("  NO REAL CAPITAL AT RISK. Founder decides go-live.")
        print("=" * 60)

        return decisions


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="ALLADIN V1 Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="NO LIVE TRADING. Paper mode only. Founder decides go-live."
    )
    parser.add_argument(
        '--mode', choices=['backtest', 'paper'], default='backtest',
        help='Run mode: backtest (historical) or paper (latest data, no real trades)'
    )
    parser.add_argument(
        '--tickers', nargs='+',
        default=['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS'],
        help='NSE tickers (append .NS suffix)'
    )
    parser.add_argument(
        '--lookback', type=int, default=365,
        help='Lookback period in days (default: 365)'
    )

    args = parser.parse_args()

    # Boot kernel
    kernel.start()

    try:
        orchestrator = AlladinOrchestrator(
            tickers=args.tickers,
            lookback_days=args.lookback,
        )

        if args.mode == 'backtest':
            orchestrator.run_backtest()
        elif args.mode == 'paper':
            orchestrator.run_paper_session()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        kernel.stop()


if __name__ == "__main__":
    main()
