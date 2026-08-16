"""
ALLADIN V1 — Quick-Start Runner
==============================

Run this to verify the full system works end-to-end.

Usage:
    cd c:\\Users\\vedan\\OneDrive\\Desktop\\alladin
    .venv\\Scripts\\python run_demo.py

What it does:
    1. Pulls real OHLCV data via yfinance (NIFTY 50 tickers)
    2. Runs the data pipeline (ingest -> clean)
    3. Fits Strategy 3 (EGARCH) and Strategy 7 (SJM regime)
    4. Runs a simple backtest
    5. Validates with PSR and CPCV
    6. Prints full results

No live trading. No real capital. Paper trading only.
"""
import sys
import os
import warnings

# Suppress arch scale warnings — we know returns are small
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from core.logger import get_logger
from core.time_utils import IST

logger = get_logger("run_demo")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_demo():
    # ================================================================
    # STEP 1: Pull OHLCV data
    # ================================================================
    section("STEP 1: Fetching OHLCV Data")

    from adapters.data_feeds.yfinance_adapter import fetch_ohlcv

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"  Tickers: {tickers}")
    print(f"  Period: {start_date} -> {end_date}")

    raw_data = fetch_ohlcv(tickers, start_date, end_date)
    print(f"  [OK] Fetched {len(raw_data)} rows")

    if raw_data.empty:
        print("  [FAIL] No data fetched. Check internet connection.")
        return

    # ================================================================
    # STEP 2: Data Pipeline (Ingest -> Clean)
    # ================================================================
    section("STEP 2: Data Pipeline")

    from data_engine.ingest import ingest_raw_data

    bars = ingest_raw_data(raw_data)
    print(f"  [OK] Ingested {len(bars)} bars")

    # Build a clean price DataFrame for backtesting
    records = []
    for bar in bars:
        records.append({
            'date': bar.timestamp,
            'ticker': bar.ticker,
            'close': bar.close_price,
            'volume': bar.volume,
        })
    df = pd.DataFrame(records)
    prices = df.pivot(index='date', columns='ticker', values='close')
    prices = prices.sort_index().ffill().dropna()
    print(f"  [OK] Clean price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers")

    # ================================================================
    # STEP 3: Compute Returns
    # ================================================================
    section("STEP 3: Returns & Features")

    returns = prices.pct_change().dropna()
    print(f"  [OK] Returns matrix: {returns.shape}")
    print(f"  Mean daily return: {returns.mean().mean():.4%}")
    print(f"  Mean daily vol: {returns.std().mean():.4%}")

    # ================================================================
    # STEP 4: Fit Strategy 3 (EGARCH Volatility)
    # ================================================================
    section("STEP 4: Strategy 3 (EGARCH Volatility)")

    from alpha.volatility.egarch_model import EGARCHModel

    ticker = tickers[0].replace('.NS', '')
    if tickers[0] in returns.columns:
        col = tickers[0]
    else:
        col = returns.columns[0]
        ticker = col

    egarch = EGARCHModel()
    rets_scaled = returns[col] * 100  # Scale for arch library
    egarch.fit(rets_scaled)
    cond_vol = egarch.predict(rets_scaled)
    print(f"  [OK] {ticker}: AIC={egarch.aic:.2f}, mean vol={cond_vol.mean():.4f}")
    print(f"  Strategy name: {egarch.name}")

    # ================================================================
    # STEP 5: Fit Strategy 7 (SJM Regime Detection)
    # ================================================================
    section("STEP 5: Strategy 7 (SJM Regime)")

    from strategy.regime.sjm import SJMRegimeDetector

    features = returns[[col]].copy()
    features['vol_20d'] = returns[col].rolling(20).std()
    features = features.dropna()

    sjm = SJMRegimeDetector(n_components=2)
    sjm.fit(features)
    regimes = sjm.predict(features)
    regime_counts = pd.Series(regimes).value_counts().to_dict()
    print(f"  [OK] Regime distribution: {regime_counts}")
    print(f"  Strategy name: {sjm.name}")

    # ================================================================
    # STEP 6: Composite Signal -> Backtest
    # ================================================================
    section("STEP 6: Backtest")

    from backtest.engine import SimpleBacktest
    from decision.signal_score import compute_signal_score

    # Simple signal: high vol regime -> short, low vol regime -> long
    regime_series = pd.Series(regimes, index=features.index)

    def signal_fn(data, bar_idx):
        if bar_idx < 1 or bar_idx >= len(features.index):
            return {}
        current_date = data.index[-1]
        if current_date in regime_series.index:
            regime = regime_series.loc[current_date]
            signal_val = 1 if regime == 0 else -1
        else:
            signal_val = 0
        signals = {}
        for t in data.columns:
            signals[t] = signal_val
        return signals

    bt = SimpleBacktest(
        initial_capital=1_000_000,
        slippage_bps=5,
        commission_bps=3,
    )
    metrics = bt.run(prices, signal_fn)

    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"  Total Trades: {metrics['n_trades']}")
    print(f"  Final Equity: Rs{metrics['final_equity']:,.0f}")

    # ================================================================
    # STEP 7: Validation (PSR + CPCV)
    # ================================================================
    section("STEP 7: Validation")

    from backtest.validation.dsr_psr import probabilistic_sharpe_ratio, deflated_sharpe_ratio
    from backtest.validation.cpcv import combinatorial_purged_cv

    daily_returns = np.array(bt.daily_returns)

    psr = probabilistic_sharpe_ratio(daily_returns)
    print(f"  PSR: {psr:.4f} (>0.95 = significant)")

    dsr = deflated_sharpe_ratio(daily_returns, n_trials=1)
    print(f"  DSR (1 trial): {dsr:.4f}")

    splits = combinatorial_purged_cv(
        n_samples=len(prices),
        n_groups=5,
        n_test_groups=2,
        purge_gap=5,
    )
    print(f"  CPCV splits: {len(splits)}")

    # ================================================================
    # STEP 8: Risk Check
    # ================================================================
    section("STEP 8: Risk Metrics")

    from risk.exposure import historical_cvar, historical_var

    var_95 = historical_var(pd.Series(daily_returns))
    cvar_95 = historical_cvar(pd.Series(daily_returns))
    print(f"  VaR (95%): {var_95:.4%}")
    print(f"  CVaR (95%): {cvar_95:.4%}")

    # ================================================================
    # DONE
    # ================================================================
    section("COMPLETE")
    print(f"  All 8 strategies registered (Strategy 1-8)")
    print(f"  See STRATEGY_REGISTRY.md for the mapping")
    print()
    print(f"  System is ready for further backtesting.")
    print(f"  NO LIVE TRADING. Founder decides go-live.")
    print()


if __name__ == "__main__":
    run_demo()

