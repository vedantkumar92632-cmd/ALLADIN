"""
Regime cross-validation test: SJM (Strategy 7) vs GMM.
Checks label stability and transition frequency to ensure SJM doesn't over-switch regimes.
"""
import sys
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

sys.path.append(r'c:\Users\vedan\OneDrive\Desktop\artha')

from strategy.regime.sjm import SJMRegimeDetector
from adapters.data_feeds.yfinance_adapter import fetch_ohlcv
from data_engine.ingest import ingest_raw_data

def test_regime_crossval():
    print("=" * 60)
    print("REGIME STABILITY CROSS-VALIDATION (SJM vs GMM)")
    print("=" * 60)
    
    # Fetch sample data for RELIANCE
    raw = fetch_ohlcv(["RELIANCE.NS"], "2025-01-01", "2026-08-01")
    bars = ingest_raw_data(raw)
    records = [{'date': b.timestamp, 'close': b.close_price} for b in bars]
    df = pd.DataFrame(records).set_index('date').sort_index()
    returns = df['close'].pct_change().dropna()
    
    features = pd.DataFrame({'ret': returns})
    features['vol'] = returns.rolling(20).std()
    features = features.dropna()
    
    # 1. Fit SJM (Strategy 7)
    sjm = SJMRegimeDetector(n_components=2, jump_penalty=3.0)
    sjm.fit(features)
    sjm_regimes = sjm.predict(features)
    
    # 2. Fit GMM benchmark
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(features)
    
    # Count regime switches (flip frequency)
    sjm_switches = np.sum(np.diff(sjm_regimes) != 0)
    gmm_switches = np.sum(np.diff(gmm_labels) != 0)
    
    print(f"Total observations: {len(features)}")
    print(f"SJM (Strategy 7) regime switches: {sjm_switches} ({sjm_switches/len(features):.1%} flip rate)")
    print(f"GMM benchmark regime switches:    {gmm_switches} ({gmm_switches/len(features):.1%} flip rate)")
    
    # SJM jump penalty should reduce over-switching compared to standard unconstrained GMM/HMM
    assert sjm_switches <= gmm_switches, "SJM should have equal or lower switch frequency due to jump penalty"
    print("\n[OK] PASS: Strategy 7 (SJM) demonstrates superior regime stability (lower flip rate) over unconstrained GMM.")

if __name__ == "__main__":
    test_regime_crossval()
