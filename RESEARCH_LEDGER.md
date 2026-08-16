# ARTHA V1 Research Ledger

| ID | Source | Type | License | Maintained? | One-line verdict |
|---|---|---|---|---|---|
| C8 | yfinance | DEPENDENCY | Apache 2.0 | Yes | Standard free OHLCV bar fetcher, used as bootstrap data source |
| D1.1 | CatBoost (catboost) | DEPENDENCY | Apache 2.0 | Yes | Domain 1 alpha model — ordered boosting with native categorical support for NSE sector codes |
| D1.2 | LightGBM (lightgbm) | DEPENDENCY | MIT | Yes | Domain 1 alpha model — fast leaf-wise GBM for rolling window retraining |
| D2.1 | jumpmodels (PyPI) | DEPENDENCY | MIT | Yes (v0.1.1) | Domain 2 SJM regime detection — official Princeton implementation, scikit-learn API |
| D2.2 | Adams & MacKay 2007 BOCPD | REFERENCE_PATTERN | N/A | N/A | Domain 2 changepoint detection — re-implemented from algorithm paper, no vendored code |
| D3.1 | arch (PyPI) | DEPENDENCY | NCSA | Yes (v8.0.0) | Domain 3 EGARCH/GJR-GARCH volatility — standard Python library for ARCH/GARCH family models |
| D3.2 | HAR-RV | DEFERRED | N/A | N/A | Domain 3 — requires intraday tick data not available via yfinance bootstrap; defer until TrueData/Kite added |
| D4.1 | Ornstein-Uhlenbeck (MLE) | REFERENCE_PATTERN | N/A | N/A | Domain 4 mean reversion — MLE-estimated OU process with half-life computation |
| D4.2 | Kalman Filter (dynamic hedge) | REFERENCE_PATTERN | N/A | N/A | Domain 4 pairs trading — state-space model for adaptive hedge ratio estimation |
| D5.1 | vaderSentiment (PyPI) | DEPENDENCY | MIT | Yes (v3.3.2) | Domain 5 NLP pre-screen gate — filters neutral text before FinBERT, Value=4 |
| D5.2 | ProsusAI/finbert (HuggingFace) | DEPENDENCY | Apache 2.0 | Yes | Domain 5 FinBERT sentiment — ONNX quantized target for CPU deployment |
| D5.3 | spaCy + en_core_web_sm | DEPENDENCY | MIT | Yes (v3.8.14) | Domain 5 Financial NER — rule-based Indian entity matching baseline, fine-tuning planned |
| D5.4 | faiss-cpu (PyPI) | DEPENDENCY | MIT | Yes (v1.14.3) | Domain 5 semantic similarity search over historical news/events |
| D6.1 | Historical CVaR | REFERENCE_PATTERN | N/A | N/A | Domain 6 risk — 99% CVaR on portfolio returns, 252-day rolling |
| D6.2 | Half-Kelly Criterion | REFERENCE_PATTERN | N/A | N/A | Domain 6 risk — 30% shrinkage, 5% max cap per position (Thorp's recommendation) |
| D6.3 | Parametric VaR | REFERENCE_PATTERN | N/A | N/A | Domain 6 risk — EGARCH sigma fed, forward-looking complement to historical CVaR |
| D6.4 | ATR Stop-Loss | REFERENCE_PATTERN | N/A | N/A | Domain 6 risk — 2.0-3.0x ATR for NSE, factors in circuit breakers |
| D7.1 | PyPortfolioOpt (PyPI) | DEPENDENCY | MIT | Yes (v1.6.0) | Domain 7 portfolio — HRP + Black-Litterman optimization |
| D8.1 | vectorbt (PyPI) | DEPENDENCY | AGPL-3.0 | Yes (v1.1.0) | Domain 8 backtest — fast vectorized parameter sweeps |
| D8.2 | zipline-reloaded | DEFERRED | Apache 2.0 | Yes | Domain 8 — heavy Windows compatibility issues; event-driven engine built in-house instead |
| D8.3 | pyfolio-reloaded | DEFERRED | Apache 2.0 | Yes | Domain 8 — requires zipline ecosystem; deferred with zipline |
| D8.4 | CPCV (Lopez de Prado) | REFERENCE_PATTERN | N/A | N/A | Domain 8 validation — Combinatorial Purged Cross-Validation for time-series ML |
| D8.5 | DSR + PSR (Bailey & LdP) | REFERENCE_PATTERN | N/A | N/A | Domain 8 validation — Deflated Sharpe Ratio corrects for multiple testing bias |
| D8.6 | Monte Carlo Permutation | REFERENCE_PATTERN | N/A | N/A | Domain 8 validation — strategy significance testing via return shuffling |
| D9.1 | Almgren-Chriss | REFERENCE_PATTERN | N/A | N/A | Domain 11 — pre-trade market impact cost estimator (not a live execution algo) |
| D9.2 | OpenAlgo | REFERENCE_PATTERN | N/A | N/A | Domain 15 — unified Indian broker API adapter, paper mode only |
| D9.3 | NautilusTrader | DEFERRED | LGPL-3.0 | Yes | Domain 15 — architecture swap requires human sign-off per Rule Zero |
| D10.1 | Rolling IC Decay | REFERENCE_PATTERN | N/A | N/A | Domain 12 — Spearman rank IC tracking for alpha model decay detection |
| D10.2 | Correlation Spike Detector | REFERENCE_PATTERN | N/A | N/A | Domain 14 — pairwise + sector correlation monitoring for systemic risk |

