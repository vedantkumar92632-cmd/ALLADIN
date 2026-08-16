# ALLADIN V1 — Strategy Registry

**Rule**: All externally-sourced strategies are numbered `Strategy N`.
Founder-originated strategies get a founder-chosen name. See ADDENDUM in MASTER_REFERENCE.md.

---

| Strategy ID | Name in Code | Underlying Source / Technique | Domain | Key Reference |
|-------------|-------------|-------------------------------|--------|---------------|
| Strategy 1 | `strategy_1` | CatBoost gradient boosting | Alpha / ML | Awesome CatBoost, ML4T (Stefan Jansen) |
| Strategy 2 | `strategy_2` | LightGBM gradient boosting | Alpha / ML | ML4T, Qlib (Microsoft Research), LGBM-Stock-Prediction |
| Strategy 3 | `strategy_3` | EGARCH(1,1) conditional volatility | Volatility | `arch` library, BSE Sensex research |
| Strategy 4 | `strategy_4` | GJR-GARCH(1,1) asymmetric volatility | Volatility | `arch` library, NSE Nifty research |
| Strategy 5 | `strategy_5` | Ornstein-Uhlenbeck mean reversion | Mean Reversion | Statistical arbitrage / cointegration literature |
| Strategy 6 | `strategy_6` | Kalman Filter dynamic pairs trading | Mean Reversion | Kalman 1960, pairs trading literature |
| Strategy 7 | `strategy_7` | Statistical Jump Model regime detection | Regime | `jumpmodels` library |
| Strategy 8 | `strategy_8` | Bayesian Online Changepoint Detection | Regime | Adams & MacKay 2007 |

---

## NLP Signal Components (not standalone strategies — they feed into alpha models)

These are signal inputs, not tradeable strategies themselves. They keep descriptive names per the addendum:

| Component | File | Purpose |
|-----------|------|---------|
| VADER Gate | `alpha/ml_model/vader_gate.py` | Pre-filter for NLP relevance |
| FinBERT Sentiment | `alpha/ml_model/finbert_sentiment.py` | Deep learning sentiment scoring |
| Financial NER | `alpha/ml_model/financial_ner.py` | Entity extraction from news |
| NLP Pipeline | `alpha/ml_model/nlp_pipeline.py` | Orchestrates NER + sentiment |
| Similarity Search | `alpha/ml_model/similarity_search.py` | FAISS vector similarity |

## Infrastructure (keeps descriptive names — not strategies)

Message Bus, OMS, Clock, Cache, Kernel, Data Engine, Risk modules, Portfolio optimizer, Backtest engine, etc. — all keep their current descriptive names per the addendum.
