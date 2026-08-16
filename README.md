# ALLADIN V1

ALLADIN V1 is a multi-layered quantitative trading system. It includes data ingestion, alpha signal models, regime detection, volatility analysis, mean reversion, NLP sentiment parsing, portfolio construction, risk management, and execution modules.

## Project Structure
The project is organized in the following components:
- `core/`: Logger, clock, utilities
- `config/`: Configuration settings and schemas
- `engine/`: Event types, message bus, clock, cache, kernel
- `database/`: Database connection and migrations
- `input/`: Input data schemas and pipelines
- `data_engine/`: Data cleaning, ingestion, and feature computation
- `adapters/`: Broker and data feed integrations
- `alpha/`: Machine learning models and signal generators
- `strategy/`: Trading strategies and regime detection
- `decision/`: Composite score, conviction, ranking, filtering, and prioritization
- `portfolio/`: Optimizer (HRP, Black-Litterman) and rebalancer
- `risk/`: Position sizing, CVaR, VaR, drawdowns, and circuit breakers
- `backtest/`: Backtest engine, venue simulator, and validation (CPCV, DSR, PSR)
- `execution/`: Order management, reconciler, slippage monitoring, and execution algorithms
- `data/`: Local cache and SQLite databases

## License
Proprietary.
