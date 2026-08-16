"""
Pairwise Correlation Spike Detector (L9.S002) +
Cross-Asset & Sector Correlation Engine (L9.S003).

Per Part 3B Domain 14:
- Correlation spikes during market stress effectively increase portfolio risk
  beyond what position-level risk metrics show.
- NSE stocks are highly correlated during crash episodes.
- Sector correlation tracking is essential for NSE where banking sector alone
  can be 30%+ of Nifty 50.

L9.S003 is the more involved version — tracks cross-asset and sector
correlations over rolling windows and alerts on regime changes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.logger import get_logger

logger = get_logger("monitoring.correlation")


class CorrelationSpikeDetector:
    """
    Detects pairwise correlation spikes that indicate systemic risk.
    
    During market stress, correlations converge toward 1.0 — this is
    exactly when diversification benefits disappear.
    """

    def __init__(
        self,
        lookback: int = 60,
        spike_threshold: float = 0.8,
        alert_pct: float = 0.30,
    ):
        """
        Args:
            lookback: Rolling window for correlation computation.
            spike_threshold: Correlation level considered a "spike".
            alert_pct: Alert if this fraction of pairs exceed threshold.
        """
        self.lookback = lookback
        self.spike_threshold = spike_threshold
        self.alert_pct = alert_pct
        self._alerts: List[dict] = []
        
        logger.info(
            f"CorrelationSpikeDetector: lookback={lookback}, "
            f"threshold={spike_threshold}, alert_pct={alert_pct:.0%}"
        )

    def check(self, returns: pd.DataFrame) -> dict:
        """
        Check for correlation spikes in recent data.
        
        Args:
            returns: DataFrame of asset returns (columns = tickers).
            
        Returns:
            Dict with correlation matrix stats and alert status.
        """
        recent = returns.iloc[-self.lookback:]
        
        if len(recent) < 10:
            return {'n_pairs': 0, 'alert': False}
        
        corr = recent.corr()
        
        # Get upper triangle (exclude diagonal)
        n = len(corr)
        pairs = []
        spiked_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                r = corr.iloc[i, j]
                if not np.isnan(r):
                    pairs.append((corr.columns[i], corr.columns[j], r))
                    if abs(r) >= self.spike_threshold:
                        spiked_pairs.append((corr.columns[i], corr.columns[j], r))
        
        n_pairs = len(pairs)
        n_spiked = len(spiked_pairs)
        spike_pct = n_spiked / n_pairs if n_pairs > 0 else 0
        
        is_alert = spike_pct >= self.alert_pct
        
        if is_alert:
            alert = {
                'type': 'correlation_spike',
                'n_spiked': n_spiked,
                'n_pairs': n_pairs,
                'spike_pct': round(spike_pct, 4),
                'top_spiked': spiked_pairs[:5],
            }
            self._alerts.append(alert)
            logger.warning(
                f"CORRELATION SPIKE: {n_spiked}/{n_pairs} pairs ({spike_pct:.0%}) "
                f"exceed {self.spike_threshold} threshold"
            )
        
        avg_corr = np.mean([abs(p[2]) for p in pairs]) if pairs else 0
        
        return {
            'n_pairs': n_pairs,
            'n_spiked': n_spiked,
            'spike_pct': round(spike_pct, 4),
            'avg_abs_corr': round(avg_corr, 4),
            'alert': is_alert,
            'top_spiked': spiked_pairs[:5],
        }

    def get_alerts(self) -> List[dict]:
        return self._alerts


class SectorCorrelationEngine:
    """
    Cross-asset and sector correlation monitoring (L9.S003).
    
    Tracks correlation structure at the sector level — critical for NSE
    where banking alone can be 30%+ of Nifty 50.
    """

    def __init__(
        self,
        sector_map: Optional[Dict[str, str]] = None,
        lookback: int = 60,
    ):
        """
        Args:
            sector_map: {ticker: sector_name} mapping.
            lookback: Rolling window for sector correlation.
        """
        self.sector_map = sector_map or {
            # Default NSE sector mapping (top Nifty 50 stocks)
            'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking',
            'KOTAKBANK': 'Banking', 'AXISBANK': 'Banking',
            'TCS': 'IT', 'INFY': 'IT', 'WIPRO': 'IT', 'HCLTECH': 'IT',
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy',
            'ITC': 'FMCG', 'HINDUNILVR': 'FMCG', 'NESTLEIND': 'FMCG',
            'BHARTIARTL': 'Telecom', 'LT': 'Infrastructure',
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma',
        }
        self.lookback = lookback
        self._sector_corr_history: List[dict] = []

    def compute_sector_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Compute equal-weighted sector returns from stock returns."""
        sectors = {}
        for ticker in returns.columns:
            sector = self.sector_map.get(ticker, 'Other')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        
        sector_returns = {}
        for sector, tickers in sectors.items():
            available = [t for t in tickers if t in returns.columns]
            if available:
                sector_returns[sector] = returns[available].mean(axis=1)
        
        return pd.DataFrame(sector_returns)

    def check_sector_correlations(self, returns: pd.DataFrame) -> dict:
        """
        Compute and check sector-level correlations.
        
        Returns:
            Dict with sector correlation matrix and risk assessment.
        """
        sector_ret = self.compute_sector_returns(returns.iloc[-self.lookback:])
        
        if sector_ret.shape[1] < 2:
            return {'n_sectors': sector_ret.shape[1], 'alert': False}
        
        corr = sector_ret.corr()
        
        # Find highly correlated sectors
        high_corr_pairs = []
        n = len(corr)
        for i in range(n):
            for j in range(i + 1, n):
                r = corr.iloc[i, j]
                if not np.isnan(r) and abs(r) > 0.7:
                    high_corr_pairs.append((corr.columns[i], corr.columns[j], round(r, 4)))
        
        avg_corr = np.nanmean(corr.values[np.triu_indices(n, k=1)])
        
        result = {
            'n_sectors': n,
            'avg_cross_sector_corr': round(float(avg_corr), 4),
            'high_corr_pairs': high_corr_pairs,
            'alert': len(high_corr_pairs) > 0,
            'sector_names': list(corr.columns),
        }
        
        self._sector_corr_history.append(result)
        
        if high_corr_pairs:
            logger.warning(
                f"SECTOR CORRELATION: {len(high_corr_pairs)} highly correlated pairs: "
                f"{high_corr_pairs[:3]}"
            )
        
        return result
