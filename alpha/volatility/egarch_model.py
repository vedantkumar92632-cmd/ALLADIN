"""
EGARCH(1,1) volatility model with Student-t distribution.

Uses the arch Python package. Student-t distribution is mandatory — never use
Gaussian for Indian market data (fat tails confirmed by multiple studies).

Per Part 3B Domain 3: EGARCH wins on BSE Sensex, GJR-GARCH wins on NSE Nifty.
Fit both per asset, keep whichever wins by AIC — use vol_selector.py for this.

Tuning (Part 3A):
- Distribution: 't' (Student-t) — empirically superior on NSE data
- p=1, q=1 sufficient for daily data
- Refit frequency: Monthly rolling window (20 trading days, lookback min 252 days)
"""
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.volatility.egarch_model")


class EGARCHModel(BaseAlpha):
    """
    EGARCH(1,1) conditional volatility model.
    
    Captures asymmetric volatility (leverage effect) through an exponential
    specification. Does not require non-negativity constraints on parameters.
    """

    @property
    def name(self) -> str:
        return "strategy_3"

    @property
    def domain(self) -> str:
        return "volatility"

    def __init__(self, dist: str = "t", p: int = 1, q: int = 1):
        """
        Args:
            dist: Error distribution. Use 't' (Student-t) for Indian markets.
                  Never use 'normal' — fat tails are confirmed.
            p: ARCH order (default 1).
            q: GARCH order (default 1).
        """
        self.dist = dist
        self.p = p
        self.q = q
        self._fit_result = None
        self._is_fitted = False
        
        logger.info(f"EGARCHModel initialized: p={p}, q={q}, dist='{dist}'")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit EGARCH on returns data.
        
        X_train should be a single-column DataFrame or Series of returns (percentage).
        y_train is ignored (unsupervised model, BaseAlpha compatibility).
        """
        returns = self._extract_returns(X_train)
        
        logger.info(f"Fitting EGARCH({self.p},{self.q}) on {len(returns)} observations...")
        
        model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, dist=self.dist)
        self._fit_result = model.fit(disp='off', options={'maxiter': 500})
        
        self._is_fitted = True
        logger.info(f"EGARCH fitted. AIC={self._fit_result.aic:.4f}, BIC={self._fit_result.bic:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return conditional volatility — in-sample if X matches training data, forecast otherwise."""
        if not self._is_fitted:
            raise RuntimeError("EGARCH model not fitted yet.")
        
        in_sample_vol = self._fit_result.conditional_volatility.values
        
        # If X length matches training, return in-sample fitted values
        if X is not None and len(X) == len(in_sample_vol):
            return in_sample_vol
        
        # Otherwise, re-fit on new data and return new conditional volatility
        returns = self._extract_returns(X)
        model = arch_model(returns, vol='EGARCH', p=self.p, q=self.q, dist=self.dist)
        result = model.fit(disp='off', options={'maxiter': 500})
        return result.conditional_volatility.values

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """
        Out-of-sample volatility forecast.
        
        Args:
            horizon: Number of steps ahead to forecast.
        Returns:
            DataFrame with variance forecasts.
        """
        if not self._is_fitted:
            raise RuntimeError("EGARCH model not fitted yet.")
        
        fcast = self._fit_result.forecast(horizon=horizon)
        return fcast.variance.dropna()

    @property
    def aic(self) -> float:
        """AIC of the fitted model (lower is better)."""
        if not self._is_fitted:
            raise RuntimeError("EGARCH model not fitted yet.")
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        """BIC of the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("EGARCH model not fitted yet.")
        return self._fit_result.bic

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        result = {'p': self.p, 'q': self.q, 'dist': self.dist}
        if self._is_fitted:
            result['aic'] = self._fit_result.aic
            result['bic'] = self._fit_result.bic
        return result

    @staticmethod
    def _extract_returns(X) -> pd.Series:
        """Extract a 1D returns series from various input types."""
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            return X
        else:
            return pd.Series(np.asarray(X).ravel())
