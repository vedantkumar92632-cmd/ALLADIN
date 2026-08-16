"""
GJR-GARCH(1,1) volatility model with Student-t distribution.

Calibration challenger to EGARCH — NOT a parallel model. Fit both per asset,
keep whichever wins by AIC. Use vol_selector.py for the comparison.

Per Part 3B Domain 3: GJR-GARCH wins on NSE Nifty, EGARCH wins on BSE Sensex.
For GJR-GARCH, gamma (asymmetry) should be positive and significant (leverage
effect) — if not, EGARCH may fit better for that asset.

Tuning (Part 3A):
- Distribution: 't' (Student-t) — empirically superior on NSE data
- p=1, o=1, q=1 (o=1 triggers GJR asymmetric term)
- Refit frequency: Monthly rolling window
"""
import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.volatility.gjrgarch_model")


class GJRGARCHModel(BaseAlpha):
    """
    GJR-GARCH(1,1) conditional volatility model.
    
    Captures asymmetric volatility through the GJR specification (o=1 parameter).
    Gamma > 0 indicates leverage effect (negative returns increase volatility more).
    """

    @property
    def name(self) -> str:
        return "strategy_4"

    @property
    def domain(self) -> str:
        return "volatility"

    def __init__(self, dist: str = "t", p: int = 1, o: int = 1, q: int = 1):
        """
        Args:
            dist: Error distribution. Use 't' (Student-t) for Indian markets.
            p: ARCH order (default 1).
            o: Asymmetry order (default 1 — triggers GJR term).
            q: GARCH order (default 1).
        """
        self.dist = dist
        self.p = p
        self.o = o
        self.q = q
        self._fit_result = None
        self._is_fitted = False
        
        logger.info(f"GJRGARCHModel initialized: p={p}, o={o}, q={q}, dist='{dist}'")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit GJR-GARCH on returns data.
        
        X_train should be a single-column DataFrame or Series of returns (percentage).
        """
        returns = self._extract_returns(X_train)
        
        logger.info(f"Fitting GJR-GARCH({self.p},{self.o},{self.q}) on {len(returns)} observations...")
        
        # o=1 triggers GJR asymmetric term
        model = arch_model(returns, vol='GARCH', p=self.p, o=self.o, q=self.q, dist=self.dist)
        self._fit_result = model.fit(disp='off', options={'maxiter': 500})
        
        self._is_fitted = True
        logger.info(f"GJR-GARCH fitted. AIC={self._fit_result.aic:.4f}, BIC={self._fit_result.bic:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return conditional volatility — in-sample if X matches training data, forecast otherwise."""
        if not self._is_fitted:
            raise RuntimeError("GJR-GARCH model not fitted yet.")
        
        in_sample_vol = self._fit_result.conditional_volatility.values
        
        # If X length matches training, return in-sample fitted values
        if X is not None and len(X) == len(in_sample_vol):
            return in_sample_vol
        
        # Otherwise, re-fit on new data and return new conditional volatility
        returns = self._extract_returns(X)
        model = arch_model(returns, vol='GARCH', p=self.p, o=self.o, q=self.q, dist=self.dist)
        result = model.fit(disp='off', options={'maxiter': 500})
        return result.conditional_volatility.values

    def forecast(self, horizon: int = 1) -> pd.DataFrame:
        """
        Out-of-sample volatility forecast.
        """
        if not self._is_fitted:
            raise RuntimeError("GJR-GARCH model not fitted yet.")
        fcast = self._fit_result.forecast(horizon=horizon)
        return fcast.variance.dropna()

    @property
    def aic(self) -> float:
        """AIC of the fitted model (lower is better)."""
        if not self._is_fitted:
            raise RuntimeError("GJR-GARCH model not fitted yet.")
        return self._fit_result.aic

    @property
    def bic(self) -> float:
        """BIC of the fitted model."""
        if not self._is_fitted:
            raise RuntimeError("GJR-GARCH model not fitted yet.")
        return self._fit_result.bic

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        result = {'p': self.p, 'o': self.o, 'q': self.q, 'dist': self.dist}
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
