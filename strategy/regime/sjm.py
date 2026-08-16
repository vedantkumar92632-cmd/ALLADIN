"""
Statistical Jump Model (SJM) regime detection.

Replaces HMM for regime classification — SJM applies an explicit jump penalty
at each state transition, forcing the model to switch only when evidence is strong.
Out-of-sample tests show SJM outperforms HMM on Sharpe ratio and max drawdown
after transaction costs (Part 3B Domain 2).

SJM classifies *which* regime you're in (bull/bear/sideways).
BOCPD (bocpd.py) detects the *moment* of a break.
They are complementary, NOT redundant — do NOT merge.

Tuning (Part 3A):
- jump_penalty: Start at 1-5; calibrate so regime durations match NSE cycles
  (Nifty bull cycles avg 18-24 months, bear cycles 6-12 months)
- n_components: 2 (bull/bear) standard; 3 (bull/sideways/bear) optional
- Input features: returns, realized volatility, volume ratio — NOT price levels

Resources:
- jumpmodels (PyPI): Official implementation by paper's authors (Princeton)
- Paper: "Identifying patterns in financial markets" (2024, Annals of Operations Research)
"""
import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from typing import Optional, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("strategy.regime.sjm")

# Default parameters per Part 3A
DEFAULT_PARAMS = {
    'n_components': 2,        # 2 = bull/bear; 3 = bull/sideways/bear
    'jump_penalty': 3.0,      # Range: 1-5; calibrate per asset to match NSE cycles
    'cont': False,            # Use discrete SJM (not Continuous SJM variant)
    'grid_size': 0.05,
    'mode_loss': True,
    'random_state': 42,
    'max_iter': 1000,
    'tol': 1e-8,
    'n_init': 10,
    'verbose': 0,
}


class SJMRegimeDetector(BaseAlpha):
    """
    Statistical Jump Model wrapper for market regime detection.
    
    Classifies time periods into regimes (bull/bear or bull/sideways/bear).
    Feed NIFTY 50 returns + VIX India (when available) + FII flows as inputs.
    Use Sparse SJM variant for multi-feature inputs to avoid overfitting.
    """

    @property
    def name(self) -> str:
        return "strategy_7"

    @property
    def domain(self) -> str:
        return "regime"

    def __init__(
        self,
        n_components: int = 2,
        jump_penalty: float = 3.0,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            n_components: Number of regimes (2=bull/bear, 3=bull/sideways/bear).
            jump_penalty: Penalty for regime switches. Higher = fewer switches.
                         Start at 1-5; calibrate to match NSE cycle durations.
            params: Additional JumpModel parameters to override defaults.
        """
        self._n_components = n_components
        
        merged_params = {**DEFAULT_PARAMS}
        merged_params['n_components'] = n_components
        merged_params['jump_penalty'] = jump_penalty
        if params:
            merged_params.update(params)
        
        self._merged_params = merged_params
        self.model = JumpModel(**merged_params)
        self._is_fitted = False
        
        logger.info(f"SJMRegimeDetector initialized: {n_components} regimes, jump_penalty={jump_penalty}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit the SJM on historical feature data.
        
        SJM is unsupervised — y_train is ignored (kept for BaseAlpha compatibility).
        Input features should be returns, realized volatility, volume ratio — NOT price levels.
        """
        X_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.atleast_2d(X_train)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        logger.info(f"Fitting SJM on {X_arr.shape[0]} observations, {X_arr.shape[1]} features...")
        self.model.fit(X_arr)
        
        self._is_fitted = True
        logger.info("SJM model fitted successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime labels for each observation.
        
        Returns:
            np.ndarray of integer regime labels (0, 1, ... n_components-1).
            In 2-regime mode: typically 0=bear, 1=bull (ordering depends on fit).
        """
        if not self._is_fitted:
            raise RuntimeError("SJM model has not been fitted yet. Call fit() first.")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities for each observation.
        
        Returns:
            np.ndarray of shape (n_samples, n_components) with regime probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("SJM model has not been fitted yet. Call fit() first.")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_arr)

    def predict_online(self, X: pd.DataFrame) -> np.ndarray:
        """Online regime prediction — for live use."""
        if not self._is_fitted:
            raise RuntimeError("SJM model has not been fitted yet. Call fit() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.model.predict(X_arr)

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {**self._merged_params}
