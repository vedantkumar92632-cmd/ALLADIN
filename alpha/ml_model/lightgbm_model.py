"""
LightGBM alpha model for forward return prediction.

Tuning parameters sourced from MASTER_REFERENCE.md Part 3A, Model 2: LightGBM.
LightGBM's leaf-wise growth makes it aggressive — overfits faster than CatBoost
on small NSE universes. Use as a fast retraining workhorse on rolling windows.

Resources:
- ML4T (Stefan Jansen): github.com/stefan-jansen/machine-learning-for-trading
- Qlib (Microsoft Research): github.com/microsoft/qlib
- LGBM Stock Prediction: github.com/singyanchiu/LGBM-Stock-Prediction
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Optional, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.ml_model.lightgbm_model")

# Default tuning ranges from Part 3A (MASTER_REFERENCE.md)
DEFAULT_PARAMS = {
    'num_leaves': 31,              # Range: 15-31 (lower than default 127 for financial data)
    'min_data_in_leaf': 30,        # Range: 20-50 (force each leaf to cover enough observations)
    'feature_fraction': 0.7,       # Range: 0.6-0.8 (column subsampling)
    'bagging_fraction': 0.8,       # Range: 0.8 (row subsampling)
    'bagging_freq': 5,
    'lambda_l1': 0.5,              # Range: 0.1-1.0 (L1 regularization)
    'lambda_l2': 0.5,              # Range: 0.1-1.0 (L2 regularization)
    'learning_rate': 0.03,         # Range: 0.01-0.05 with early stopping
    'n_estimators': 1000,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}


class LightGBMAlpha(BaseAlpha):
    """
    LightGBM wrapper for alpha signal generation.
    
    Supports both regression (forward return prediction) and classification
    (direction prediction) modes.
    
    Use early_stopping_rounds=50 with a validation window that respects time order.
    Never use random splits for financial data — use TimeSeriesSplit or CPCV (Domain 8).
    """

    @property
    def name(self) -> str:
        return "strategy_2"

    @property
    def domain(self) -> str:
        return "alpha"

    def __init__(
        self,
        mode: str = "regression",
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            mode: 'regression' for forward return prediction,
                  'classification' for direction prediction.
            params: Override default hyperparameters. Merged with DEFAULT_PARAMS.
        """
        self.mode = mode
        
        self._merged_params = {**DEFAULT_PARAMS}
        if params:
            self._merged_params.update(params)

        if self.mode == "regression":
            self._merged_params['objective'] = 'regression'
            self._merged_params['metric'] = 'rmse'
            self.model = lgb.LGBMRegressor(**self._merged_params)
        elif self.mode == "classification":
            self._merged_params['objective'] = 'binary'
            self._merged_params['metric'] = 'binary_logloss'
            self.model = lgb.LGBMClassifier(**self._merged_params)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'regression' or 'classification'.")
            
        self._is_fitted = False
        logger.info(f"LightGBMAlpha initialized in {mode} mode with params: num_leaves={self._merged_params['num_leaves']}, lr={self._merged_params['learning_rate']}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit the LightGBM model on training data.
        
        Uses early stopping if validation data is provided.
        IMPORTANT: Never use random splits for financial data — use TimeSeriesSplit
        or CPCV (Domain 8). This is the caller's responsibility.
        """
        fit_kwargs = {}
        if X_val is not None and y_val is not None:
            fit_kwargs['eval_set'] = [(X_val, y_val)]
            fit_kwargs['callbacks'] = [lgb.early_stopping(50, verbose=False)]
            
        logger.info(f"Fitting LightGBM on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        
        self.model.fit(X_train, y_train, **fit_kwargs)
        
        self._is_fitted = True
        logger.info("LightGBM model fitted successfully.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions (forward returns or direction probabilities)."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if self.mode == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
        
    def get_feature_importance(self) -> pd.Series:
        """Return feature importances as a named Series."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        importances = self.model.feature_importances_
        if hasattr(self.model, 'feature_name_') and self.model.feature_name_:
            return pd.Series(importances, index=self.model.feature_name_).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {**self._merged_params}
