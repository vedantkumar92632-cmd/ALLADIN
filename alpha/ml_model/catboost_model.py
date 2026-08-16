"""
CatBoost alpha model for forward return prediction.

Tuning parameters sourced from MASTER_REFERENCE.md Part 3A, Model 1: CatBoost.
Indian market-specific: NSE sector codes passed as cat_features (no one-hot encoding).

Resources:
- Awesome CatBoost: github.com/valeman/Awesome_CatBoost
- ML4T (Stefan Jansen): github.com/stefan-jansen/machine-learning-for-trading
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from typing import Optional, List, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.ml_model.catboost_model")

# Default tuning ranges from Part 3A (MASTER_REFERENCE.md)
DEFAULT_PARAMS = {
    'depth': 5,                    # Range: 4-6 (shallower than default for noisy financial signals)
    'learning_rate': 0.03,         # Range: 0.01-0.05 (lower than default, more iterations)
    'l2_leaf_reg': 5.0,            # Range: 3-10 (stronger regularization for noisy returns)
    'subsample': 0.7,              # Range: 0.6-0.8 (bagging to reduce variance)
    'one_hot_max_size': 3,         # Range: 2-5 (for sector/exchange categoricals)
    'iterations': 1000,
    'early_stopping_rounds': 50,
    'verbose': 0,
    'random_seed': 42,
    'thread_count': -1,
}


class CatBoostAlpha(BaseAlpha):
    """
    CatBoost wrapper for alpha signal generation.
    
    Supports both regression (forward return prediction) and classification
    (direction prediction) modes, per Part 3A specifications.
    
    NSE sector codes, exchange type, and F&O eligibility flags should be
    passed as cat_features — CatBoost handles them natively without encoding.
    """

    @property
    def name(self) -> str:
        return "strategy_1"

    @property
    def domain(self) -> str:
        return "alpha"

    def __init__(
        self,
        mode: str = "regression",
        cat_features: Optional[List[int]] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            mode: 'regression' for forward return prediction (RMSE loss),
                  'classification' for direction prediction (Logloss).
            cat_features: List of column indices that are categorical features
                         (e.g., NSE sector codes, exchange type).
            params: Override default hyperparameters. Merged with DEFAULT_PARAMS.
        """
        self.mode = mode
        self.cat_features = cat_features or []
        
        self._merged_params = {**DEFAULT_PARAMS}
        if params:
            self._merged_params.update(params)
            
        # Set loss function based on mode
        if self.mode == "regression":
            self._merged_params['loss_function'] = 'RMSE'
            self.model = CatBoostRegressor(**self._merged_params)
        elif self.mode == "classification":
            self._merged_params['loss_function'] = 'Logloss'
            self.model = CatBoostClassifier(**self._merged_params)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'regression' or 'classification'.")
            
        self._is_fitted = False
        logger.info(f"CatBoostAlpha initialized in {mode} mode with params: depth={self._merged_params['depth']}, lr={self._merged_params['learning_rate']}, l2_leaf_reg={self._merged_params['l2_leaf_reg']}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit the CatBoost model on training data.
        
        Uses early stopping if validation data is provided.
        IMPORTANT: Never use random splits for financial data — use TimeSeriesSplit
        or CPCV (Domain 8). This is the caller's responsibility.
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            
        logger.info(f"Fitting CatBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
        
        self.model.fit(
            X_train, y_train,
            cat_features=self.cat_features if self.cat_features else None,
            eval_set=eval_set,
            use_best_model=eval_set is not None
        )
        
        self._is_fitted = True
        logger.info("CatBoost model fitted successfully.")
        
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
        importances = self.model.get_feature_importance()
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
            return pd.Series(importances, index=self.model.feature_names_).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters."""
        return {**self._merged_params}
