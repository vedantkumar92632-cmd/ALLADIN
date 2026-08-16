"""
Abstract base class for all alpha models in ALLADIN V1.

Every alpha model (CatBoost, LightGBM, regime detectors, vol models, etc.)
must implement this interface so they can be swapped, ensembled, and orchestrated
uniformly by the signal consolidation layer.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


class BaseAlpha(ABC):
    """
    Abstract base for alpha signal generators.
    
    Subclasses must implement:
        - fit(): Train or calibrate on historical data
        - predict(): Generate forward-looking signals
        - name: A unique identifier for the alpha source
    
    Optional overrides:
        - get_feature_importance(): Feature attribution (ML models)
        - get_params(): Current hyperparameters
        - save() / load(): Persistence
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this alpha model (e.g. 'catboost_regression')."""
        ...

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain category (e.g. 'alpha', 'regime', 'volatility', 'mean_reversion', 'nlp')."""
        ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Train or calibrate the model.
        
        For ML models: supervised training with features and targets.
        For statistical models: parameter estimation from historical data.
        
        IMPORTANT: Callers must ensure time-ordered splits (TimeSeriesSplit or CPCV).
        Never use random splits for financial data.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions or signals.
        
        Returns:
            np.ndarray of signal values. Interpretation depends on model type:
            - Alpha models: forward return predictions or direction probabilities
            - Regime models: regime labels (0, 1, 2, ...)
            - Volatility models: conditional variance / vol forecasts
            - Mean reversion models: z-scores or entry/exit signals
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted/calibrated."""
        return getattr(self, '_is_fitted', False)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Return feature importances as a named Series, if applicable.
        Returns None for models without feature attribution (e.g. statistical models).
        """
        return None

    def get_params(self) -> Dict[str, Any]:
        """Return current hyperparameters as a dict."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', domain='{self.domain}', fitted={self.is_fitted})"
