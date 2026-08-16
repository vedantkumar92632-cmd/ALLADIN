"""
Ornstein-Uhlenbeck (OU) process for mean reversion signal generation.

Provides the half-life of a spread — the critical parameter that tells you
how long the spread takes to revert. If half-life < 1 day or > 100 trading days,
the pair is NOT tradeable in ALLADIN's framework.

MLE estimation of (mu, theta, sigma) gives better parameters than simple regression.

Resources (Part 3A Domain 4):
- Statistical arbitrage engine: github.com/topics/cointegration
- QuantStart: Dynamic Hedge Ratio with Kalman Filter
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict, Any, Tuple
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.mean_reversion.ou_process")


class OUProcess(BaseAlpha):
    """
    Ornstein-Uhlenbeck mean reversion model.
    
    Estimates OU parameters (mu, theta, sigma) via MLE, computes half-life,
    and generates z-score signals for pairs trading entry/exit.
    
    Half-life sweet spot for ALLADIN daily rebalancing: 5-30 trading days.
    """

    @property
    def name(self) -> str:
        return "strategy_5"

    @property
    def domain(self) -> str:
        return "mean_reversion"

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        min_half_life: float = 1.0,
        max_half_life: float = 100.0
    ):
        """
        Args:
            entry_z: Z-score threshold for entry (default 2.0 = 2 sigma).
            exit_z: Z-score threshold for exit (default 0.5).
            min_half_life: Minimum acceptable half-life in days.
            max_half_life: Maximum acceptable half-life in days.
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        self._mu = None        # Long-run mean
        self._theta = None     # Mean reversion speed
        self._sigma = None     # Volatility
        self._half_life = None
        self._is_fitted = False
        
        logger.info(f"OUProcess initialized: entry_z={entry_z}, exit_z={exit_z}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Fit OU parameters via MLE on the spread time series.
        
        X_train should be a single-column DataFrame or Series of spread values.
        y_train is ignored (unsupervised, BaseAlpha compatibility).
        """
        spread = self._extract_series(X_train)
        if len(spread) < 2:
            raise ValueError("Need at least 2 observations to fit OU process.")
        
        logger.info(f"Fitting OU process on {len(spread)} observations...")
        
        self._mu, self._theta, self._sigma = self._fit_mle(spread)
        
        # Half-life = ln(2) / theta
        if self._theta > 0:
            self._half_life = np.log(2) / self._theta
        else:
            self._half_life = np.inf
        
        self._is_fitted = True
        
        tradeable = self.min_half_life <= self._half_life <= self.max_half_life
        logger.info(
            f"OU fitted: mu={self._mu:.4f}, theta={self._theta:.4f}, "
            f"sigma={self._sigma:.4f}, half_life={self._half_life:.1f} days, "
            f"tradeable={'YES' if tradeable else 'NO'}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate z-score signals from spread values.
        
        Returns:
            np.ndarray of z-scores relative to the OU equilibrium.
            |z| > entry_z suggests entry; |z| < exit_z suggests exit.
        """
        if not self._is_fitted:
            raise RuntimeError("OU model not fitted yet. Call fit() first.")
        
        spread = self._extract_series(X)
        z_scores = (spread - self._mu) / self._sigma
        return z_scores.values

    def is_tradeable(self) -> bool:
        """Check if the fitted half-life is in the tradeable range."""
        if not self._is_fitted:
            raise RuntimeError("OU model not fitted yet. Call fit() first.")
        return self.min_half_life <= self._half_life <= self.max_half_life

    @property
    def half_life(self) -> float:
        """Half-life in trading days."""
        if not self._is_fitted:
            raise RuntimeError("OU model not fitted yet.")
        return self._half_life

    @property
    def mu(self) -> float:
        """Long-run equilibrium mean."""
        if not self._is_fitted:
            raise RuntimeError("OU model not fitted yet.")
        return self._mu

    @property
    def theta(self) -> float:
        """Mean reversion speed."""
        if not self._is_fitted:
            raise RuntimeError("OU model not fitted yet.")
        return self._theta

    def _fit_mle(self, spread: np.ndarray) -> Tuple[float, float, float]:
        """
        Maximum Likelihood Estimation of OU parameters.
        
        dX = theta * (mu - X) * dt + sigma * dW
        
        Returns: (mu, theta, sigma)
        """
        X = np.asarray(spread)
        n = len(X)
        dt = 1.0  # daily
        
        # Initial guesses from OLS regression: X_{t+1} - X_t = a + b*X_t + eps
        dX = np.diff(X)
        X_lag = X[:-1]
        
        # Simple linear regression for initial estimates
        b_hat = np.cov(dX, X_lag)[0, 1] / np.var(X_lag) if np.var(X_lag) > 1e-10 else -0.01
        a_hat = np.mean(dX) - b_hat * np.mean(X_lag)
        
        theta_init = max(-b_hat / dt, 0.001)
        mu_init = a_hat / (theta_init * dt) if theta_init > 0 else np.mean(X)
        sigma_init = np.std(dX) / np.sqrt(dt)
        
        def neg_log_likelihood(params):
            mu, theta, sigma = params
            if theta <= 0 or sigma <= 0:
                return 1e10
            
            # Expected value and variance of X_{t+1} given X_t under OU
            exp_neg_theta = np.exp(-theta * dt)
            mean_t = mu + (X_lag - mu) * exp_neg_theta
            var_t = (sigma**2 / (2 * theta)) * (1 - exp_neg_theta**2)
            
            if np.isnan(var_t) or var_t <= 0:
                return 1e10
            
            # Gaussian log-likelihood
            residuals = X[1:] - mean_t
            ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * n * np.log(var_t) - np.sum(residuals**2) / (2 * var_t)
            return -ll
        
        result = minimize(
            neg_log_likelihood,
            x0=[mu_init, theta_init, sigma_init],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        mu_mle, theta_mle, sigma_mle = result.x
        theta_mle = max(theta_mle, 1e-6)
        sigma_mle = max(sigma_mle, 1e-6)
        
        return mu_mle, theta_mle, sigma_mle

    def get_params(self) -> Dict[str, Any]:
        """Return fitted parameters."""
        result = {
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
            'min_half_life': self.min_half_life,
            'max_half_life': self.max_half_life,
        }
        if self._is_fitted:
            result.update({
                'mu': self._mu,
                'theta': self._theta,
                'sigma': self._sigma,
                'half_life': self._half_life,
                'tradeable': self.is_tradeable(),
            })
        return result

    @staticmethod
    def _extract_series(X) -> pd.Series:
        """Extract a 1D series from various input types."""
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            return X
        else:
            return pd.Series(np.asarray(X).ravel())
