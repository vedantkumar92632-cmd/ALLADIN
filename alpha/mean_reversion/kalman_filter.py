"""
Kalman Filter for dynamic hedge ratio estimation in pairs trading.

Solves the proven failure mode of static OLS hedge ratios — they become stale
during regime shifts. The Kalman filter adapts the hedge ratio continuously.

Key calibration (Part 3A):
- Observation covariance / transition covariance ratio determines adaptation speed
- Set so adaptation speed matches OU half-life of the pair
- Pair with 20-day half-life -> Kalman adapts over ~20-day timescale

NSE pairs to screen (Part 3A): HDFC Bank/ICICI Bank, Reliance/ONGC,
Infosys/TCS, Tata Motors/Maruti. Johansen eigenvalue > 0.15 = candidate.
Half-life sweet spot: 5-30 trading days.

Resources:
- kalman-filter.com pairs trading derivation
- QuantStart: Dynamic Hedge Ratio with Kalman Filter
- QuantInsti: Kalman Filter Python hyperparameter tuning
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("alpha.mean_reversion.kalman_filter")


class KalmanFilterPairs(BaseAlpha):
    """
    Kalman Filter for dynamic hedge ratio estimation in pairs trading.
    
    Uses a linear regression state-space model:
        y_t = beta_t * x_t + alpha_t + epsilon_t  (observation)
        beta_t = beta_{t-1} + eta_t                (state transition)
    
    The hedge ratio (beta) adapts over time, tracking cointegration drift.
    """

    @property
    def name(self) -> str:
        return "strategy_6"

    @property
    def domain(self) -> str:
        return "mean_reversion"

    def __init__(
        self,
        delta: float = 1e-4,
        Ve: float = 1e-3,
        entry_z: float = 2.0,
        exit_z: float = 0.5
    ):
        """
        Args:
            delta: Transition covariance scaling. Controls how fast the hedge 
                   ratio adapts. Larger = faster adaptation.
                   Calibrate so adaptation speed matches OU half-life of the pair.
            Ve: Observation noise variance. Smaller = trust observations more.
            entry_z: Z-score threshold for entry signal.
            exit_z: Z-score threshold for exit signal.
        """
        self.delta = delta
        self.Ve = Ve
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        self._betas = None       # Dynamic hedge ratio history
        self._alphas = None      # Dynamic intercept history
        self._spreads = None     # Spread = y - beta*x - alpha
        self._z_scores = None    # Z-scores of the spread
        self._is_fitted = False
        
        logger.info(f"KalmanFilterPairs initialized: delta={delta}, Ve={Ve}, entry_z={entry_z}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Run Kalman filter on paired price series.
        
        X_train must have exactly 2 columns: [y_series, x_series].
        y_train is ignored (BaseAlpha compatibility).
        """
        if isinstance(X_train, pd.DataFrame) and X_train.shape[1] >= 2:
            y = X_train.iloc[:, 0].values
            x = X_train.iloc[:, 1].values
        else:
            raise ValueError("KalmanFilterPairs requires a 2-column DataFrame [y, x].")
        
        n = len(y)
        logger.info(f"Running Kalman filter on {n} observations...")
        
        # State: [alpha, beta] (intercept and hedge ratio)
        # State dimension = 2
        
        # Initial state estimate
        theta = np.zeros(2)  # [alpha, beta]
        
        # Initial state covariance (large = uncertain)
        P = np.eye(2) * 1.0
        
        # Transition covariance
        Wt = self.delta / (1 - self.delta) * np.eye(2)
        
        # Storage
        betas = np.zeros(n)
        alphas = np.zeros(n)
        spreads = np.zeros(n)
        e_t = np.zeros(n)  # prediction errors
        Q_t = np.zeros(n)  # prediction error variance
        
        for t in range(n):
            # Observation vector: [1, x_t]
            F = np.array([1.0, x[t]])
            
            # Prediction
            # theta_t|t-1 = theta_{t-1} (random walk state transition)
            # P_t|t-1 = P_{t-1} + Wt
            P = P + Wt
            
            # Prediction error
            y_hat = F @ theta
            e_t[t] = y[t] - y_hat
            
            # Prediction error variance
            Q_t[t] = F @ P @ F + self.Ve
            
            # Kalman gain
            K = P @ F / Q_t[t]
            
            # Update state
            theta = theta + K * e_t[t]
            
            # Update covariance
            P = P - np.outer(K, F) @ P
            
            # Store results
            alphas[t] = theta[0]
            betas[t] = theta[1]
            spreads[t] = e_t[t]
        
        self._betas = betas
        self._alphas = alphas
        self._spreads = spreads
        
        # Compute z-scores using rolling window of spread
        spread_series = pd.Series(spreads)
        rolling_mean = spread_series.rolling(window=20, min_periods=5).mean()
        rolling_std = spread_series.rolling(window=20, min_periods=5).std()
        rolling_std = rolling_std.replace(0, np.nan)
        self._z_scores = ((spread_series - rolling_mean) / rolling_std).fillna(0.0).values
        
        self._is_fitted = True
        logger.info(
            f"Kalman filter completed. Final beta={betas[-1]:.4f}, "
            f"alpha={alphas[-1]:.4f}, spread_std={np.nanstd(spreads):.4f}"
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return z-scores of the Kalman-filtered spread."""
        if not self._is_fitted:
            raise RuntimeError("Kalman filter not fitted yet. Call fit() first.")
        
        if X is not None:
            # Always run filter on the provided data
            # Store original state, run on new data, restore
            orig_betas = self._betas.copy()
            orig_alphas = self._alphas.copy() 
            orig_spreads = self._spreads.copy()
            orig_z = self._z_scores.copy()
            
            self.fit(X)
            new_z = self._z_scores.copy()
            
            # Restore original fitted state
            self._betas = orig_betas
            self._alphas = orig_alphas
            self._spreads = orig_spreads
            self._z_scores = orig_z
            
            return new_z
        
        return self._z_scores

    @property
    def betas(self) -> np.ndarray:
        """Dynamic hedge ratio history."""
        if not self._is_fitted:
            raise RuntimeError("Kalman filter not fitted yet.")
        return self._betas

    @property
    def spreads(self) -> np.ndarray:
        """Filtered spread history."""
        if not self._is_fitted:
            raise RuntimeError("Kalman filter not fitted yet.")
        return self._spreads

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        result = {
            'delta': self.delta,
            'Ve': self.Ve,
            'entry_z': self.entry_z,
            'exit_z': self.exit_z,
        }
        if self._is_fitted:
            result['final_beta'] = self._betas[-1]
            result['final_alpha'] = self._alphas[-1]
        return result
