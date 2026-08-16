"""
Bayesian Online Changepoint Detection (BOCPD).

Detects the *moment* of a structural break in real-time.
Complementary to SJM (sjm.py) which classifies *which* regime you're in.
These two answer different questions and do NOT overlap — do NOT merge.

Algorithm: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"
Reference implementation: gwgundersen/bocd (Gregory Gundersen)

Tuning (Part 3A):
- cp_prior (hazard rate) = 1 / (expected avg days between structural breaks)
- Threshold at 0.5-0.7 probability; lower for early warnings, higher for confirmed
- Use empirical priors derived from data when available

Resources:
- gwgundersen/bocd: Clean Python implementation of Adams & MacKay 2007
- Gregory Gundersen's blog: Detailed mathematical walkthrough
- Facebook Kats: Production BOCPD with hyperparameter search (heavy dependency)
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any, Tuple
from core.logger import get_logger
from alpha.base_alpha import BaseAlpha

logger = get_logger("strategy.regime.bocpd")


class BOCPD(BaseAlpha):
    """
    Bayesian Online Changepoint Detection.
    
    Implements Adams & MacKay (2007) algorithm for detecting structural breaks
    in time series data. Outputs run-length probabilities at each timestep.
    
    This is a fast alarm — fires early, verify with SJM for regime classification.
    """

    @property
    def name(self) -> str:
        return "strategy_8"

    @property
    def domain(self) -> str:
        return "regime"

    def __init__(
        self,
        hazard_rate: float = 1/250,
        threshold: float = 0.5,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            hazard_rate: Prior probability of a changepoint at any timestep.
                        Default: 1/250 (expect ~1 changepoint per trading year).
                        Tune to: 1 / (expected avg days between structural breaks in NSE stocks).
            threshold: Probability threshold for declaring a changepoint.
                      Range: 0.5-0.7. Lower = more sensitive (early warnings).
            params: Additional parameters (mu_prior, kappa_prior, alpha_prior, beta_prior).
        """
        self.hazard_rate = hazard_rate
        self.threshold = threshold
        
        # Normal-Inverse-Gamma conjugate prior parameters
        # These are updated online via Bayesian updating
        self.mu_prior = 0.0          # Prior mean of data
        self.kappa_prior = 1.0       # Confidence in prior mean
        self.alpha_prior = 1.0       # Shape parameter
        self.beta_prior = 1.0        # Scale parameter
        
        if params:
            self.mu_prior = params.get('mu_prior', self.mu_prior)
            self.kappa_prior = params.get('kappa_prior', self.kappa_prior)
            self.alpha_prior = params.get('alpha_prior', self.alpha_prior)
            self.beta_prior = params.get('beta_prior', self.beta_prior)
        
        self._is_fitted = False
        self._run_length_probs = None
        self._changepoints = None
        
        logger.info(f"BOCPD initialized: hazard_rate={hazard_rate:.6f}, threshold={threshold}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Run BOCPD on the training data to calibrate and detect changepoints.
        
        BOCPD is online/unsupervised — y_train is ignored (BaseAlpha compatibility).
        X_train should be a single-column DataFrame or Series of returns/prices.
        """
        if isinstance(X_train, pd.DataFrame):
            if X_train.shape[1] == 1:
                data = X_train.iloc[:, 0].values
            else:
                # Use first column if multiple provided
                data = X_train.iloc[:, 0].values
                logger.warning("BOCPD received multi-column input; using first column only.")
        elif isinstance(X_train, pd.Series):
            data = X_train.values
        else:
            data = np.asarray(X_train).ravel()
        
        logger.info(f"Running BOCPD on {len(data)} observations...")
        
        self._run_length_probs, self._changepoints = self._run_bocpd(data)
        
        self._is_fitted = True
        n_cp = len(self._changepoints)
        logger.info(f"BOCPD detected {n_cp} changepoint(s) at threshold={self.threshold}.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run BOCPD on new data and return changepoint indicators.
        
        Returns:
            np.ndarray of 0/1 indicating detected changepoints at each timestep.
        """
        if isinstance(X, pd.DataFrame):
            data = X.iloc[:, 0].values if X.shape[1] >= 1 else X.values.ravel()
        elif isinstance(X, pd.Series):
            data = X.values
        else:
            data = np.asarray(X).ravel()
        
        _, changepoints = self._run_bocpd(data)
        
        result = np.zeros(len(data), dtype=int)
        for cp_idx in changepoints:
            if 0 <= cp_idx < len(data):
                result[cp_idx] = 1
        return result

    def _run_bocpd(self, data: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Core BOCPD algorithm (Adams & MacKay 2007).
        
        Returns:
            run_length_probs: (T, T+1) matrix of run-length probabilities
            changepoints: list of indices where changepoints were detected
        """
        T = len(data)
        
        # Run-length probability matrix
        # R[t, r] = P(run length = r at time t)
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0  # Initial run length is 0
        
        # Sufficient statistics for Normal-Inverse-Gamma conjugate prior
        # Tracked for each possible run length
        mu = np.full(T + 1, self.mu_prior)
        kappa = np.full(T + 1, self.kappa_prior)
        alpha = np.full(T + 1, self.alpha_prior)
        beta = np.full(T + 1, self.beta_prior)
        
        changepoints = []
        
        for t in range(T):
            x = data[t]
            
            # 1. Evaluate predictive probability for each run length
            # Student-t predictive distribution parameters
            df_t = 2 * alpha[:t + 1]
            loc = mu[:t + 1]
            scale = np.sqrt(beta[:t + 1] * (kappa[:t + 1] + 1) / (alpha[:t + 1] * kappa[:t + 1]))
            
            # Predictive probabilities
            pred_probs = np.exp(stats.t.logpdf(x, df=df_t, loc=loc, scale=scale))
            pred_probs = np.maximum(pred_probs, 1e-300)  # Prevent total probability collapse
            
            # 2. Growth probabilities (run continues)
            R[t + 1, 1:t + 2] = R[t, :t + 1] * pred_probs * (1 - self.hazard_rate)
            
            # 3. Changepoint probability (run resets to 0)
            R[t + 1, 0] = np.sum(R[t, :t + 1] * pred_probs * self.hazard_rate)
            
            # 4. Normalize
            evidence = R[t + 1, :t + 2].sum()
            if evidence > 0:
                R[t + 1, :t + 2] /= evidence
            else:
                # Recovery: reset to uniform over existing run lengths
                R[t + 1, 0] = 1.0  # Assume changepoint on collapse
            
            # 5. Update sufficient statistics (Bayesian updating)
            new_mu = (kappa[:t + 1] * mu[:t + 1] + x) / (kappa[:t + 1] + 1)
            new_kappa = kappa[:t + 1] + 1
            new_alpha = alpha[:t + 1] + 0.5
            new_beta = beta[:t + 1] + (kappa[:t + 1] * (x - mu[:t + 1]) ** 2) / (2 * (kappa[:t + 1] + 1))
            
            # Shift statistics forward (for run lengths 1 to t+1)
            mu[1:t + 2] = new_mu
            kappa[1:t + 2] = new_kappa
            alpha[1:t + 2] = new_alpha
            beta[1:t + 2] = new_beta
            
            # Reset run length 0 to prior
            mu[0] = self.mu_prior
            kappa[0] = self.kappa_prior
            alpha[0] = self.alpha_prior
            beta[0] = self.beta_prior
            
            # 6. Detect changepoints: high probability of run length 0
            if R[t + 1, 0] > self.threshold:
                changepoints.append(t)
        
        return R, changepoints

    def get_changepoints(self) -> list:
        """Return list of detected changepoint indices from the last fit."""
        if not self._is_fitted:
            raise RuntimeError("BOCPD has not been fitted yet. Call fit() first.")
        return self._changepoints

    def get_run_length_probs(self) -> np.ndarray:
        """Return the full run-length probability matrix from the last fit."""
        if not self._is_fitted:
            raise RuntimeError("BOCPD has not been fitted yet. Call fit() first.")
        return self._run_length_probs

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            'hazard_rate': self.hazard_rate,
            'threshold': self.threshold,
            'mu_prior': self.mu_prior,
            'kappa_prior': self.kappa_prior,
            'alpha_prior': self.alpha_prior,
            'beta_prior': self.beta_prior,
        }
