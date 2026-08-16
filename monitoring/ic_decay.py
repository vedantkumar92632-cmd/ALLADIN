"""
Rolling IC Decay Tracker (L9.S001).

Per Part 3B Domain 12: tracks whether each Layer 3 model's Information
Coefficient (IC) is decaying over time.

IC = rank correlation between model predictions and actual forward returns.
A decaying IC means the model's edge is eroding — possibly due to:
- Alpha decay (competitors found the same signal)
- Regime change (model trained on wrong regime)
- Data distribution shift
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import spearmanr
from core.logger import get_logger

logger = get_logger("monitoring.ic_decay")


class ICDecayTracker:
    """
    Tracks rolling Information Coefficient for each alpha model.
    
    Alerts when IC drops below threshold or shows consistent decline.
    """

    def __init__(
        self,
        window_size: int = 60,
        alert_threshold: float = 0.03,
        decay_lookback: int = 10,
    ):
        """
        Args:
            window_size: Rolling window for IC computation (trading days).
            alert_threshold: Alert if rolling IC drops below this value.
            decay_lookback: Number of IC observations to check for decay trend.
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.decay_lookback = decay_lookback
        
        # Store IC history per model
        self._ic_history: Dict[str, List[float]] = {}
        self._alerts: List[dict] = []
        
        logger.info(
            f"ICDecayTracker: window={window_size}, threshold={alert_threshold}, "
            f"decay_lookback={decay_lookback}"
        )

    def update(
        self,
        model_name: str,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> float:
        """
        Compute and record IC for a model.
        
        Args:
            model_name: Name of the alpha model.
            predictions: Model's predictions (ranked or continuous).
            actual_returns: Actual forward returns corresponding to predictions.
            
        Returns:
            Current IC value.
        """
        if len(predictions) < 5 or len(actual_returns) < 5:
            return 0.0
        
        # Spearman rank correlation
        ic, p_value = spearmanr(predictions, actual_returns)
        
        if np.isnan(ic):
            ic = 0.0
        
        if model_name not in self._ic_history:
            self._ic_history[model_name] = []
        
        self._ic_history[model_name].append(ic)
        
        # Check for decay
        self._check_decay(model_name, ic)
        
        return ic

    def _check_decay(self, model_name: str, current_ic: float):
        """Check if IC is decaying and trigger alerts."""
        history = self._ic_history[model_name]
        
        # Alert if IC below threshold
        if current_ic < self.alert_threshold and len(history) >= self.window_size:
            alert = {
                'type': 'ic_below_threshold',
                'model': model_name,
                'current_ic': round(current_ic, 4),
                'threshold': self.alert_threshold,
            }
            self._alerts.append(alert)
            logger.warning(
                f"IC DECAY ALERT: {model_name} IC={current_ic:.4f} "
                f"below threshold {self.alert_threshold}"
            )
        
        # Check for consistent decline over decay_lookback
        if len(history) >= self.decay_lookback:
            recent = history[-self.decay_lookback:]
            # Simple linear trend
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            
            if slope < -0.005:  # Consistently declining
                alert = {
                    'type': 'ic_declining_trend',
                    'model': model_name,
                    'slope': round(slope, 6),
                    'recent_ics': [round(ic, 4) for ic in recent],
                }
                self._alerts.append(alert)
                logger.warning(
                    f"IC DECLINE TREND: {model_name} slope={slope:.6f} "
                    f"over last {self.decay_lookback} observations"
                )

    def get_ic_summary(self) -> Dict[str, dict]:
        """Get IC summary for all models."""
        summary = {}
        for model, history in self._ic_history.items():
            if not history:
                continue
            summary[model] = {
                'current_ic': round(history[-1], 4),
                'mean_ic': round(np.mean(history), 4),
                'std_ic': round(np.std(history), 4),
                'min_ic': round(min(history), 4),
                'max_ic': round(max(history), 4),
                'n_observations': len(history),
            }
        return summary

    def get_alerts(self) -> List[dict]:
        return self._alerts
