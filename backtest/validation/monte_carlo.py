"""
Monte Carlo Permutation Testing (L7.S006).

Tests whether strategy returns are statistically significant by shuffling
the return series and comparing the original performance to the distribution
of shuffled performances.
"""
import numpy as np
from typing import Callable, Optional
from core.logger import get_logger

logger = get_logger("backtest.validation.monte_carlo")


def monte_carlo_permutation_test(
    returns: np.ndarray,
    n_simulations: int = 1000,
    metric_fn: Optional[Callable] = None,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo permutation test for strategy significance.
    
    Shuffles the order of returns n_simulations times and computes the
    performance metric for each shuffle. The p-value is the fraction
    of shuffled results that are at least as good as the original.
    
    Args:
        returns: Original strategy returns (1D array).
        n_simulations: Number of random permutations.
        metric_fn: Function that takes returns array and returns a scalar metric.
                  Defaults to Sharpe ratio.
        seed: Random seed for reproducibility.
        
    Returns:
        Dict with keys: original_metric, p_value, simulated_metrics,
        percentile_rank.
    """
    rng = np.random.RandomState(seed)
    
    if metric_fn is None:
        # Default: annualized Sharpe ratio
        def metric_fn(r):
            if np.std(r) == 0:
                return 0.0
            return float(np.mean(r) / np.std(r) * np.sqrt(252))
    
    original_metric = metric_fn(returns)
    
    simulated_metrics = []
    for _ in range(n_simulations):
        shuffled = rng.permutation(returns)
        simulated_metrics.append(metric_fn(shuffled))
    
    simulated_metrics = np.array(simulated_metrics)
    
    # p-value: fraction of simulations >= original
    p_value = float(np.mean(simulated_metrics >= original_metric))
    
    # Percentile rank of original in the simulation distribution
    percentile_rank = float(np.mean(simulated_metrics < original_metric) * 100)
    
    logger.info(
        f"Monte Carlo: original={original_metric:.3f}, "
        f"p-value={p_value:.4f}, percentile={percentile_rank:.1f}%"
    )
    
    return {
        'original_metric': original_metric,
        'p_value': p_value,
        'percentile_rank': percentile_rank,
        'simulated_mean': float(simulated_metrics.mean()),
        'simulated_std': float(simulated_metrics.std()),
        'n_simulations': n_simulations,
    }


def monte_carlo_drawdown_test(
    returns: np.ndarray,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo test specifically for maximum drawdown.
    
    Tests whether the observed max drawdown is unusually large.
    """
    rng = np.random.RandomState(seed)
    if len(returns) == 0: return {'original_drawdown': 0.0, 'p_value': 1.0, 'simulated_mean_dd': 0.0, 'simulated_std_dd': 0.0}
    
    def max_drawdown(r):
        cumulative = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    original_dd = max_drawdown(returns)
    
    simulated_dds = []
    for _ in range(n_simulations):
        shuffled = rng.permutation(returns)
        simulated_dds.append(max_drawdown(shuffled))
    
    simulated_dds = np.array(simulated_dds)
    
    # p-value: fraction of simulations with drawdown >= original
    p_value = float(np.mean(simulated_dds >= original_dd))
    
    logger.info(
        f"MC Drawdown: original_dd={original_dd:.4f}, "
        f"p-value={p_value:.4f}"
    )
    
    return {
        'original_drawdown': original_dd,
        'p_value': p_value,
        'simulated_mean_dd': float(simulated_dds.mean()),
        'simulated_std_dd': float(simulated_dds.std()),
    }
