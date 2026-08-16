"""
Deflated Sharpe Ratio + Probabilistic Sharpe Ratio (L7.S005).

Per Part 6 D3: if you tested many variations and picked the best, raw Sharpe
is biased — DSR corrects for this. Never report a raw Sharpe from a multi-variant
search without also reporting DSR.

PSR tests whether the Sharpe is statistically different from zero.

Resources:
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
- Lopez de Prado: Advances in Financial Machine Learning (Chapter 14)
"""
import numpy as np
from scipy import stats
from core.logger import get_logger

logger = get_logger("backtest.validation.dsr_psr")


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.
    """
    excess = returns - risk_free_rate / periods_per_year
    if np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR).
    
    Tests whether the observed Sharpe is statistically greater than benchmark_sr.
    
    Returns:
        PSR as a probability (0 to 1). PSR > 0.95 means the Sharpe is
        significantly different from the benchmark at the 5% level.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    
    sr = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    
    # Skewness and kurtosis of returns
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))
    
    # Standard error of Sharpe ratio (Lo, 2002)
    sr_std = np.sqrt(
        max(1e-8, (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / (n - 1))
    )
    
    if sr_std == 0:
        return 0.0
    
    # PSR = Prob(SR > benchmark_sr)
    z = (sr - benchmark_sr) / sr_std
    psr = float(stats.norm.cdf(z))
    
    logger.info(f"PSR: SR={sr:.3f}, PSR={psr:.4f} (benchmark={benchmark_sr:.2f})")
    return psr


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (DSR).
    
    Adjusts the Sharpe ratio for multiple testing bias. If you tested
    n_trials strategy variations and picked the best, DSR tells you the
    probability that the best Sharpe is genuinely above zero.
    
    Args:
        returns: Returns of the BEST strategy from the search.
        n_trials: Number of strategy variations tested.
        
    Returns:
        DSR as a probability (0 to 1).
    """
    if n_trials < 1:
        n_trials = 1
    
    if n_trials <= 1:
        # With only 1 trial, no multiple-testing correction needed
        return probabilistic_sharpe_ratio(returns, benchmark_sr=0.0,
                                          risk_free_rate=risk_free_rate,
                                          periods_per_year=periods_per_year)

    n = len(returns)
    if n < 3:
        return 0.0
    
    # Expected maximum Sharpe under null (all strategies have SR=0)
    # Euler-Mascheroni approximation
    gamma = 0.5772156649
    expected_max_sr = np.sqrt(2 * np.log(n_trials)) - \
        (np.log(np.pi) + gamma) / (2 * np.sqrt(2 * np.log(n_trials)))
    
    # DSR = PSR with the expected max as benchmark
    dsr = probabilistic_sharpe_ratio(
        returns, 
        benchmark_sr=expected_max_sr,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    
    sr = sharpe_ratio(returns, risk_free_rate, periods_per_year)
    logger.info(
        f"DSR: SR={sr:.3f}, n_trials={n_trials}, "
        f"expected_max_SR={expected_max_sr:.3f}, DSR={dsr:.4f}"
    )
    return dsr
