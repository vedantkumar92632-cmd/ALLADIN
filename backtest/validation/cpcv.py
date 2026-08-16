"""
CPCV — Combinatorial Purged Cross-Validation (L7.S004).

Prevents data leakage in time-series splits. This is the Marcos Lopez de Prado
standard and is non-negotiable for financial ML validation.

Per Part 3B Domain 8: Standard k-fold CV causes data leakage in time-series
because training data may contain future information relative to test data.
CPCV creates all possible train/test combinations with purging and embargo gaps.

Resources:
- Lopez de Prado: Advances in Financial Machine Learning (Chapter 12)
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from core.logger import get_logger

logger = get_logger("backtest.validation.cpcv")


def purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    purge_gap: int = 5,
    embargo_pct: float = 0.01,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate purged k-fold cross-validation splits for time-series data.
    
    Args:
        n_samples: Total number of samples.
        n_splits: Number of folds.
        purge_gap: Number of samples to purge between train and test sets.
        embargo_pct: Fraction of test set size to embargo after test period.
        
    Returns:
        List of (train_indices, test_indices) tuples.
    """
    indices = np.arange(n_samples)
    fold_size = n_samples // n_splits
    embargo_size = max(1, int(fold_size * embargo_pct))
    
    splits = []
    
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n_samples)
        
        test_idx = indices[test_start:test_end]
        
        # Purge: remove `purge_gap` samples before and after test set
        purge_start = max(0, test_start - purge_gap)
        purge_end = min(n_samples, test_end + purge_gap + embargo_size)
        
        # Train = everything outside purged zone
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[purge_start:purge_end] = False
        train_idx = indices[train_mask]
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    
    logger.info(
        f"CPCV: {len(splits)} splits, {n_samples} samples, "
        f"purge_gap={purge_gap}, embargo={embargo_size}"
    )
    return splits


def combinatorial_purged_cv(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge_gap: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Full Combinatorial Purged Cross-Validation (CPCV).
    
    Generates all C(n_groups, n_test_groups) possible train/test combinations
    with purging between them.
    
    Args:
        n_samples: Total number of observations.
        n_groups: Number of groups to divide data into.
        n_test_groups: Number of groups to use as test in each split.
        purge_gap: Gap between train and test to prevent leakage.
        
    Returns:
        List of (train_indices, test_indices) tuples.
    """
    from itertools import combinations
    
    group_size = n_samples // n_groups
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = min((i + 1) * group_size, n_samples) if i < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))
    
    splits = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[i] for i in test_combo])
        
        train_groups = [i for i in range(n_groups) if i not in test_combo]
        
        # Purge: remove samples near test boundaries
        train_indices = []
        for tg in train_groups:
            group_idx = groups[tg]
            # Check if this group is adjacent to any test group
            is_adjacent = any(abs(tg - tc) == 1 for tc in test_combo)
            if is_adjacent and purge_gap > 0:
                # Purge right boundary if test group is to the right
                if (tg + 1) in test_combo:
                    group_idx = group_idx[:-purge_gap] if len(group_idx) > purge_gap else np.array([], dtype=int)
                # Purge left boundary if test group is to the left
                if (tg - 1) in test_combo:
                    group_idx = group_idx[purge_gap:] if len(group_idx) > purge_gap else np.array([], dtype=int)
            if len(group_idx) > 0:
                train_indices.append(group_idx)
        
        if train_indices:
            train_idx = np.concatenate(train_indices)
            splits.append((train_idx, test_idx))
    
    logger.info(
        f"CPCV: {len(splits)} combinatorial splits "
        f"({n_groups} groups, {n_test_groups} test groups, purge={purge_gap})"
    )
    return splits
