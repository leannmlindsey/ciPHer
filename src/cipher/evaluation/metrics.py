"""Standard metrics for ranking evaluation."""

import numpy as np


def hr_at_k(ranks, k):
    """Compute Hit Rate @ k.

    Args:
        ranks: list of integer ranks (1-based)
        k: cutoff

    Returns:
        float: fraction of ranks <= k
    """
    if not ranks:
        return 0.0
    return float(np.mean([1 if r <= k else 0 for r in ranks]))


def mrr(ranks):
    """Compute Mean Reciprocal Rank.

    Args:
        ranks: list of integer ranks (1-based)

    Returns:
        float: mean of 1/rank
    """
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / r for r in ranks]))


def hr_curve(ranks, max_k=20):
    """Compute HR@k for k=1..max_k.

    Args:
        ranks: list of integer ranks
        max_k: maximum k value

    Returns:
        dict: {k: hr_at_k} for k=1..max_k
    """
    return {k: hr_at_k(ranks, k) for k in range(1, max_k + 1)}
