"""Statistical variance and entropy calculations for predictive uncertainty."""

import numpy as np
from typing import Dict, Any


def compute_predictive_mean(probabilities: np.ndarray) -> np.ndarray:
    """Compute mean probability vector across stochastic Monte Carlo passes.

    Args:
        probabilities: Array of shape (T, C) where T = MC samples, C = classes.

    Returns:
        Array of shape (C,) representing mean probability for each class.
    """
    return np.mean(probabilities, axis=0)


def compute_predictive_variance(probabilities: np.ndarray) -> np.ndarray:
    """Compute variance vector across stochastic Monte Carlo passes.

    Args:
        probabilities: Array of shape (T, C).

    Returns:
        Array of shape (C,) representing variance for each class.
    """
    return np.var(probabilities, axis=0)


def compute_predictive_std(probabilities: np.ndarray) -> np.ndarray:
    """Compute standard deviation vector across stochastic Monte Carlo passes.

    Args:
        probabilities: Array of shape (T, C).

    Returns:
        Array of shape (C,) representing standard deviation for each class.
    """
    return np.std(probabilities, axis=0)


def compute_predictive_entropy(mean_probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy H(p) = - sum(p * log2(p + eps)) across classes.

    Args:
        mean_probabilities: Array of shape (C,) representing mean probability distribution.
        eps: Small epsilon to prevent log(0).

    Returns:
        Shannon entropy scalar value (bits).
    """
    p = np.clip(mean_probabilities, eps, 1.0)
    p = p / np.sum(p) # Ensure normalization
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)


def summarize_uncertainty(probabilities: np.ndarray) -> Dict[str, Any]:
    """Summarize uncertainty metrics from a collection of stochastic probability vectors.

    Args:
        probabilities: Array of shape (T, C).

    Returns:
        Dictionary containing mean_probability, variance, std, entropy, max_variance, mean_variance.
    """
    mean_p = compute_predictive_mean(probabilities)
    var = compute_predictive_variance(probabilities)
    std = compute_predictive_std(probabilities)
    entropy = compute_predictive_entropy(mean_p)

    return {
        "mean_probability": [round(float(x), 4) for x in mean_p],
        "variance": [round(float(x), 6) for x in var],
        "std": [round(float(x), 4) for x in std],
        "entropy": round(float(entropy), 4),
        "max_variance": round(float(np.max(var)), 6),
        "mean_variance": round(float(np.mean(var)), 6),
        "mc_samples": int(probabilities.shape[0])
    }
