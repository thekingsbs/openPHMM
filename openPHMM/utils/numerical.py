"""
Numerical utility functions for openPHMM.

Provides numerically stable implementations of log-space
operations required by the forward, backward, and Viterbi
algorithms.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

# Constants
NEG_INF: float = -np.inf
LOG_ZERO: float = -1e300


def log_sum_exp(
    log_probs: NDArray[np.float64],
    axis: int | None = None
) -> NDArray[np.float64]:
    """
    Numerically stable log(sum(exp(log_probs))).

    Avoids overflow/underflow by factoring out the maximum value.

    Parameters
    ----------
    log_probs : NDArray[np.float64]
        Array of log probabilities
    axis : int or None
        Axis to sum over. None sums over all elements.

    Returns
    -------
    NDArray[np.float64]
        log(sum(exp(log_probs))) computed stably

    Examples
    --------
    >>> log_sum_exp(np.array([0.0, 0.0]))
    0.6931471805599453  # log(2)
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)

    if np.all(log_probs == NEG_INF):
        return np.float64(NEG_INF)

    max_val = np.max(log_probs, axis=axis, keepdims=True)
    result = max_val + np.log(
        np.sum(np.exp(log_probs - max_val), axis=axis, keepdims=True)
    )
    return np.squeeze(result)


def log_normalize(
    log_probs: NDArray[np.float64],
    axis: int = -1
) -> NDArray[np.float64]:
    """
    Normalize log probabilities so they sum to 1 in probability space.

    Parameters
    ----------
    log_probs : NDArray[np.float64]
        Array of log probabilities to normalize
    axis : int
        Axis along which to normalize. Default is -1 (last axis).

    Returns
    -------
    NDArray[np.float64]
        Normalized log probabilities

    Raises
    ------
    ValueError
        If log_probs is empty

    Examples
    --------
    >>> log_normalize(np.array([0.0, 0.0]))
    array([-0.693, -0.693])  # log(0.5) each
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)

    if log_probs.size == 0:
        raise ValueError(
            "log_probs must not be empty"
        )

    return log_probs - log_sum_exp(log_probs, axis=axis)


def safe_log(
    x: NDArray[np.float64],
    floor: float = 1e-300
) -> NDArray[np.float64]:
    """
    Compute log(x) with a floor to prevent log(0).

    Parameters
    ----------
    x : NDArray[np.float64]
        Input array
    floor : float
        Minimum value before taking log. Default is 1e-300.

    Returns
    -------
    NDArray[np.float64]
        log(max(x, floor))

    Raises
    ------
    ValueError
        If floor is not positive

    Examples
    --------
    >>> safe_log(np.array([0.0, 1.0]))
    array([-690.77, 0.0])
    """
    if floor <= 0:
        raise ValueError(
            f"floor must be positive, got {floor}"
        )

    x = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(x, floor, None))