"""
Baum-Welch algorithm for openPHMM.

Implements the E-step (state and transition posterior computation)
and M-step (parameter re-estimation) for a standard Gaussian HMM.
This is the Stage 2 sanity-check layer, validated against hmmlearn.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from openPHMM.utils.numerical import log_sum_exp

def compute_posteriors(log_alpha: NDArray[np.float64], log_beta: NDArray[np.float64], log_A: NDArray[np.float64], log_B: NDArray[np.float64],) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    log_alpha = np.asarray(log_alpha, dtype=np.float64)
    log_beta = np.asarray(log_beta, dtype=np.float64)
    log_A = np.asarray(log_A, dtype=np.float64)
    log_B = np.asarray(log_B, dtype=np.float64)

    n_states = log_alpha.shape[0]
    T = log_alpha.shape[1]

    if log_A.shape != (n_states, n_states):
        raise ValueError("Incorrect shape for log_A!")
    elif log_B.shape != (n_states, T):
        raise ValueError("Incorrect shape for log_B!")
    elif log_alpha.shape != log_beta.shape:
        raise ValueError("Shape mismatch!")

    log_likelihood = log_sum_exp(log_alpha[:, T-1])

    log_gamma = log_alpha + log_beta - log_likelihood

    log_xi = np.zeros((n_states, n_states, T - 1))

    for t in range(0, T - 1):
        log_xi[:, :, t] = (log_alpha[:, t].reshape(-1, 1) + log_A + log_B[:, t + 1] + log_beta[:, t + 1] - log_likelihood)

    return (log_gamma, log_xi)


def m_step_transitions(log_gamma: NDArray[np.float64], log_xi: NDArray[np.float64],) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    log_gamma = np.asarray(log_gamma, dtype=np.float64)
    log_xi = np.asarray(log_xi, dtype=np.float64)

    n_states = log_gamma.shape[0]
    T = log_gamma.shape[1]

    if T < 2: 
        raise ValueError("Cannot produce estimations from single timestep sequence!")

    new_log_pi = log_gamma[:, 0]

    numerator = log_sum_exp(log_xi, axis=2)
    denominator = log_sum_exp(log_gamma[:, :T-1], axis=1)

    new_log_A = numerator - denominator.reshape(-1, 1)

    return (new_log_pi, new_log_A)

def m_step_gaussian_emissions(observations: NDArray[np.float64], log_gamma: NDArray[np.float64],) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    observations = np.asarray(observations, dtype=np.float64)
    log_gamma = np.asarray(log_gamma, dtype=np.float64)

    T = observations.shape[0]
    n_features = observations.shape[1]
    n_states = log_gamma.shape[0]

    if log_gamma.shape[1] != T:
        raise ValueError("Shape mismatch!")

    gamma = np.exp(log_gamma)

    new_means = np.zeros((n_states, n_features))
    new_covars = np.zeros((n_states, n_features, n_features))

    for i in range(n_states):
        weights = gamma[i, :]
        total_weight = np.sum(weights)
        new_means[i] = np.sum(weights[:, None] * observations, axis=0) / total_weight
        diff = observations - new_means[i]
        new_covars[i] = (diff * weights[:, None]).T @ diff / total_weight

    return (new_means, new_covars)