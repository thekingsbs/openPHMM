"""
Backward algorithm for openPHMM.

Computes the log-space backward variable table (log beta) for an HMM,
used together with the forward algorithm to compute state-occupation
probabilities for the Baum-Welch E-step.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from openPHMM.utils.numerical import log_sum_exp

def backward(log_A: NDArray[np.float64], log_B: NDArray[np.float64]) -> NDArray[np.float64]:
    n_states = log_A.shape[0]
    T = log_B.shape[1]

    log_A = np.asarray(log_A, dtype=np.float64)
    log_B = np.asarray(log_B, dtype=np.float64)
    
    log_beta = np.zeros((n_states, T), dtype=np.float64)
        
    if log_A.shape[0] != log_A.shape[1]:
        raise ValueError("Log_A isn't square!")
    elif log_A.shape[0] != log_B.shape[0]:
        raise ValueError("Shape Mismatch!")

    log_beta[:, T-1] = 0

    for t in range(T-2, -1, -1):
        for i in range(n_states):
            log_beta[i, t] = log_sum_exp(log_A[i, :] + log_B[:, t+1] + log_beta[:, t+1])

    return log_beta