"""
Forward algorithm for openPHMM.

Computes the log-space forward variable table (log alpha) for an HMM,
used to evaluate sequence likelihood and as an input to the
Baum-Welch E-step.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray

from openPHMM.utils.numerical import log_sum_exp

def forward(log_pi: NDArray[np.float64], log_A: NDArray[np.float64], log_B: NDArray[np.float64],) -> NDArray[np.float64]:
    n_states = log_A.shape[0]
    T = log_B.shape[1]

    log_pi = np.asarray(log_pi, dtype=np.float64)
    log_A = np.asarray(log_A, dtype=np.float64)
    log_B = np.asarray(log_B, dtype=np.float64)

    log_alpha = np.zeros((n_states, T), dtype=np.float64)
    

    if log_A.shape[0] != log_A.shape[1]:
        raise ValueError("Log_A isn't square!")
    elif log_A.shape[0] != log_B.shape[0] or log_A.shape[0] != log_pi.shape[0] or log_B.shape[0] != log_pi.shape[0]:
        raise ValueError("Shape Mismatch!")


    log_alpha[:, 0] = log_pi + log_B[:, 0]

    for t in range(1, T):
        for j in range(n_states):
            log_alpha[j, t] = log_sum_exp(log_alpha[:, t-1] + log_A[:, j]) + log_B[j, t]

    return log_alpha
            