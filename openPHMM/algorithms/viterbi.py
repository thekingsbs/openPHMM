"""
Viterbi algorithm for openPHMM.

Computes the single most likely hidden state sequence given an HMM
and an observation sequence, via log-space dynamic programming with
traceback.
"""

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

def viterbi(log_pi: NDArray[np.float64], log_A: NDArray[np.float64], log_B: NDArray[np.float64],) -> Tuple[NDArray[np.intp], NDArray[np.float64]]:

    log_pi = np.asarray(log_pi, dtype=np.float64)
    log_A = np.asarray(log_A, dtype=np.float64)
    log_B = np.asarray(log_B, dtype=np.float64)

    n_states = log_A.shape[0]
    T = log_B.shape[1]

    log_delta = np.zeros((n_states, T), dtype=np.float64)
    psi = np.zeros((n_states, T), dtype=int)

    if log_A.shape[0] != log_A.shape[1]:
        raise ValueError("Log_A isn't square!")
    elif log_A.shape[0] != log_B.shape[0] or log_A.shape[0] != log_pi.shape[0] or log_B.shape[0] != log_pi.shape[0]:
        raise ValueError("Shape Mismatch!")

    log_delta[:, 0] = log_pi + log_B[:, 0]

    for t in range(1, T):
        for j in range(n_states):
            log_delta_t = log_delta[:, t-1] + log_A[:, j]
            log_delta[j, t] = np.max(log_delta_t) + log_B[j, t]
            psi[j, t] = np.argmax(log_delta_t)

    best_last_state = np.argmax(log_delta[:, T-1])
    best_log_prob = log_delta[best_last_state, T-1]

    best_path = np.zeros((T,), dtype=int)

    best_path[-1] = best_last_state

    for t in range(T-2, -1, -1):
        best_path[t] = psi[best_path[t+1], t+1]

    return (best_path, best_log_prob)