"""
Covariance and multivariate Gaussian utility functions for openPHMM.

Provides numerically stable log-space evaluation of multivariate
and diagonal-covariance Gaussian densities, used by Gaussian
emission distributions in the forward, backward, Viterbi, and
Baum-Welch algorithms.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

def multivariate_gaussian_logpdf(x: NDArray[np.float64], mean: NDArray[np.float64], cov: NDArray[np.float64]) -> Union[np.float64, NDArray[np.float64]]:

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix isn't square!")
    elif mean.shape[0] != cov.shape[0]:
        raise ValueError("Covariance matrix and mean have different dimensions!")

    d = cov.shape[1]

    if x.shape[-1] != d:
        raise ValueError(f"{x.shape[-1]} != {d}")

    if x.ndim == 1:
        single_point = True
        x = x.reshape(1, d)
    elif x.ndim == 2:
        single_point = False
    else:
        raise ValueError("Shape of x is wrong!")

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Cov isn't positive definite!")

    log_det = 2 * sum(np.log(np.diag(L)))

    diff = x - mean

    diff_transpose = diff.T

    z = solve_triangular(L, diff_transpose, lower=True)
    z = np.transpose(z)

    mahalanobis = np.sum(z**2, axis=1)

    log_pdf = -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)

    if single_point:
        return log_pdf[0]

    return log_pdf

def diagonal_gaussian_logpdf(x: NDArray[np.float64], mean: NDArray[np.float64], var: NDArray[np.float64]) -> Union[np.float64, NDArray[np.float64]]:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    

    if mean.shape[0] != var.shape[0]:
        raise ValueError("Mean and variance have different dimensions!")

    d = var.shape[0]
    
    if x.shape[-1] != d:
        raise ValueError(f"{x.shape[-1]} != {d}")
    
    if x.ndim == 1:
        single_point = True
        x = x.reshape(1, d)
    elif x.ndim == 2:
        single_point = False
    else:
        raise ValueError("Shape of x is wrong!")

    if np.any(var <= 0):
        raise ValueError("Variance has non-positive terms!")

    diff = x - mean

    term = np.log(2 * np.pi) + np.log(var) + diff ** 2 / var

    result = -0.5 * np.sum(term, axis=1)

    if single_point:
        return result[0]

    return result