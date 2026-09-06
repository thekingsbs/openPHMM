"""
Unit tests for openPHMM.utils.covariance.

Validates multivariate_gaussian_logpdf and diagonal_gaussian_logpdf
against scipy.stats reference implementations, checks batch/loop
consistency, and checks edge-case error handling.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from openPHMM.utils.covariance import (
    multivariate_gaussian_logpdf,
    diagonal_gaussian_logpdf,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def random_pd_cov(d: int, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """Generate a random positive-definite covariance matrix of size d x d."""
    A = rng.standard_normal((d, d))
    return A @ A.T + eps * np.eye(d)


# ---------------------------------------------------------------------
# Test 1: multivariate_gaussian_logpdf matches scipy
# ---------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 5])
def test_multivariate_matches_scipy_single_point(d):
    rng = np.random.default_rng(0)
    mean = rng.standard_normal(d)
    cov = random_pd_cov(d, rng)
    x = rng.standard_normal(d)

    got = multivariate_gaussian_logpdf(x, mean, cov)
    expected = multivariate_normal.logpdf(x, mean=mean, cov=cov)

    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("d", [1, 2, 5])
def test_multivariate_matches_scipy_batch(d):
    rng = np.random.default_rng(1)
    mean = rng.standard_normal(d)
    cov = random_pd_cov(d, rng)
    x = rng.standard_normal((10, d))

    got = multivariate_gaussian_logpdf(x, mean, cov)
    expected = multivariate_normal.logpdf(x, mean=mean, cov=cov)

    np.testing.assert_allclose(got, expected, atol=1e-10)


# ---------------------------------------------------------------------
# Test 2: batch vs. loop consistency
# ---------------------------------------------------------------------

def test_multivariate_batch_matches_loop():
    rng = np.random.default_rng(2)
    d = 3
    mean = rng.standard_normal(d)
    cov = random_pd_cov(d, rng)
    x = rng.standard_normal((7, d))

    batch_result = multivariate_gaussian_logpdf(x, mean, cov)
    loop_result = np.array(
        [multivariate_gaussian_logpdf(row, mean, cov) for row in x]
    )

    np.testing.assert_allclose(batch_result, loop_result, atol=1e-10)


# ---------------------------------------------------------------------
# Test 3: edge cases / error handling
# ---------------------------------------------------------------------

def test_multivariate_raises_on_nonsquare_cov():
    mean = np.zeros(2)
    cov = np.ones((2, 3))  # not square
    x = np.zeros(2)

    with pytest.raises(ValueError):
        multivariate_gaussian_logpdf(x, mean, cov)


def test_multivariate_raises_on_mean_cov_mismatch():
    mean = np.zeros(3)
    cov = np.eye(2)  # mismatched with mean
    x = np.zeros(2)

    with pytest.raises(ValueError):
        multivariate_gaussian_logpdf(x, mean, cov)


def test_multivariate_raises_on_x_cov_mismatch():
    mean = np.zeros(2)
    cov = np.eye(2)
    x = np.zeros(3)  # mismatched with cov/mean

    with pytest.raises(ValueError):
        multivariate_gaussian_logpdf(x, mean, cov)


def test_multivariate_raises_on_non_pd_cov():
    mean = np.zeros(2)
    # symmetric but not positive-definite: eigenvalues are 3 and -1
    cov = np.array([[1.0, 2.0], [2.0, 1.0]])
    x = np.zeros(2)

    with pytest.raises(ValueError):
        multivariate_gaussian_logpdf(x, mean, cov)


# ---------------------------------------------------------------------
# Test 4: 1-D degenerate case matches scipy.stats.norm
# ---------------------------------------------------------------------

def test_multivariate_1d_matches_norm():
    rng = np.random.default_rng(3)
    mean = rng.standard_normal(1)
    var = np.array([[rng.uniform(0.5, 2.0)]])  # 1x1 "covariance"
    x = rng.standard_normal(1)

    got = multivariate_gaussian_logpdf(x, mean, var)
    expected = norm.logpdf(x[0], loc=mean[0], scale=np.sqrt(var[0, 0]))

    np.testing.assert_allclose(got, expected, atol=1e-10)


# ---------------------------------------------------------------------
# Diagonal function tests (require diagonal_gaussian_logpdf to be implemented)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 5])
def test_diagonal_matches_multivariate_with_diag_cov(d):
    rng = np.random.default_rng(4)
    mean = rng.standard_normal(d)
    var = rng.uniform(0.5, 2.0, size=d)
    x = rng.standard_normal((10, d))

    diag_result = diagonal_gaussian_logpdf(x, mean, var)
    multi_result = multivariate_gaussian_logpdf(x, mean, np.diag(var))

    np.testing.assert_allclose(diag_result, multi_result, atol=1e-10)


def test_diagonal_matches_scipy_norm_1d():
    rng = np.random.default_rng(5)
    mean = rng.standard_normal(1)
    var = np.array([rng.uniform(0.5, 2.0)])
    x = rng.standard_normal(1)

    got = diagonal_gaussian_logpdf(x, mean, var)
    expected = norm.logpdf(x[0], loc=mean[0], scale=np.sqrt(var[0]))

    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_diagonal_raises_on_nonpositive_var():
    mean = np.zeros(2)
    var = np.array([1.0, -0.5])  # invalid: negative variance
    x = np.zeros(2)

    with pytest.raises(ValueError):
        diagonal_gaussian_logpdf(x, mean, var)


def test_diagonal_raises_on_shape_mismatch():
    mean = np.zeros(3)
    var = np.ones(2)  # mismatched with mean
    x = np.zeros(2)

    with pytest.raises(ValueError):
        diagonal_gaussian_logpdf(x, mean, var)


def test_diagonal_batch_matches_loop():
    rng = np.random.default_rng(6)
    d = 4
    mean = rng.standard_normal(d)
    var = rng.uniform(0.5, 2.0, size=d)
    x = rng.standard_normal((6, d))

    batch_result = diagonal_gaussian_logpdf(x, mean, var)
    loop_result = np.array(
        [diagonal_gaussian_logpdf(row, mean, var) for row in x]
    )

    np.testing.assert_allclose(batch_result, loop_result, atol=1e-10)