"""
Unit tests for openPHMM.algorithms.forward.

Validates the forward algorithm against hmmlearn on a fitted
GaussianHMM, against a hand-computed toy example, and checks
basic probability-conservation sanity.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from openPHMM.algorithms.forward import forward
from openPHMM.utils.covariance import multivariate_gaussian_logpdf
from openPHMM.utils.numerical import log_sum_exp


# ---------------------------------------------------------------------
# Test 1: matches hmmlearn's log-likelihood on a fitted GaussianHMM
# ---------------------------------------------------------------------

def test_forward_matches_hmmlearn_score():
    rng = np.random.default_rng(0)

    n_states = 3
    n_features = 2
    T = 20

    # Generate synthetic data and fit an hmmlearn GaussianHMM to it.
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=0,
    )
    # hmmlearn needs initial params before sampling; use its own
    # random init by fitting on random data first, or just sample
    # from a manually-constructed model. Simpler: build the model's
    # parameters directly, then sample from it.
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    model.means_ = rng.standard_normal((n_states, n_features)) * 3
    # Build valid covariance matrices (positive-definite).
    covars = []
    for _ in range(n_states):
        A = rng.standard_normal((n_features, n_features))
        covars.append(A @ A.T + 0.5 * np.eye(n_features))
    model.covars_ = np.array(covars)

    obs, _ = model.sample(T, random_state=1)

    # Build log_pi, log_A directly from the model's parameters.
    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)

    # Build log_B using openPHMM's own multivariate Gaussian log-pdf,
    # evaluated at each state's fitted mean/covariance.
    log_B = np.zeros((n_states, T))
    for i in range(n_states):
        log_B[i, :] = multivariate_gaussian_logpdf(
            obs, model.means_[i], model.covars_[i]
        )

    log_alpha = forward(log_pi, log_A, log_B)
    total_loglik = log_sum_exp(log_alpha[:, -1])

    expected_loglik = model.score(obs)

    np.testing.assert_allclose(total_loglik, expected_loglik, atol=1e-8)


# ---------------------------------------------------------------------
# Test 2: hand-computed toy example (2 states, 3 timesteps)
# ---------------------------------------------------------------------

def test_forward_matches_hand_computation():
    # Simple 2-state HMM with easy-to-verify-by-hand numbers.
    pi = np.array([0.6, 0.4])
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
    ])
    # Emission probabilities P(O_t | state i), shape (n_states, T).
    B = np.array([
        [0.5, 0.4, 0.3],
        [0.1, 0.6, 0.7],
    ])

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    log_alpha = forward(log_pi, log_A, log_B)

    # Hand computation in plain probability space.
    alpha = np.zeros((2, 3))
    alpha[:, 0] = pi * B[:, 0]

    for t in range(1, 3):
        for j in range(2):
            alpha[j, t] = np.sum(alpha[:, t - 1] * A[:, j]) * B[j, t]

    np.testing.assert_allclose(np.exp(log_alpha), alpha, atol=1e-10)


# ---------------------------------------------------------------------
# Test 3: probability-conservation sanity check
# ---------------------------------------------------------------------

def test_forward_probability_conservation():
    # Small, well-behaved example unlikely to underflow, so we can
    # cross-check the logspace result against a direct (non-log)
    # computation of the same recurrence.
    pi = np.array([0.5, 0.5])
    A = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    B = np.array([
        [0.6, 0.6, 0.6],
        [0.4, 0.4, 0.4],
    ])

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    log_alpha = forward(log_pi, log_A, log_B)
    total_loglik = log_sum_exp(log_alpha[:, -1])
    total_lik = np.exp(total_loglik)

    # Valid probability: strictly between 0 and 1 for this example.
    assert 0.0 < total_lik < 1.0

    # Cross-check against a direct (non-log) forward pass.
    alpha = np.zeros((2, 3))
    alpha[:, 0] = pi * B[:, 0]
    for t in range(1, 3):
        for j in range(2):
            alpha[j, t] = np.sum(alpha[:, t - 1] * A[:, j]) * B[j, t]

    direct_lik = np.sum(alpha[:, -1])

    np.testing.assert_allclose(total_lik, direct_lik, atol=1e-10)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_forward_single_timestep():
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    log_B = np.log(np.array([[0.5], [0.2]]))

    log_alpha = forward(log_pi, log_A, log_B)

    expected = np.log(np.array([0.5, 0.5]) * np.array([0.5, 0.2]))
    np.testing.assert_allclose(log_alpha[:, 0], expected, atol=1e-10)
    assert log_alpha.shape == (2, 1)


def test_forward_raises_on_nonsquare_A():
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.log(np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]]))  # 2x3, not square
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3]]))

    with pytest.raises(ValueError):
        forward(log_pi, log_A, log_B)


def test_forward_raises_on_shape_mismatch():
    log_pi = np.log(np.array([0.5, 0.3, 0.2]))  # 3 states
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))  # 2 states
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3]]))  # 2 states

    with pytest.raises(ValueError):
        forward(log_pi, log_A, log_B)


def test_forward_single_state():
    log_pi = np.log(np.array([1.0]))
    log_A = np.log(np.array([[1.0]]))
    log_B = np.log(np.array([[0.5, 0.3, 0.7]]))

    log_alpha = forward(log_pi, log_A, log_B)

    assert log_alpha.shape == (1, 3)
    # With a single state, alpha_t should just be the running product
    # of emission probabilities (no real transition choice to make).
    expected = np.cumsum(np.log(np.array([0.5, 0.3, 0.7])))
    np.testing.assert_allclose(log_alpha[0, :], expected, atol=1e-10)