"""
Unit tests for openPHMM.algorithms.backward.

Validates the backward algorithm via its consistency with the
forward algorithm (alpha_t . beta_t gives the same total likelihood
for any t), against a hand-computed toy example, and edge cases.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from openPHMM.algorithms.forward import forward
from openPHMM.algorithms.backward import backward
from openPHMM.utils.covariance import multivariate_gaussian_logpdf
from openPHMM.utils.numerical import log_sum_exp


# ---------------------------------------------------------------------
# Test 1: forward-backward consistency (alpha_t . beta_t is constant in t)
# ---------------------------------------------------------------------

def test_backward_consistent_with_forward_on_fitted_model():
    rng = np.random.default_rng(0)

    n_states = 3
    n_features = 2
    T = 20

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=0,
    )
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.2, 0.6],
    ])
    model.means_ = rng.standard_normal((n_states, n_features)) * 3
    covars = []
    for _ in range(n_states):
        A = rng.standard_normal((n_features, n_features))
        covars.append(A @ A.T + 0.5 * np.eye(n_features))
    model.covars_ = np.array(covars)

    obs, _ = model.sample(T, random_state=1)

    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)

    log_B = np.zeros((n_states, T))
    for i in range(n_states):
        log_B[i, :] = multivariate_gaussian_logpdf(
            obs, model.means_[i], model.covars_[i]
        )

    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = backward(log_A, log_B)

    # sum_i alpha_t(i) * beta_t(i) should equal the total likelihood
    # for every t, not just t = T-1.
    reference_loglik = log_sum_exp(log_alpha[:, -1])

    for t in [0, T // 2, T - 1]:
        combined = log_alpha[:, t] + log_beta[:, t]
        loglik_at_t = log_sum_exp(combined)
        np.testing.assert_allclose(loglik_at_t, reference_loglik, atol=1e-8)


# ---------------------------------------------------------------------
# Test 2: hand-computed toy example (2 states, 3 timesteps)
# ---------------------------------------------------------------------

def test_backward_matches_hand_computation():
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
    ])
    B = np.array([
        [0.5, 0.4, 0.3],
        [0.1, 0.6, 0.7],
    ])

    log_A = np.log(A)
    log_B = np.log(B)

    log_beta = backward(log_A, log_B)

    # Hand computation in plain probability space, backward in time.
    beta = np.zeros((2, 3))
    beta[:, 2] = 1.0

    for t in range(1, -1, -1):
        for i in range(2):
            beta[i, t] = np.sum(A[i, :] * B[:, t + 1] * beta[:, t + 1])

    np.testing.assert_allclose(np.exp(log_beta), beta, atol=1e-10)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_backward_single_timestep():
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    log_B = np.log(np.array([[0.5], [0.2]]))

    log_beta = backward(log_A, log_B)

    assert log_beta.shape == (2, 1)
    np.testing.assert_allclose(log_beta[:, 0], np.array([0.0, 0.0]), atol=1e-10)


def test_backward_raises_on_nonsquare_A():
    log_A = np.log(np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]]))  # 2x3, not square
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3]]))

    with pytest.raises(ValueError):
        backward(log_A, log_B)


def test_backward_raises_on_shape_mismatch():
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))  # 2 states
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3], [0.1, 0.9]]))  # 3 states

    with pytest.raises(ValueError):
        backward(log_A, log_B)


def test_backward_single_state():
    log_A = np.log(np.array([[1.0]]))
    log_B = np.log(np.array([[0.5, 0.3, 0.7]]))

    log_beta = backward(log_A, log_B)

    assert log_beta.shape == (1, 3)
    # With a single state, beta_t is the running product of future
    # emission probabilities (no transition choice to make).
    expected = np.zeros(3)
    expected[1] = np.log(0.7)
    expected[0] = np.log(0.3) + np.log(0.7)
    np.testing.assert_allclose(log_beta[0, :], expected, atol=1e-10)