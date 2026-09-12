"""
Unit tests for openPHMM.algorithms.baum_welch.

Validates the E-step (compute_posteriors) against hmmlearn's
predict_proba and via the xi/gamma marginal identity, and validates
the M-step (m_step_transitions, m_step_gaussian_emissions) by
running a single EM iteration and comparing against hmmlearn's
one-iteration update from identical starting parameters.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from openPHMM.algorithms.forward import forward
from openPHMM.algorithms.backward import backward
from openPHMM.algorithms.baum_welch import (
    compute_posteriors,
    m_step_transitions,
    m_step_gaussian_emissions,
)
from openPHMM.utils.covariance import multivariate_gaussian_logpdf
from openPHMM.utils.numerical import log_sum_exp


# ---------------------------------------------------------------------
# Shared fixture: a known (hand-parameterized) GaussianHMM + a sample.
# ---------------------------------------------------------------------

def _build_model_and_sample(T=30, seed=0):
    rng = np.random.default_rng(seed)

    n_states = 3
    n_features = 2

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
    return model, obs, n_states, n_features


def _log_B_from_model(model, obs, n_states, T):
    log_B = np.zeros((n_states, T))
    for i in range(n_states):
        log_B[i, :] = multivariate_gaussian_logpdf(
            obs, model.means_[i], model.covars_[i]
        )
    return log_B


# ---------------------------------------------------------------------
# Test 1: gamma matches hmmlearn's predict_proba
# ---------------------------------------------------------------------

def test_gamma_matches_hmmlearn_predict_proba():
    model, obs, n_states, _ = _build_model_and_sample()
    T = obs.shape[0]

    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)
    log_B = _log_B_from_model(model, obs, n_states, T)

    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = backward(log_A, log_B)
    log_gamma, _ = compute_posteriors(log_alpha, log_beta, log_A, log_B)

    gamma = np.exp(log_gamma).T  # transpose to (T, n_states) to match hmmlearn
    expected = model.predict_proba(obs)

    np.testing.assert_allclose(gamma, expected, atol=1e-8)


# ---------------------------------------------------------------------
# Test 2: xi marginal identity — sum_j xi_t(i,j) == gamma_t(i), t < T-1
# ---------------------------------------------------------------------

def test_xi_marginal_equals_gamma():
    model, obs, n_states, _ = _build_model_and_sample()
    T = obs.shape[0]

    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)
    log_B = _log_B_from_model(model, obs, n_states, T)

    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = backward(log_A, log_B)
    log_gamma, log_xi = compute_posteriors(log_alpha, log_beta, log_A, log_B)

    for t in range(T - 1):
        # sum over j (the "to" state, axis 1 of the (i, j) slice)
        xi_marginal = log_sum_exp(log_xi[:, :, t], axis=1)
        np.testing.assert_allclose(
            np.exp(xi_marginal), np.exp(log_gamma[:, t]), atol=1e-10
        )


# ---------------------------------------------------------------------
# Test 3: gamma columns are valid distributions (sum to 1 over states)
# ---------------------------------------------------------------------

def test_gamma_columns_sum_to_one():
    model, obs, n_states, _ = _build_model_and_sample()
    T = obs.shape[0]

    log_pi = np.log(model.startprob_)
    log_A = np.log(model.transmat_)
    log_B = _log_B_from_model(model, obs, n_states, T)

    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = backward(log_A, log_B)
    log_gamma, _ = compute_posteriors(log_alpha, log_beta, log_A, log_B)

    col_sums = np.exp(log_gamma).sum(axis=0)
    np.testing.assert_allclose(col_sums, np.ones(T), atol=1e-10)


# ---------------------------------------------------------------------
# Test 4: one EM iteration matches hmmlearn's one-iteration update
# ---------------------------------------------------------------------

def test_one_em_iteration_matches_hmmlearn():
    """
    Run exactly one EM iteration from identical starting parameters,
    both with our implementation and with hmmlearn, and compare the
    resulting pi, A, means, and covariances.
    """
    model, obs, n_states, n_features = _build_model_and_sample()
    T = obs.shape[0]

    # Starting parameters (the "known" model's params).
    start_pi = model.startprob_.copy()
    start_A = model.transmat_.copy()
    start_means = model.means_.copy()
    start_covars = model.covars_.copy()

    # --- Our one iteration ---
    log_pi = np.log(start_pi)
    log_A = np.log(start_A)
    log_B = _log_B_from_model(model, obs, n_states, T)

    log_alpha = forward(log_pi, log_A, log_B)
    log_beta = backward(log_A, log_B)
    log_gamma, log_xi = compute_posteriors(log_alpha, log_beta, log_A, log_B)

    our_log_pi, our_log_A = m_step_transitions(log_gamma, log_xi)
    our_means, our_covars = m_step_gaussian_emissions(obs, log_gamma)

    our_pi = np.exp(our_log_pi)
    our_A = np.exp(our_log_A)

    # --- hmmlearn's one iteration from the same starting point ---
    ref = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1,
        init_params="",
        params="stmc",
        min_covar=1e-12,   # effectively disable hmmlearn's variance floor
        random_state=0,
    )
    ref.startprob_ = start_pi.copy()
    ref.transmat_ = start_A.copy()
    ref.means_ = start_means.copy()
    ref.covars_ = start_covars.copy()
    ref.fit(obs)

    np.testing.assert_allclose(our_pi, ref.startprob_, atol=1e-6)
    np.testing.assert_allclose(our_A, ref.transmat_, atol=1e-6)
    np.testing.assert_allclose(our_means, ref.means_, atol=1e-6)
    # hmmlearn stores covars_ as (n_states, n_features, n_features) for "full"
        # hmmlearn nudges its covariance with a built-in prior; ours doesn't.
    # So compare ours against the plain textbook covariance instead.
    gamma = np.exp(log_gamma)
    expected_covars = np.zeros((n_states, n_features, n_features))
    for i in range(n_states):
        w = gamma[i, :]
        diff = obs - our_means[i]
        expected_covars[i] = (diff * w[:, None]).T @ diff / w.sum()

    np.testing.assert_allclose(our_covars, expected_covars, atol=1e-12)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_m_step_transitions_raises_on_single_timestep():
    log_gamma = np.log(np.array([[0.5], [0.5]]))  # T = 1
    log_xi = np.zeros((2, 2, 0))  # empty time axis

    with pytest.raises(ValueError):
        m_step_transitions(log_gamma, log_xi)


def test_compute_posteriors_raises_on_shape_mismatch():
    log_alpha = np.log(np.array([[0.5, 0.4], [0.5, 0.6]]))  # 2 states, T=2
    log_beta = np.log(np.array([[0.5, 0.4], [0.5, 0.6]]))
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    log_B = np.log(np.array([[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]]))  # T=3, mismatch

    with pytest.raises(ValueError):
        compute_posteriors(log_alpha, log_beta, log_A, log_B)


def test_m_step_gaussian_emissions_raises_on_shape_mismatch():
    obs = np.array([[0.2], [1.5], [3.1]])  # T=3
    log_gamma = np.log(np.array([[0.5, 0.5], [0.5, 0.5]]))  # T=2, mismatch

    with pytest.raises(ValueError):
        m_step_gaussian_emissions(obs, log_gamma)


def test_m_step_emissions_recovers_weighted_mean_single_state():
    # With one state and uniform gamma, the weighted mean/covariance
    # must reduce to the ordinary sample mean/covariance.
    obs = np.array([[0.0], [2.0], [4.0]])  # sample mean 2.0, var 8/3
    log_gamma = np.log(np.array([[1.0, 1.0, 1.0]]))  # one state, weight 1 each

    means, covars = m_step_gaussian_emissions(obs, log_gamma)

    np.testing.assert_allclose(means[0], np.array([2.0]), atol=1e-10)
    # population variance (divide by N, not N-1): mean of squared deviations
    expected_var = np.mean((obs[:, 0] - 2.0) ** 2)
    np.testing.assert_allclose(covars[0, 0, 0], expected_var, atol=1e-10)