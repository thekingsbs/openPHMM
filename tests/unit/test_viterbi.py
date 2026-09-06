"""
Unit tests for openPHMM.algorithms.viterbi.

Validates the Viterbi algorithm against hmmlearn's decode(), against
a hand-computed toy example with an unambiguous best path, and edge
cases including masked (-inf) transitions relevant to future profile
HMM topology.
"""

import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from openPHMM.algorithms.viterbi import viterbi
from openPHMM.utils.covariance import multivariate_gaussian_logpdf


# ---------------------------------------------------------------------
# Test 1: matches hmmlearn's decode() on a fitted GaussianHMM
# ---------------------------------------------------------------------

def test_viterbi_matches_hmmlearn_decode():
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

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    expected_log_prob, expected_path = model.decode(obs, algorithm="viterbi")

    np.testing.assert_array_equal(best_path, expected_path)
    np.testing.assert_allclose(best_log_prob, expected_log_prob, atol=1e-8)


# ---------------------------------------------------------------------
# Test 2: hand-computed toy example with an unambiguous best path
# ---------------------------------------------------------------------

def test_viterbi_matches_hand_computation():
    # Constructed so the best path is unambiguous: state 0 is strongly
    # favored early (high pi, strong self-transition, high emission
    # for early observations), state 1 strongly favored late.
    pi = np.array([0.9, 0.1])
    A = np.array([
        [0.6, 0.4],
        [0.1, 0.9],
    ])
    B = np.array([
        [0.9, 0.8, 0.1, 0.05],
        [0.1, 0.2, 0.9, 0.95],
    ])

    log_pi = np.log(pi)
    log_A = np.log(A)
    log_B = np.log(B)

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    # Hand-computed via brute-force enumeration over all 2^4 paths,
    # to avoid re-deriving the DP recurrence by hand and accidentally
    # encoding the same bug the implementation might have.
    import itertools

    best_prob_bf = -np.inf
    best_path_bf = None
    for path in itertools.product([0, 1], repeat=4):
        prob = pi[path[0]] * B[path[0], 0]
        for t in range(1, 4):
            prob *= A[path[t - 1], path[t]] * B[path[t], t]
        if prob > best_prob_bf:
            best_prob_bf = prob
            best_path_bf = path

    np.testing.assert_array_equal(best_path, np.array(best_path_bf))
    np.testing.assert_allclose(np.exp(best_log_prob), best_prob_bf, atol=1e-10)


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_viterbi_single_timestep():
    log_pi = np.log(np.array([0.3, 0.7]))
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    log_B = np.log(np.array([[0.2], [0.9]]))

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    # State 1 should win: 0.7*0.9 = 0.63 vs state 0: 0.3*0.2 = 0.06
    assert best_path.shape == (1,)
    assert best_path[0] == 1
    np.testing.assert_allclose(np.exp(best_log_prob), 0.63, atol=1e-10)


def test_viterbi_raises_on_nonsquare_A():
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.log(np.array([[0.7, 0.2, 0.1], [0.4, 0.5, 0.1]]))  # 2x3
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3]]))

    with pytest.raises(ValueError):
        viterbi(log_pi, log_A, log_B)


def test_viterbi_raises_on_shape_mismatch():
    log_pi = np.log(np.array([0.5, 0.3, 0.2]))  # 3 states
    log_A = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))  # 2 states
    log_B = np.log(np.array([[0.5, 0.4], [0.2, 0.3]]))  # 2 states

    with pytest.raises(ValueError):
        viterbi(log_pi, log_A, log_B)


def test_viterbi_single_state():
    log_pi = np.log(np.array([1.0]))
    log_A = np.log(np.array([[1.0]]))
    log_B = np.log(np.array([[0.5, 0.3, 0.7]]))

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    assert best_path.shape == (3,)
    np.testing.assert_array_equal(best_path, np.array([0, 0, 0]))
    expected_log_prob = np.sum(np.log(np.array([0.5, 0.3, 0.7])))
    np.testing.assert_allclose(best_log_prob, expected_log_prob, atol=1e-10)


# ---------------------------------------------------------------------
# Masked (-inf) transitions: relevant to future profile HMM topology,
# where structurally forbidden transitions are represented as -inf
# in log_A. Viterbi must never "choose" a -inf-probability path when
# a finite-probability path exists.
# ---------------------------------------------------------------------

def test_viterbi_respects_masked_transitions():
    # State 0 cannot transition to state 1 at all (masked as -inf).
    # A naive/buggy implementation that mishandles -inf could still
    # pick an impossible path if e.g. -inf + -inf produced nan and
    # nan comparisons behaved unpredictably in max/argmax.
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.array([
        [0.0, -np.inf],   # state 0 -> state 0 only
        [-np.inf, 0.0],   # state 1 -> state 1 only
    ])
    log_B = np.log(np.array([
        [0.9, 0.1, 0.9],
        [0.1, 0.9, 0.1],
    ]))

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    # With no valid cross-transitions, the only reachable paths are
    # all-state-0 or all-state-1. Given the emissions, all-state-0
    # (favored at t=0 and t=2) should win over all-state-1.
    assert np.all(best_path == best_path[0])
    assert np.isfinite(best_log_prob)


def test_viterbi_underflow_resistance_long_sequence():
    # A long sequence with modest per-step probabilities would
    # underflow to exactly 0.0 in naive (non-log) probability space.
    # In log-space, the result must remain finite and informative.
    T = 2000
    log_pi = np.log(np.array([0.5, 0.5]))
    log_A = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
    rng = np.random.default_rng(42)
    B = rng.uniform(0.05, 0.95, size=(2, T))
    log_B = np.log(B)

    best_path, best_log_prob = viterbi(log_pi, log_A, log_B)

    assert best_path.shape == (T,)
    assert np.isfinite(best_log_prob)
    # Sanity: naive probability-space product would underflow to 0.
    with np.errstate(under="ignore"):
        naive_prob = np.exp(best_log_prob)
    assert naive_prob == 0.0  # confirms *why* log-space is necessary