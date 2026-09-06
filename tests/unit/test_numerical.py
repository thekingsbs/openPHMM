"""
Tests for openPHMM/utils/numerical.py
"""

import numpy as np
import pytest
from openPHMM.utils.numerical import log_sum_exp, log_normalize, safe_log


class TestLogSumExp:

    def test_basic(self):
        # log(exp(0) + exp(0)) = log(2)
        result = log_sum_exp(np.array([0.0, 0.0]))
        assert np.isclose(result, np.log(2))

    def test_single_element(self):
        result = log_sum_exp(np.array([3.0]))
        assert np.isclose(result, 3.0)

    def test_numerical_stability_large(self):
        # Should not overflow
        result = log_sum_exp(np.array([1000.0, 1000.0]))
        assert np.isfinite(result)
        assert np.isclose(result, 1000.0 + np.log(2))

    def test_numerical_stability_small(self):
        # Should not underflow
        result = log_sum_exp(np.array([-1000.0, -1000.0]))
        assert np.isfinite(result)
        assert np.isclose(result, -1000.0 + np.log(2))

    def test_all_neg_inf(self):
        result = log_sum_exp(np.array([-np.inf, -np.inf]))
        assert result == -np.inf

    def test_with_axis(self):
        arr = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = log_sum_exp(arr, axis=1)
        assert result.shape == (2,)
        assert np.allclose(result, np.log(2))

    def test_2d_no_axis(self):
        arr = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = log_sum_exp(arr)
        assert np.isclose(result, np.log(4))


class TestLogNormalize:

    def test_basic(self):
        log_probs = np.array([0.0, 0.0])
        result = log_normalize(log_probs)
        assert np.allclose(np.exp(result).sum(), 1.0)

    def test_already_normalized(self):
        log_probs = np.log(np.array([0.5, 0.5]))
        result = log_normalize(log_probs)
        assert np.allclose(np.exp(result), [0.5, 0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            log_normalize(np.array([]))

    def test_2d_axis(self):
        log_probs = np.array([[0.0, 0.0], [0.0, 0.0]])
        result = log_normalize(log_probs, axis=1)
        assert np.allclose(np.exp(result).sum(axis=1), [1.0, 1.0])


class TestSafeLog:

    def test_basic(self):
        result = safe_log(np.array([1.0]))
        assert np.isclose(result, 0.0)

    def test_zero_input(self):
        # Should not return -inf
        result = safe_log(np.array([0.0]))
        assert np.isfinite(result)

    def test_negative_floor_raises(self):
        with pytest.raises(ValueError):
            safe_log(np.array([1.0]), floor=-1.0)

    def test_zero_floor_raises(self):
        with pytest.raises(ValueError):
            safe_log(np.array([1.0]), floor=0.0)

    def test_array_input(self):
        result = safe_log(np.array([0.0, 1.0, 2.0]))
        assert result.shape == (3,)
        assert np.isfinite(result).all()