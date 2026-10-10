"""Tests for the Gaussian Q-function reference."""

import numpy as np
import pytest
from scipy.special import ndtr

from commulab.numerical.q_function import log_q_function, q_function


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.5),
        (1.0, 0.15865525393145707),
        (2.0, 0.022750131948179195),
        (3.0, 0.0013498980316300933),
        (5.0, 2.866515718791933e-7),
    ],
)
def test_known_q_values(x: float, expected: float) -> None:
    np.testing.assert_allclose(q_function(x), expected, rtol=1e-12, atol=0)


def test_vectorized_input() -> None:
    x = np.array([0.0, 1.0, 2.0, 5.0])
    result = q_function(x)

    assert result.shape == x.shape
    np.testing.assert_allclose(result, ndtr(-x), rtol=1e-14)


def test_q_symmetry() -> None:
    x = np.array([0.0, 0.5, 1.0, 3.0])
    np.testing.assert_allclose(q_function(-x), 1.0 - q_function(x), atol=1e-14)


def test_q_monotonic_decreasing() -> None:
    x = np.linspace(-8.0, 8.0, 1001)
    differences = np.diff(q_function(x))

    # Floating-point saturation near Q(x) = 1 can produce equal values.
    assert np.all(differences <= 0)

    # Strict monotonicity where consecutive values are distinguishable.
    central_x = np.linspace(-5.0, 5.0, 1001)
    assert np.all(np.diff(q_function(central_x)) < 0)


def test_log_q_extreme_tail() -> None:
    value = float(log_q_function(40.0))
    assert np.isfinite(value)
    assert value < -800.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        q_function(value)

    with pytest.raises(ValueError):
        log_q_function(value)
