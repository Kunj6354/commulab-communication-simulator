"""Tests for numerical integration of Craig's Q-function integral."""

import numpy as np
import pytest

from commulab.numerical.q_function import q_function
from commulab.numerical.quadrature import craig_quadrature

METHODS = ("left", "right", "midpoint", "trapezoidal", "simpson")


@pytest.mark.parametrize("method", METHODS)
def test_q_zero_exact(method: str) -> None:
    # n=100 guarantees a valid Simpson configuration.
    result = craig_quadrature(0.0, method=method, intervals=100)
    np.testing.assert_allclose(result, 0.5, atol=1e-14, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_approximation_matches_reference(method: str) -> None:
    x = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    result = craig_quadrature(x, method=method, intervals=2000)
    expected = q_function(x)

    np.testing.assert_allclose(result, expected, atol=2e-4, rtol=0)


@pytest.mark.parametrize("method", METHODS)
def test_output_shape(method: str) -> None:
    values = np.array([[0.5, 1.0], [2.0, 3.0]])
    result = craig_quadrature(values, method=method, intervals=100)

    assert result.shape == values.shape


def test_simpson_requires_even_intervals() -> None:
    with pytest.raises(ValueError, match="even"):
        craig_quadrature(1.0, method="simpson", intervals=99)


@pytest.mark.parametrize("bad_intervals", [0, -1])
def test_invalid_interval_values(bad_intervals: int) -> None:
    with pytest.raises(ValueError):
        craig_quadrature(1.0, intervals=bad_intervals)


@pytest.mark.parametrize("bad_intervals", [1.5, True])
def test_invalid_interval_types(bad_intervals: object) -> None:
    with pytest.raises(TypeError):
        craig_quadrature(1.0, intervals=bad_intervals)


@pytest.mark.parametrize("bad_x", [-1.0, float("nan"), float("inf")])
def test_invalid_x(bad_x: float) -> None:
    with pytest.raises(ValueError):
        craig_quadrature(bad_x)


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        craig_quadrature(1.0, method="unknown")


def test_simpson_convergence() -> None:
    reference = float(q_function(1.0))

    coarse = float(craig_quadrature(1.0, method="simpson", intervals=4))
    fine = float(craig_quadrature(1.0, method="simpson", intervals=32))

    assert abs(fine - reference) < abs(coarse - reference)


def test_trapezoidal_convergence() -> None:
    reference = float(q_function(1.0))

    coarse = float(craig_quadrature(1.0, method="trapezoidal", intervals=4))
    fine = float(craig_quadrature(1.0, method="trapezoidal", intervals=32))

    assert abs(fine - reference) < abs(coarse - reference)


def test_small_grid_difference() -> None:
    left = float(craig_quadrature(1.0, method="left", intervals=4))
    right = float(craig_quadrature(1.0, method="right", intervals=4))

    assert left < right


def test_empty_input_preserves_shape() -> None:
    result = craig_quadrature(np.array([]), intervals=100)
    assert result.shape == (0,)
