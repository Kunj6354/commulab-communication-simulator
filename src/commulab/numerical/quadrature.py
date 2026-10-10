"""Composite quadrature methods for Craig's Gaussian Q-function integral."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

QuadratureMethod = Literal[
    "left",
    "right",
    "midpoint",
    "trapezoidal",
    "simpson",
]


def _integrand(
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate Craig's integrand with explicit endpoint handling."""
    sine_squared = np.sin(theta) ** 2
    exponent = np.zeros(np.broadcast_shapes(x.shape, theta.shape), dtype=np.float64)

    np.divide(
        -(x * x),
        2.0 * sine_squared,
        out=exponent,
        where=sine_squared > 0,
    )

    result = np.exp(exponent)

    return np.where(
        sine_squared == 0,
        np.where(x == 0, 1.0, 0.0),
        result,
    )


def craig_quadrature(
    x: ArrayLike,
    *,
    method: QuadratureMethod = "simpson",
    intervals: int = 100,
) -> NDArray[np.float64]:
    """Approximate Q(x) for nonnegative real x using composite quadrature.

    Simpson's rule requires a positive even number of intervals.
    Results retain the input shape, including scalar zero-dimensional arrays.
    """
    values = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("x must be finite")

    if np.any(values < 0):
        raise ValueError("Craig's integral requires x >= 0")

    if isinstance(intervals, (bool, np.bool_)) or not isinstance(
        intervals, (int, np.integer)
    ):
        raise TypeError("intervals must be a positive integer")

    if intervals < 1:
        raise ValueError("intervals must be positive")

    if method not in ("left", "right", "midpoint", "trapezoidal", "simpson"):
        raise ValueError("Unknown quadrature method")

    if method == "simpson" and intervals % 2:
        raise ValueError("Simpson's rule requires even intervals")

    h = (np.pi / 2.0) / intervals
    indices = np.arange(intervals + 1, dtype=np.float64)
    theta = h * indices

    expanded_x = np.expand_dims(values, axis=-1)
    samples = _integrand(expanded_x, theta)

    if method == "left":
        integral = h * np.sum(samples[..., :-1], axis=-1)
    elif method == "right":
        integral = h * np.sum(samples[..., 1:], axis=-1)
    elif method == "trapezoidal":
        integral = h * (
            0.5 * samples[..., 0]
            + np.sum(samples[..., 1:-1], axis=-1)
            + 0.5 * samples[..., -1]
        )
    elif method == "midpoint":
        mid_theta = h * (np.arange(intervals, dtype=np.float64) + 0.5)
        integral = h * np.sum(_integrand(expanded_x, mid_theta), axis=-1)
    else:
        integral = (h / 3.0) * (
            samples[..., 0]
            + samples[..., -1]
            + 4.0 * np.sum(samples[..., 1:-1:2], axis=-1)
            + 2.0 * np.sum(samples[..., 2:-1:2], axis=-1)
        )

    return np.asarray(integral / np.pi, dtype=np.float64)
