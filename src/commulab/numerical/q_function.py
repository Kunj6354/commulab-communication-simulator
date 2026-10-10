"""Reference Gaussian Q-function implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr, ndtr


def q_function(x: ArrayLike) -> NDArray[np.float64]:
    """Return Q(x) = P(Z > x), for standard normal Z."""
    values = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    return np.asarray(ndtr(-values), dtype=np.float64)


def log_q_function(x: ArrayLike) -> NDArray[np.float64]:
    """Return log(Q(x)) without avoidable probability underflow."""
    values = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    return np.asarray(log_ndtr(-values), dtype=np.float64)
