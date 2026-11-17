"""Binary phase-shift keying modulation and hard-decision detection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_bits(
    count: int,
    *,
    seed: int,
) -> NDArray[np.uint8]:
    """Generate a reproducible one-dimensional binary sequence."""
    if isinstance(count, (bool, np.bool_)) or not isinstance(
        count, (int, np.integer)
    ):
        raise TypeError("count must be an integer")
    if count < 1:
        raise ValueError("count must be positive")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    return rng.integers(0, 2, size=int(count), dtype=np.uint8)


def bpsk_modulate(bits: ArrayLike) -> NDArray[np.float64]:
    """Map binary bits to real, unit-energy BPSK symbols.

    Bit 0 maps to +1 and bit 1 maps to -1.
    """
    values = np.asarray(bits)

    if values.ndim != 1:
        raise ValueError("bits must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("bits must not be empty")
    if values.dtype.kind not in "biu":
        raise TypeError("bits must have a boolean or integer dtype")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must contain only 0 and 1")

    return 1.0 - 2.0 * values.astype(np.float64)


def bpsk_demodulate(samples: ArrayLike) -> NDArray[np.uint8]:
    """Recover bits using zero-threshold hard decisions."""
    values = np.asarray(samples)

    if values.ndim != 1:
        raise ValueError("samples must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("samples must not be empty")
    if values.dtype.kind not in "biuf":
        raise TypeError("samples must be real-valued")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")

    return np.asarray(values < 0.0, dtype=np.uint8)
