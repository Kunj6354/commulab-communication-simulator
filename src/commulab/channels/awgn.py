"""Seeded real AWGN channel for unit-energy BPSK symbols."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def bpsk_noise_std(ebn0_db: float) -> float:
    """Return real AWGN standard deviation for unit-energy BPSK.

    Eb = 1 and sigma^2 = N0 / 2.
    """
    if isinstance(ebn0_db, (bool, np.bool_)) or not isinstance(
        ebn0_db, (int, float, np.integer, np.floating)
    ):
        raise TypeError("ebn0_db must be a real number")

    db = float(ebn0_db)

    if not math.isfinite(db):
        raise ValueError("ebn0_db must be finite")

    try:
        sigma = math.exp(-db * math.log(10.0) / 20.0) / math.sqrt(2.0)
    except OverflowError as exc:
        raise ValueError("ebn0_db is outside the supported range") from exc

    if not math.isfinite(sigma) or sigma == 0.0:
        raise ValueError("ebn0_db is outside the supported range")

    return sigma


def add_bpsk_awgn(
    symbols: ArrayLike,
    ebn0_db: float,
    *,
    seed: int,
) -> NDArray[np.float64]:
    """Add seeded Gaussian noise to a 1D array of unit-energy BPSK symbols."""
    values = np.asarray(symbols)

    if values.ndim != 1:
        raise ValueError("symbols must be one-dimensional")
    if values.size == 0:
        raise ValueError("symbols must not be empty")
    if values.dtype.kind not in "biuf":
        raise TypeError("symbols must be real-valued")
    if not np.all(np.isfinite(values)):
        raise ValueError("symbols must be finite")
    if not np.all((values == -1) | (values == 1)):
        raise ValueError("symbols must be unit-energy BPSK values (+1 or -1)")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")

    sigma = bpsk_noise_std(ebn0_db)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=sigma, size=values.size)

    return values.astype(np.float64) + noise
