"""Complex AWGN for constellations normalized to average Es = 1."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def complex_noise_std(
    ebn0_db: float,
    *,
    bits_per_symbol: int,
) -> float:
    """Return the Gaussian noise standard deviation per I/Q axis.

    Assumes average symbol energy Es = 1 and Eb = 1 / k,
    where k is bits_per_symbol.
    """
    if isinstance(bits_per_symbol, (bool, np.bool_)) or not isinstance(
        bits_per_symbol, (int, np.integer)
    ):
        raise TypeError("bits_per_symbol must be an integer")
    if bits_per_symbol < 1:
        raise ValueError("bits_per_symbol must be positive")

    if isinstance(ebn0_db, (bool, np.bool_)) or not isinstance(
        ebn0_db, (int, float, np.integer, np.floating)
    ):
        raise TypeError("ebn0_db must be a real number")

    db = float(ebn0_db)
    if not math.isfinite(db):
        raise ValueError("ebn0_db must be finite")

    try:
        sigma = (
            math.exp(-db * math.log(10.0) / 20.0)
            / math.sqrt(2.0 * bits_per_symbol)
        )
    except (OverflowError, ValueError) as exc:
        raise ValueError("noise scale is outside supported range") from exc

    if not math.isfinite(sigma) or sigma == 0.0:
        raise ValueError("noise scale is outside supported range")

    return sigma


def add_complex_awgn(
    symbols: ArrayLike,
    ebn0_db: float,
    *,
    bits_per_symbol: int,
    seed: int,
) -> NDArray[np.complex128]:
    """Add seeded independent I/Q noise to complex symbols.

    The caller must supply a constellation with average Es = 1.
    This function does not rescale the input signal.
    """
    values = np.asarray(symbols)

    if values.ndim != 1:
        raise ValueError("symbols must be one-dimensional")
    if values.size == 0:
        raise ValueError("symbols must not be empty")
    if values.dtype.kind != "c":
        raise TypeError("symbols must have a complex dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("symbols must be finite")

    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer")

    sigma = complex_noise_std(
        ebn0_db,
        bits_per_symbol=bits_per_symbol,
    )

    rng = np.random.default_rng(int(seed))
    noise_i = rng.normal(0.0, sigma, size=values.size)
    noise_q = rng.normal(0.0, sigma, size=values.size)

    return np.asarray(
        values + noise_i + 1j * noise_q,
        dtype=np.complex128,
    )
