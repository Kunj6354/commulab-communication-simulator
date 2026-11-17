"""Gray-coded, unit-average-symbol-energy QPSK modulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

INV_SQRT2 = 1.0 / np.sqrt(2.0)


def qpsk_modulate(bits: ArrayLike) -> NDArray[np.complex128]:
    """Map bit pairs to QPSK symbols with unit symbol energy.

    Pairs 00, 01, 11, 10 are adjacent around the constellation.
    The first bit controls I and the second controls Q.
    """
    values = np.asarray(bits)

    if values.ndim != 1:
        raise ValueError("bits must be one-dimensional")
    if values.size == 0:
        raise ValueError("bits must not be empty")
    if values.size % 2 != 0:
        raise ValueError("QPSK requires an even number of bits")
    if values.dtype.kind not in "biu":
        raise TypeError("bits must have a boolean or integer dtype")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must contain only 0 and 1")

    pairs = values.astype(np.float64).reshape(-1, 2)

    in_phase = 1.0 - 2.0 * pairs[:, 0]
    quadrature = 1.0 - 2.0 * pairs[:, 1]

    return np.asarray(
        (in_phase + 1j * quadrature) * INV_SQRT2,
        dtype=np.complex128,
    )


def qpsk_demodulate(samples: ArrayLike) -> NDArray[np.uint8]:
    """Recover QPSK bit pairs using independent I/Q hard decisions.

    A zero-valued component is assigned bit zero.
    """
    values = np.asarray(samples)

    if values.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    if values.size == 0:
        raise ValueError("samples must not be empty")
    if values.dtype.kind not in "biufc":
        raise TypeError("samples must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must be finite")

    bits = np.empty(2 * values.size, dtype=np.uint8)
    bits[0::2] = np.real(values) < 0.0
    bits[1::2] = np.imag(values) < 0.0

    return bits
