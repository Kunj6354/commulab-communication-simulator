"""Gray-coded 16-QAM modulation with unit average symbol energy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

SQRT10 = np.sqrt(10.0)
INNER_OUTER_BOUNDARY = 2.0 / SQRT10


def qam16_modulate(bits: ArrayLike) -> NDArray[np.complex128]:
    """Map groups of four bits to Gray-coded 16-QAM symbols.

    Each axis uses:
        00 -> +3
        01 -> +1
        11 -> -1
        10 -> -3

    Every complex symbol is divided by sqrt(10), giving
    average constellation energy Es = 1 for equally likely symbols.
    """
    values = np.asarray(bits)

    if values.ndim != 1:
        raise ValueError("bits must be one-dimensional")
    if values.size == 0:
        raise ValueError("bits must not be empty")
    if values.size % 4 != 0:
        raise ValueError("16-QAM requires a multiple of four bits")
    if values.dtype.kind not in "biu":
        raise TypeError("bits must have a boolean or integer dtype")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must contain only 0 and 1")

    groups = values.astype(np.float64).reshape(-1, 4)

    i_sign = 1.0 - 2.0 * groups[:, 0]
    i_amplitude = 3.0 - 2.0 * groups[:, 1]

    q_sign = 1.0 - 2.0 * groups[:, 2]
    q_amplitude = 3.0 - 2.0 * groups[:, 3]

    symbols = (
        i_sign * i_amplitude
        + 1j * q_sign * q_amplitude
    ) / SQRT10

    return np.asarray(symbols, dtype=np.complex128)


def qam16_demodulate(samples: ArrayLike) -> NDArray[np.uint8]:
    """Recover Gray-coded bits using hard-decision I/Q thresholds.

    Zero is assigned to the positive half-plane.
    A point exactly on an inner/outer boundary is assigned
    to the outer-amplitude region.
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

    in_phase = np.real(values)
    quadrature = np.imag(values)

    bits = np.empty(4 * values.size, dtype=np.uint8)
    bits[0::4] = in_phase < 0.0
    bits[1::4] = np.abs(in_phase) < INNER_OUTER_BOUNDARY
    bits[2::4] = quadrature < 0.0
    bits[3::4] = np.abs(quadrature) < INNER_OUTER_BOUNDARY

    return bits
