"""Binomial confidence intervals for observed bit-error rates."""

from __future__ import annotations

import math
from statistics import NormalDist


def wilson_ber_interval(
    bit_errors: int,
    bits_transmitted: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a two-sided Wilson score interval for BER.

    Assumes independent Bernoulli bit-error observations.
    """
    for name, value in (
        ("bit_errors", bit_errors),
        ("bits_transmitted", bits_transmitted),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    if bits_transmitted <= 0:
        raise ValueError("bits_transmitted must be positive")

    if bit_errors < 0 or bit_errors > bits_transmitted:
        raise ValueError("bit_errors must be between 0 and bits_transmitted")

    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        raise TypeError("confidence must be a real number")

    confidence = float(confidence)

    if not math.isfinite(confidence) or not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1")

    n = bits_transmitted
    p = bit_errors / n
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    z2 = z * z

    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (
        z
        * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        / denominator
    )

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    # Avoid floating-point cancellation at exact binomial boundaries.
    if bit_errors == 0:
        lower = 0.0
    if bit_errors == bits_transmitted:
        upper = 1.0

    return lower, upper
