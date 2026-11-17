"""Theoretical BER for normalized Gray-coded digital constellations."""

from __future__ import annotations

import math

from commulab.analysis.ber import theoretical_bpsk_ber
from commulab.channels.complex_awgn import complex_noise_std
from commulab.numerical.q_function import q_function


def theoretical_qpsk_ber(ebn0_db: float) -> float:
    """Return coherent Gray-coded QPSK BER over AWGN."""
    return theoretical_bpsk_ber(ebn0_db)


def theoretical_qam16_ber(ebn0_db: float) -> float:
    """Return exact Gray-coded 16-QAM BER for the CommuLab mapping.

    Assumes equiprobable symbols, normalized average Es = 1,
    independent I/Q AWGN, and ideal hard-decision detection.
    """
    sigma = complex_noise_std(ebn0_db, bits_per_symbol=4)
    a = 1.0 / (math.sqrt(10.0) * sigma)

    return float(
        0.75 * q_function(a)
        + 0.50 * q_function(3.0 * a)
        - 0.25 * q_function(5.0 * a)
    )
