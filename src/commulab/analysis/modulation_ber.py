"""Bounded-memory QPSK and Gray-coded 16-QAM BER simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from commulab.analysis.confidence import wilson_ber_interval
from commulab.analysis.modulation_theory import (
    theoretical_qam16_ber,
    theoretical_qpsk_ber,
)
from commulab.channels.complex_awgn import complex_noise_std
from commulab.modulation.qam16 import qam16_demodulate, qam16_modulate
from commulab.modulation.qpsk import qpsk_demodulate, qpsk_modulate

Modulation = Literal["qpsk", "16qam"]


@dataclass(frozen=True, slots=True)
class ModulationBerResult:
    """A reproducible complex-modulation BER experiment."""

    modulation: Modulation
    ebn0_db: float
    bits_per_symbol: int
    bits_transmitted: int
    symbols_transmitted: int
    bit_errors: int
    simulated_ber: float
    theoretical_ber: float
    confidence_lower: float
    confidence_upper: float
    seed: int


def simulate_modulation_ber(
    modulation: Modulation,
    ebn0_db: float,
    *,
    bit_count: int = 200_000,
    batch_symbols: int = 50_000,
    seed: int = 42,
) -> ModulationBerResult:
    """Simulate Gray-coded QPSK or 16-QAM over complex AWGN.

    Constellations have average Es = 1. Noise variance per axis is
    1 / (2 * bits_per_symbol * EbN0_linear).

    Bit and noise streams are independently seeded and deterministic.
    Memory use scales with batch_symbols, not the total bit count.
    """
    if modulation not in ("qpsk", "16qam"):
        raise ValueError("modulation must be 'qpsk' or '16qam'")

    for name, value in (
        ("bit_count", bit_count),
        ("batch_symbols", batch_symbols),
        ("seed", seed),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    if bit_count < 1:
        raise ValueError("bit_count must be positive")
    if batch_symbols < 1:
        raise ValueError("batch_symbols must be positive")

    bits_per_symbol = 2 if modulation == "qpsk" else 4

    if bit_count % bits_per_symbol:
        raise ValueError(
            "bit_count must be divisible by bits_per_symbol"
        )

    sigma = complex_noise_std(
        ebn0_db,
        bits_per_symbol=bits_per_symbol,
    )

    bit_rng = np.random.default_rng(int(seed))
    streams = np.random.SeedSequence(int(seed)).spawn(2)
    noise_seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])
    noise_rng = np.random.default_rng(noise_seed)

    total_symbols = bit_count // bits_per_symbol
    errors = 0

    for start in range(0, total_symbols, batch_symbols):
        count = min(batch_symbols, total_symbols - start)

        bits = bit_rng.integers(
            0,
            2,
            size=count * bits_per_symbol,
            dtype=np.uint8,
        )

        if modulation == "qpsk":
            symbols = qpsk_modulate(bits)
        else:
            symbols = qam16_modulate(bits)

        noise_i = noise_rng.normal(0.0, sigma, size=count)
        noise_q = noise_rng.normal(0.0, sigma, size=count)
        received = symbols + noise_i + 1j * noise_q

        if modulation == "qpsk":
            recovered = qpsk_demodulate(received)
        else:
            recovered = qam16_demodulate(received)

        errors += int(np.count_nonzero(bits != recovered))

    theoretical = (
        theoretical_qpsk_ber(ebn0_db)
        if modulation == "qpsk"
        else theoretical_qam16_ber(ebn0_db)
    )

    lower, upper = wilson_ber_interval(errors, bit_count)

    return ModulationBerResult(
        modulation=modulation,
        ebn0_db=float(ebn0_db),
        bits_per_symbol=bits_per_symbol,
        bits_transmitted=int(bit_count),
        symbols_transmitted=total_symbols,
        bit_errors=errors,
        simulated_ber=errors / bit_count,
        theoretical_ber=theoretical,
        confidence_lower=lower,
        confidence_upper=upper,
        seed=int(seed),
    )
