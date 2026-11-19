"""Reproducible constellation samples for the interactive dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from commulab.channels.awgn import bpsk_noise_std
from commulab.channels.complex_awgn import complex_noise_std
from commulab.modulation.bpsk import bpsk_demodulate, bpsk_modulate
from commulab.modulation.qam16 import qam16_demodulate, qam16_modulate
from commulab.modulation.qpsk import qpsk_demodulate, qpsk_modulate

ConstellationModulation = Literal["bpsk", "qpsk", "16qam"]


@dataclass(frozen=True, slots=True)
class ConstellationResult:
    """One deterministic constellation observation over AWGN."""

    modulation: ConstellationModulation
    ebn0_db: float
    symbol_count: int
    bits_per_symbol: int
    seed: int
    ideal_symbols: NDArray[np.complex128]
    received_symbols: NDArray[np.complex128]
    bit_errors: int

    @property
    def transmitted_bits(self) -> int:
        return self.symbol_count * self.bits_per_symbol

    @property
    def observed_ber(self) -> float:
        return self.bit_errors / self.transmitted_bits


def simulate_constellation(
    modulation: ConstellationModulation,
    ebn0_db: float,
    *,
    symbol_count: int = 1_000,
    seed: int = 42,
) -> ConstellationResult:
    """Generate ideal and noisy symbols without changing the BER engine."""
    if modulation not in ("bpsk", "qpsk", "16qam"):
        raise ValueError("Unsupported modulation")

    for name, value in (("symbol_count", symbol_count), ("seed", seed)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    if symbol_count < 1:
        raise ValueError("symbol_count must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    bits_per_symbol = {"bpsk": 1, "qpsk": 2, "16qam": 4}[modulation]

    bit_rng = np.random.default_rng(int(seed))
    noise_sequence = np.random.SeedSequence(int(seed)).spawn(2)[1]
    noise_rng = np.random.default_rng(noise_sequence)

    bits = bit_rng.integers(
        0,
        2,
        size=symbol_count * bits_per_symbol,
        dtype=np.uint8,
    )

    if modulation == "bpsk":
        real_symbols = bpsk_modulate(bits)
        sigma = bpsk_noise_std(ebn0_db)
        noisy_real = real_symbols + noise_rng.normal(
            0.0, sigma, size=symbol_count
        )
        recovered = bpsk_demodulate(noisy_real)

        ideal = np.asarray(real_symbols, dtype=np.complex128)
        received = np.asarray(noisy_real, dtype=np.complex128)

    else:
        if modulation == "qpsk":
            symbols = qpsk_modulate(bits)
        else:
            symbols = qam16_modulate(bits)

        sigma = complex_noise_std(
            ebn0_db,
            bits_per_symbol=bits_per_symbol,
        )
        noise_i = noise_rng.normal(0.0, sigma, size=symbol_count)
        noise_q = noise_rng.normal(0.0, sigma, size=symbol_count)

        received = np.asarray(
            symbols + noise_i + 1j * noise_q,
            dtype=np.complex128,
        )
        ideal = np.asarray(symbols, dtype=np.complex128)

        if modulation == "qpsk":
            recovered = qpsk_demodulate(received)
        else:
            recovered = qam16_demodulate(received)

    errors = int(np.count_nonzero(bits != recovered))

    return ConstellationResult(
        modulation=modulation,
        ebn0_db=float(ebn0_db),
        symbol_count=int(symbol_count),
        bits_per_symbol=bits_per_symbol,
        seed=int(seed),
        ideal_symbols=ideal,
        received_symbols=received,
        bit_errors=errors,
    )
