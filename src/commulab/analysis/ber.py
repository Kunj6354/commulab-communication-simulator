"""Bounded-memory Monte Carlo BPSK BER analysis over real AWGN."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from commulab.analysis.confidence import wilson_ber_interval
from commulab.channels.awgn import bpsk_noise_std
from commulab.modulation.bpsk import bpsk_demodulate, bpsk_modulate
from commulab.numerical.q_function import q_function


@dataclass(frozen=True, slots=True)
class BerResult:
    """One reproducible BPSK BER experiment result."""

    ebn0_db: float
    bits_transmitted: int
    bit_errors: int
    simulated_ber: float
    theoretical_ber: float
    seed: int
    confidence_lower: float
    confidence_upper: float


def theoretical_bpsk_ber(ebn0_db: float) -> float:
    """Return coherent BPSK theoretical BER over real AWGN."""
    sigma = bpsk_noise_std(ebn0_db)
    return float(q_function(1.0 / sigma))


def simulate_bpsk_ber(
    ebn0_db: float,
    *,
    bit_count: int = 100_000,
    seed: int = 42,
    batch_size: int = 100_000,
) -> BerResult:
    """Simulate BPSK using bounded-size bit and noise arrays.

    Bit and noise RNGs have independent deterministic streams.
    Results are reproducible for a given configuration.
    """
    for name, value in (
        ("bit_count", bit_count),
        ("batch_size", batch_size),
        ("seed", seed),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")

    if bit_count < 1:
        raise ValueError("bit_count must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    sigma = bpsk_noise_std(ebn0_db)

    # Keep the original bit and noise stream initialization.
    streams = np.random.SeedSequence(seed).spawn(2)
    noise_seed = int(streams[1].generate_state(1, dtype=np.uint64)[0])

    bit_rng = np.random.default_rng(int(seed))
    noise_rng = np.random.default_rng(noise_seed)

    errors = 0

    for start in range(0, bit_count, batch_size):
        count = min(batch_size, bit_count - start)

        bits = bit_rng.integers(0, 2, size=count, dtype=np.uint8)
        symbols = bpsk_modulate(bits)

        noise = noise_rng.normal(
            loc=0.0,
            scale=sigma,
            size=count,
        )
        received = symbols + noise
        recovered = bpsk_demodulate(received)

        errors += int(np.count_nonzero(bits != recovered))

    lower, upper = wilson_ber_interval(errors, bit_count)

    return BerResult(
        ebn0_db=float(ebn0_db),
        bits_transmitted=int(bit_count),
        bit_errors=errors,
        simulated_ber=errors / bit_count,
        theoretical_ber=theoretical_bpsk_ber(ebn0_db),
        seed=int(seed),
        confidence_lower=lower,
        confidence_upper=upper,
    )
