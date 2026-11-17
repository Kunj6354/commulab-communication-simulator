"""Tests for the unit-energy BPSK real AWGN channel."""

import numpy as np
import pytest

from commulab.channels.awgn import add_bpsk_awgn, bpsk_noise_std
from commulab.modulation.bpsk import (
    bpsk_demodulate,
    bpsk_modulate,
    generate_bits,
)


@pytest.mark.parametrize(
    ("ebn0_db", "expected_variance"),
    [
        (0.0, 0.5),
        (10.0, 0.05),
        (-10.0, 5.0),
    ],
)
def test_noise_variance_formula(
    ebn0_db: float, expected_variance: float
) -> None:
    assert bpsk_noise_std(ebn0_db) ** 2 == pytest.approx(
        expected_variance, rel=1e-12
    )


def test_seeded_noise_reproducibility() -> None:
    symbols = np.array([1.0, -1.0, 1.0, -1.0])
    first = add_bpsk_awgn(symbols, 4.0, seed=123)
    second = add_bpsk_awgn(symbols, 4.0, seed=123)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_noise() -> None:
    symbols = np.ones(100)
    first = add_bpsk_awgn(symbols, 4.0, seed=123)
    second = add_bpsk_awgn(symbols, 4.0, seed=124)
    assert np.any(first != second)


def test_noise_mean_and_variance() -> None:
    symbols = np.ones(200_000)
    received = add_bpsk_awgn(symbols, 0.0, seed=42)
    noise = received - symbols

    assert abs(float(np.mean(noise))) < 0.01
    assert float(np.var(noise)) == pytest.approx(0.5, rel=0.03)


def test_noise_increases_as_ebn0_decreases() -> None:
    assert bpsk_noise_std(-5.0) > bpsk_noise_std(0.0)
    assert bpsk_noise_std(0.0) > bpsk_noise_std(10.0)


def test_channel_preserves_shape_and_dtype() -> None:
    received = add_bpsk_awgn([1, -1, 1], 5.0, seed=7)
    assert received.shape == (3,)
    assert received.dtype == np.float64


def test_bpsk_pipeline_with_noise() -> None:
    bits = generate_bits(10000, seed=101)
    symbols = bpsk_modulate(bits)
    received = add_bpsk_awgn(symbols, 0.0, seed=202)
    recovered = bpsk_demodulate(received)

    errors = int(np.count_nonzero(bits != recovered))
    assert 0 < errors < bits.size


@pytest.mark.parametrize("bad_db", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_ebn0(bad_db: float) -> None:
    with pytest.raises(ValueError):
        bpsk_noise_std(bad_db)


@pytest.mark.parametrize("bad_db", [True, "10", 1 + 2j])
def test_reject_invalid_ebn0_types(bad_db: object) -> None:
    with pytest.raises(TypeError):
        bpsk_noise_std(bad_db)


@pytest.mark.parametrize("bad_seed", [True, 1.5, "42"])
def test_reject_invalid_seed_types(bad_seed: object) -> None:
    with pytest.raises(TypeError):
        add_bpsk_awgn([1, -1], 0.0, seed=bad_seed)


@pytest.mark.parametrize(
    "bad_symbols",
    [
        [1.0, 0.0],
        [2.0, -1.0],
    ],
)
def test_reject_non_bpsk_symbols(bad_symbols: list[float]) -> None:
    with pytest.raises(ValueError):
        add_bpsk_awgn(bad_symbols, 0.0, seed=1)


def test_reject_nonfinite_symbols() -> None:
    with pytest.raises(ValueError):
        add_bpsk_awgn([1.0, float("nan")], 0.0, seed=1)


def test_reject_complex_symbols() -> None:
    with pytest.raises(TypeError):
        add_bpsk_awgn(np.array([1 + 1j]), 0.0, seed=1)


def test_reject_empty_symbols() -> None:
    with pytest.raises(ValueError):
        add_bpsk_awgn([], 0.0, seed=1)


def test_reject_multidimensional_symbols() -> None:
    with pytest.raises(ValueError):
        add_bpsk_awgn([[1, -1]], 0.0, seed=1)


@pytest.mark.parametrize("extreme_db", [-10000.0, 10000.0])
def test_reject_unrepresentable_noise_scale(extreme_db: float) -> None:
    with pytest.raises(ValueError):
        bpsk_noise_std(extreme_db)
