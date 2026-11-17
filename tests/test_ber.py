"""Tests for BPSK theoretical and Monte Carlo BER."""

import math
from itertools import pairwise

import pytest

from commulab.analysis.ber import (
    simulate_bpsk_ber,
    theoretical_bpsk_ber,
)


@pytest.mark.parametrize(
    ("db", "expected"),
    [
        (0.0, 0.07864960352514257),
        (3.0, 0.02287840756108532),
        (6.0, 0.0023882907809328075),
    ],
)
def test_theoretical_ber_reference(db: float, expected: float) -> None:
    assert theoretical_bpsk_ber(db) == pytest.approx(expected, rel=1e-10)


def test_simulation_reproducibility() -> None:
    first = simulate_bpsk_ber(3.0, bit_count=100_000, seed=123)
    second = simulate_bpsk_ber(3.0, bit_count=100_000, seed=123)
    assert first == second


def test_result_consistency() -> None:
    result = simulate_bpsk_ber(0.0, bit_count=10_000, seed=10)
    assert result.bits_transmitted == 10_000
    assert 0 <= result.bit_errors <= result.bits_transmitted
    assert result.simulated_ber == result.bit_errors / result.bits_transmitted
    assert result.theoretical_ber > 0.0


@pytest.mark.parametrize("db", [0.0, 3.0, 6.0])
def test_monte_carlo_agrees_with_theory(db: float) -> None:
    result = simulate_bpsk_ber(db, bit_count=200_000, seed=2026)

    probability = result.theoretical_ber
    standard_error = math.sqrt(
        probability * (1.0 - probability) / result.bits_transmitted
    )

    assert abs(result.simulated_ber - probability) <= (
        5.0 * standard_error + 1.0 / result.bits_transmitted
    )


def test_ber_decreases_as_ebn0_increases() -> None:
    db_values = (0.0, 3.0, 6.0)
    results = [
        simulate_bpsk_ber(db, bit_count=200_000, seed=42)
        for db in db_values
    ]
    assert all(
        first.simulated_ber > second.simulated_ber
        for first, second in pairwise(results)
    )


def test_zero_observed_errors_do_not_change_theory() -> None:
    result = simulate_bpsk_ber(20.0, bit_count=1000, seed=42)
    assert result.bit_errors == 0
    assert result.simulated_ber == 0.0
    assert result.theoretical_ber > 0.0


@pytest.mark.parametrize("bit_count", [0, -1])
def test_reject_invalid_bit_count_values(bit_count: int) -> None:
    with pytest.raises(ValueError):
        simulate_bpsk_ber(0.0, bit_count=bit_count)


@pytest.mark.parametrize("bit_count", [1.5, True, "100"])
def test_reject_invalid_bit_count_types(bit_count: object) -> None:
    with pytest.raises(TypeError):
        simulate_bpsk_ber(0.0, bit_count=bit_count)


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_reject_invalid_seed_types(seed: object) -> None:
    with pytest.raises(TypeError):
        simulate_bpsk_ber(0.0, seed=seed)


@pytest.mark.parametrize("bad_db", [math.nan, math.inf, -math.inf])
def test_reject_nonfinite_ebn0(bad_db: float) -> None:
    with pytest.raises(ValueError):
        theoretical_bpsk_ber(bad_db)


def test_reject_unrepresentable_ebn0() -> None:
    with pytest.raises(ValueError):
        theoretical_bpsk_ber(10000.0)
