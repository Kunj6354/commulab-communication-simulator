"""Tests for confidence bounds integrated with BER simulations."""

import pytest

from commulab.analysis.ber import simulate_bpsk_ber
from commulab.analysis.confidence import wilson_ber_interval


def test_simulation_reports_wilson_bounds() -> None:
    result = simulate_bpsk_ber(
        0.0, bit_count=10_000, seed=123
    )

    expected = wilson_ber_interval(
        result.bit_errors, result.bits_transmitted
    )

    assert result.confidence_lower == pytest.approx(expected[0])
    assert result.confidence_upper == pytest.approx(expected[1])

    assert (
        result.confidence_lower
        <= result.simulated_ber
        <= result.confidence_upper
    )


def test_zero_error_simulation_has_positive_upper_bound() -> None:
    result = simulate_bpsk_ber(
        20.0, bit_count=1000, seed=42
    )

    assert result.bit_errors == 0
    assert result.simulated_ber == 0.0
    assert result.confidence_lower == 0.0
    assert result.confidence_upper > 0.0
    assert result.theoretical_ber > 0.0


def test_confidence_bounds_are_deterministic() -> None:
    first = simulate_bpsk_ber(
        3.0, bit_count=50_000, seed=2026
    )
    second = simulate_bpsk_ber(
        3.0, bit_count=50_000, seed=2026
    )

    assert first == second


def test_confidence_bounds_are_valid() -> None:
    for db in (0.0, 3.0, 6.0):
        result = simulate_bpsk_ber(
            db, bit_count=20_000, seed=123
        )

        assert 0.0 <= result.confidence_lower
        assert result.confidence_lower <= result.simulated_ber
        assert result.simulated_ber <= result.confidence_upper
        assert result.confidence_upper <= 1.0
