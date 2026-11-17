"""Tests for bounded-memory Monte Carlo BPSK experiments."""

import pytest

from commulab.analysis.ber import simulate_bpsk_ber


def test_multiple_batches_process_all_bits() -> None:
    result = simulate_bpsk_ber(
        0.0,
        bit_count=10_003,
        batch_size=1024,
        seed=123,
    )

    assert result.bits_transmitted == 10_003
    assert 0 < result.bit_errors < 10_003
    assert result.simulated_ber == (
        result.bit_errors / 10_003
    )


def test_batched_simulation_is_reproducible() -> None:
    options = {
        "bit_count": 50_003,
        "batch_size": 4096,
        "seed": 2026,
    }

    first = simulate_bpsk_ber(3.0, **options)
    second = simulate_bpsk_ber(3.0, **options)

    assert first == second


def test_single_batch_matches_default() -> None:
    default = simulate_bpsk_ber(
        3.0,
        bit_count=10_000,
        seed=99,
    )

    explicit = simulate_bpsk_ber(
        3.0,
        bit_count=10_000,
        batch_size=10_000,
        seed=99,
    )

    assert default == explicit


@pytest.mark.parametrize("batch_size", [1, 17, 1024, 20000])
def test_different_batch_sizes_are_statistically_valid(
    batch_size: int,
) -> None:
    result = simulate_bpsk_ber(
        0.0,
        bit_count=20_000,
        batch_size=batch_size,
        seed=123,
    )

    assert (
        result.confidence_lower
        <= result.simulated_ber
        <= result.confidence_upper
    )

    # Broad statistical guard, not exact stream equality.
    assert abs(result.simulated_ber - result.theoretical_ber) < 0.015


def test_large_experiment_in_small_batches() -> None:
    result = simulate_bpsk_ber(
        6.0,
        bit_count=1_000_000,
        batch_size=10_000,
        seed=2026,
    )

    assert result.bits_transmitted == 1_000_000
    assert result.bit_errors > 0
    assert (
        result.confidence_lower
        <= result.theoretical_ber
        <= result.confidence_upper
    )


def test_zero_error_batches_retain_confidence_bound() -> None:
    result = simulate_bpsk_ber(
        20.0,
        bit_count=100_000,
        batch_size=1000,
        seed=42,
    )

    assert result.bit_errors == 0
    assert result.simulated_ber == 0.0
    assert result.confidence_lower == 0.0
    assert result.confidence_upper > 0.0
    assert result.theoretical_ber > 0.0


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_value(batch_size: int) -> None:
    with pytest.raises(ValueError):
        simulate_bpsk_ber(0.0, batch_size=batch_size)


@pytest.mark.parametrize("batch_size", [True, 1.5, "1000"])
def test_invalid_batch_size_type(batch_size: object) -> None:
    with pytest.raises(TypeError):
        simulate_bpsk_ber(0.0, batch_size=batch_size)
