"""Tests for reproducible ideal and noisy constellation samples."""

import numpy as np
import pytest

from commulab.analysis.constellation import simulate_constellation


@pytest.mark.parametrize(
    ("modulation", "bits_per_symbol", "point_count"),
    [
        ("bpsk", 1, 2),
        ("qpsk", 2, 4),
        ("16qam", 4, 16),
    ],
)
def test_constellation_dimensions_and_points(
    modulation: str,
    bits_per_symbol: int,
    point_count: int,
) -> None:
    result = simulate_constellation(
        modulation,
        6.0,
        symbol_count=2_000,
        seed=42,
    )

    assert result.bits_per_symbol == bits_per_symbol
    assert result.transmitted_bits == 2_000 * bits_per_symbol
    assert result.ideal_symbols.shape == (2_000,)
    assert result.received_symbols.shape == (2_000,)
    assert np.iscomplexobj(result.ideal_symbols)
    assert np.iscomplexobj(result.received_symbols)
    assert np.unique(result.ideal_symbols).size == point_count
    assert 0 <= result.observed_ber <= 1


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "16qam"])
def test_same_configuration_is_reproducible(modulation: str) -> None:
    first = simulate_constellation(
        modulation, 4.0, symbol_count=500, seed=123
    )
    second = simulate_constellation(
        modulation, 4.0, symbol_count=500, seed=123
    )

    assert np.array_equal(first.ideal_symbols, second.ideal_symbols)
    assert np.array_equal(first.received_symbols, second.received_symbols)
    assert first.bit_errors == second.bit_errors


@pytest.mark.parametrize("modulation", ["bpsk", "qpsk", "16qam"])
def test_noiseless_limit_has_no_errors(modulation: str) -> None:
    result = simulate_constellation(
        modulation, 100.0, symbol_count=500, seed=7
    )
    assert result.bit_errors == 0


def test_different_seeds_produce_different_samples() -> None:
    first = simulate_constellation("qpsk", 6.0, seed=1)
    second = simulate_constellation("qpsk", 6.0, seed=2)

    assert not np.array_equal(
        first.received_symbols, second.received_symbols
    )


@pytest.mark.parametrize(
    ("modulation", "count", "seed", "exception"),
    [
        ("invalid", 100, 42, ValueError),
        ("qpsk", 0, 42, ValueError),
        ("16qam", -1, 42, ValueError),
        ("bpsk", True, 42, TypeError),
        ("bpsk", 100, -1, ValueError),
        ("qpsk", 100, True, TypeError),
    ],
)
def test_invalid_inputs_are_rejected(
    modulation: str,
    count: int,
    seed: int,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        simulate_constellation(
            modulation, 6.0, symbol_count=count, seed=seed
        )


def test_bpsk_has_no_quadrature_component() -> None:
    result = simulate_constellation("bpsk", 3.0)
    assert np.all(result.ideal_symbols.imag == 0.0)
    assert np.all(result.received_symbols.imag == 0.0)


def test_noise_changes_samples() -> None:
    result = simulate_constellation("16qam", 2.0)
    assert np.any(result.ideal_symbols != result.received_symbols)
