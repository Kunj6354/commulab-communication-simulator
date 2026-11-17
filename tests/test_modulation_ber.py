"""Monte Carlo integration tests for QPSK and Gray-coded 16-QAM."""

import math

import pytest

from commulab.analysis.modulation_ber import simulate_modulation_ber


@pytest.mark.parametrize(
    ("modulation", "db"),
    [
        ("qpsk", 0.0),
        ("qpsk", 3.0),
        ("qpsk", 6.0),
        ("16qam", 0.0),
        ("16qam", 4.0),
        ("16qam", 8.0),
    ],
)
def test_monte_carlo_matches_theory(
    modulation: str, db: float
) -> None:
    result = simulate_modulation_ber(
        modulation,
        db,
        bit_count=400_000,
        batch_symbols=20_000,
        seed=2026,
    )

    p = result.theoretical_ber
    standard_error = math.sqrt(
        p * (1.0 - p) / result.bits_transmitted
    )

    assert abs(result.simulated_ber - p) <= (
        5.0 * standard_error + 1.0 / result.bits_transmitted
    )


@pytest.mark.parametrize(
    ("modulation", "bits_per_symbol"),
    [("qpsk", 2), ("16qam", 4)],
)
def test_bit_and_symbol_accounting(
    modulation: str, bits_per_symbol: int
) -> None:
    result = simulate_modulation_ber(
        modulation,
        4.0,
        bit_count=20_000,
        batch_symbols=1024,
        seed=7,
    )

    assert result.bits_per_symbol == bits_per_symbol
    assert result.bits_transmitted == 20_000
    assert result.symbols_transmitted == 20_000 // bits_per_symbol
    assert 0 <= result.bit_errors <= result.bits_transmitted
    assert result.simulated_ber == (
        result.bit_errors / result.bits_transmitted
    )


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_fixed_configuration_is_reproducible(modulation: str) -> None:
    first = simulate_modulation_ber(
        modulation, 5.0, bit_count=40_000,
        batch_symbols=113, seed=123
    )
    second = simulate_modulation_ber(
        modulation, 5.0, bit_count=40_000,
        batch_symbols=113, seed=123
    )

    assert first == second


@pytest.mark.parametrize("modulation", ["qpsk", "16qam"])
def test_confidence_interval_contains_observation(
    modulation: str,
) -> None:
    result = simulate_modulation_ber(
        modulation, 4.0, bit_count=40_000, seed=99
    )

    assert (
        0.0
        <= result.confidence_lower
        <= result.simulated_ber
        <= result.confidence_upper
        <= 1.0
    )


def test_zero_errors_have_positive_upper_bound() -> None:
    result = simulate_modulation_ber(
        "qpsk", 20.0, bit_count=4000,
        batch_symbols=101, seed=42
    )

    assert result.bit_errors == 0
    assert result.simulated_ber == 0.0
    assert result.confidence_lower == 0.0
    assert result.confidence_upper > 0.0
    assert result.theoretical_ber > 0.0


@pytest.mark.parametrize(
    ("modulation", "bit_count"),
    [("qpsk", 3), ("16qam", 6), ("16qam", 7)],
)
def test_reject_incomplete_bit_groups(
    modulation: str, bit_count: int
) -> None:
    with pytest.raises(ValueError):
        simulate_modulation_ber(
            modulation, 0.0, bit_count=bit_count
        )


@pytest.mark.parametrize("bit_count", [0, -4])
def test_reject_invalid_bit_count_value(bit_count: int) -> None:
    with pytest.raises(ValueError):
        simulate_modulation_ber(
            "16qam", 0.0, bit_count=bit_count
        )


@pytest.mark.parametrize("bit_count", [True, 1.5, "400"])
def test_reject_invalid_bit_count_type(bit_count: object) -> None:
    with pytest.raises(TypeError):
        simulate_modulation_ber(
            "16qam", 0.0, bit_count=bit_count
        )


@pytest.mark.parametrize("batch_symbols", [0, -1])
def test_reject_invalid_batch_size_value(batch_symbols: int) -> None:
    with pytest.raises(ValueError):
        simulate_modulation_ber(
            "qpsk", 0.0, batch_symbols=batch_symbols
        )


@pytest.mark.parametrize("batch_symbols", [True, 1.5, "100"])
def test_reject_invalid_batch_size_type(batch_symbols: object) -> None:
    with pytest.raises(TypeError):
        simulate_modulation_ber(
            "qpsk", 0.0, batch_symbols=batch_symbols
        )


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_reject_invalid_seed_type(seed: object) -> None:
    with pytest.raises(TypeError):
        simulate_modulation_ber("qpsk", 0.0, seed=seed)


def test_reject_unknown_modulation() -> None:
    with pytest.raises(ValueError):
        simulate_modulation_ber("8psk", 0.0)


@pytest.mark.parametrize("db", [math.nan, math.inf, -math.inf])
def test_reject_nonfinite_snr(db: float) -> None:
    with pytest.raises(ValueError):
        simulate_modulation_ber("16qam", db)
