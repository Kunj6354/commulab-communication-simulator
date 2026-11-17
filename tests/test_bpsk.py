"""Unit tests for deterministic BPSK modulation and demodulation."""

import numpy as np
import pytest

from commulab.modulation.bpsk import (
    bpsk_demodulate,
    bpsk_modulate,
    generate_bits,
)


def test_known_bit_mapping() -> None:
    bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    expected = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
    np.testing.assert_array_equal(bpsk_modulate(bits), expected)


def test_noiseless_round_trip() -> None:
    bits = generate_bits(10000, seed=42)
    recovered = bpsk_demodulate(bpsk_modulate(bits))
    np.testing.assert_array_equal(recovered, bits)


def test_seeded_generation_is_reproducible() -> None:
    a = generate_bits(1000, seed=123)
    b = generate_bits(1000, seed=123)
    np.testing.assert_array_equal(a, b)


def test_seeded_generation_changes_with_seed() -> None:
    a = generate_bits(1000, seed=123)
    b = generate_bits(1000, seed=124)
    assert np.any(a != b)


def test_bit_generator_contract() -> None:
    bits = generate_bits(32, seed=7)
    assert bits.shape == (32,)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)).issubset({0, 1})


@pytest.mark.parametrize("bad_count", [0, -1])
def test_invalid_bit_count_value(bad_count: int) -> None:
    with pytest.raises(ValueError):
        generate_bits(bad_count, seed=42)


@pytest.mark.parametrize("bad_count", [1.5, True, "10"])
def test_invalid_bit_count_type(bad_count: object) -> None:
    with pytest.raises(TypeError):
        generate_bits(bad_count, seed=42)


@pytest.mark.parametrize("bad_seed", [1.5, True, "42"])
def test_invalid_seed_type(bad_seed: object) -> None:
    with pytest.raises(TypeError):
        generate_bits(10, seed=bad_seed)


@pytest.mark.parametrize(
    "invalid_bits",
    [
        [0, 2, 1],
        [-1, 0, 1],
    ],
)
def test_invalid_bit_values(invalid_bits: list[int]) -> None:
    with pytest.raises(ValueError):
        bpsk_modulate(invalid_bits)


@pytest.mark.parametrize(
    "invalid_bits",
    [
        [0.0, 1.0],
        [0.0, float("nan")],
        ["0", "1"],
    ],
)
def test_invalid_bit_types(invalid_bits: list[object]) -> None:
    with pytest.raises(TypeError):
        bpsk_modulate(invalid_bits)


def test_reject_multidimensional_bits() -> None:
    with pytest.raises(ValueError):
        bpsk_modulate([[0, 1], [1, 0]])


def test_reject_empty_bits() -> None:
    with pytest.raises(ValueError):
        bpsk_modulate([])


def test_unit_symbol_energy() -> None:
    bits = generate_bits(1000, seed=99)
    symbols = bpsk_modulate(bits)
    assert np.all(symbols**2 == 1.0)


def test_hard_decision_threshold() -> None:
    samples = [-2.0, -0.01, 0.0, 0.01, 2.0]
    expected = np.array([1, 1, 0, 0, 0], dtype=np.uint8)
    np.testing.assert_array_equal(bpsk_demodulate(samples), expected)


def test_reject_nonfinite_samples() -> None:
    with pytest.raises(ValueError):
        bpsk_demodulate([0.0, float("inf")])


def test_reject_complex_samples() -> None:
    with pytest.raises(TypeError):
        bpsk_demodulate(np.array([1 + 1j]))


def test_reject_empty_samples() -> None:
    with pytest.raises(ValueError):
        bpsk_demodulate([])


def test_reject_multidimensional_samples() -> None:
    with pytest.raises(ValueError):
        bpsk_demodulate([[1.0, -1.0]])
