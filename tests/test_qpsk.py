"""Tests for Gray-coded QPSK modulation and demodulation."""

import numpy as np
import pytest

from commulab.modulation.bpsk import generate_bits
from commulab.modulation.qpsk import qpsk_demodulate, qpsk_modulate


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        ([0, 0], 1 + 1j),
        ([0, 1], 1 - 1j),
        ([1, 1], -1 - 1j),
        ([1, 0], -1 + 1j),
    ],
)
def test_known_constellation_points(
    bits: list[int], expected: complex
) -> None:
    symbol = qpsk_modulate(bits)
    assert symbol.shape == (1,)
    assert symbol.dtype == np.complex128
    assert symbol[0] == pytest.approx(expected / np.sqrt(2))


def test_all_constellation_points_are_unique() -> None:
    bits = [0, 0, 0, 1, 1, 1, 1, 0]
    symbols = qpsk_modulate(bits)
    assert len(np.unique(symbols)) == 4


def test_noiseless_round_trip() -> None:
    bits = generate_bits(20_000, seed=42)
    recovered = qpsk_demodulate(qpsk_modulate(bits))
    np.testing.assert_array_equal(recovered, bits)


def test_unit_symbol_energy() -> None:
    bits = generate_bits(10_000, seed=2026)
    symbols = qpsk_modulate(bits)
    np.testing.assert_allclose(
        np.abs(symbols) ** 2,
        1.0,
        rtol=0,
        atol=1e-14,
    )


def test_two_bits_per_symbol() -> None:
    bits = generate_bits(100, seed=123)
    symbols = qpsk_modulate(bits)

    assert symbols.shape == (50,)
    assert qpsk_demodulate(symbols).shape == (100,)


def test_gray_adjacency() -> None:
    # Moving around the square changes only one bit.
    labels = [(0, 0), (0, 1), (1, 1), (1, 0)]

    for first, second in zip(labels, labels[1:] + labels[:1]):
        assert sum(a != b for a, b in zip(first, second)) == 1


def test_demodulator_zero_threshold() -> None:
    samples = np.array([0 + 0j, -1 + 0j, 0 - 1j, -1 - 1j])

    np.testing.assert_array_equal(
        qpsk_demodulate(samples),
        np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.uint8),
    )


def test_demodulator_accepts_real_samples() -> None:
    np.testing.assert_array_equal(
        qpsk_demodulate([1.0, -1.0]),
        np.array([0, 0, 1, 0], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "bad_bits",
    [
        [0, 1, 1],
        [1],
    ],
)
def test_reject_odd_bit_count(bad_bits: list[int]) -> None:
    with pytest.raises(ValueError):
        qpsk_modulate(bad_bits)


@pytest.mark.parametrize(
    "bad_bits",
    [
        [0, 2],
        [-1, 1],
    ],
)
def test_reject_nonbinary_bits(bad_bits: list[int]) -> None:
    with pytest.raises(ValueError):
        qpsk_modulate(bad_bits)


@pytest.mark.parametrize(
    "bad_bits",
    [
        [0.0, 1.0],
        ["0", "1"],
        [0 + 0j, 1 + 0j],
    ],
)
def test_reject_invalid_bit_dtypes(bad_bits: list[object]) -> None:
    with pytest.raises(TypeError):
        qpsk_modulate(bad_bits)


def test_reject_empty_modulation_input() -> None:
    with pytest.raises(ValueError):
        qpsk_modulate([])


def test_reject_multidimensional_modulation_input() -> None:
    with pytest.raises(ValueError):
        qpsk_modulate([[0, 1], [1, 0]])


@pytest.mark.parametrize(
    "bad_samples",
    [
        np.array([np.nan + 1j]),
        np.array([1 + np.inf * 1j]),
    ],
)
def test_reject_nonfinite_samples(bad_samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        qpsk_demodulate(bad_samples)


def test_reject_empty_demodulation_input() -> None:
    with pytest.raises(ValueError):
        qpsk_demodulate([])


def test_reject_multidimensional_demodulation_input() -> None:
    with pytest.raises(ValueError):
        qpsk_demodulate([[1 + 1j, -1 - 1j]])


def test_reject_text_samples() -> None:
    with pytest.raises(TypeError):
        qpsk_demodulate(["1+1j"])
