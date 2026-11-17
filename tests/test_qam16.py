"""Tests for Gray-coded, normalized 16-QAM modulation."""

from itertools import pairwise, product

import numpy as np
import pytest

from commulab.modulation.bpsk import generate_bits
from commulab.modulation.qam16 import (
    INNER_OUTER_BOUNDARY,
    qam16_demodulate,
    qam16_modulate,
)

AXIS_LEVELS = (
    ((0, 0), 3),
    ((0, 1), 1),
    ((1, 1), -1),
    ((1, 0), -3),
)


@pytest.mark.parametrize(
    ("label", "amplitude"),
    AXIS_LEVELS,
)
def test_i_axis_mapping(
    label: tuple[int, int], amplitude: int
) -> None:
    symbol = qam16_modulate([*label, 0, 0])[0]
    assert symbol.real == pytest.approx(amplitude / np.sqrt(10))


@pytest.mark.parametrize(
    ("label", "amplitude"),
    AXIS_LEVELS,
)
def test_q_axis_mapping(
    label: tuple[int, int], amplitude: int
) -> None:
    symbol = qam16_modulate([0, 0, *label])[0]
    assert symbol.imag == pytest.approx(amplitude / np.sqrt(10))


def test_all_sixteen_constellation_points() -> None:
    labels = list(product((0, 1), repeat=4))
    bits = [bit for label in labels for bit in label]

    symbols = qam16_modulate(bits)

    assert symbols.shape == (16,)
    assert symbols.dtype == np.complex128
    assert len(np.unique(symbols)) == 16

    np.testing.assert_array_equal(
        qam16_demodulate(symbols),
        np.array(bits, dtype=np.uint8),
    )


def test_exact_average_constellation_energy() -> None:
    labels = list(product((0, 1), repeat=4))
    bits = [bit for label in labels for bit in label]
    symbols = qam16_modulate(bits)

    assert float(np.mean(np.abs(symbols) ** 2)) == pytest.approx(
        1.0, abs=1e-14
    )

    expected_energies = sorted([0.2] * 4 + [1.0] * 8 + [1.8] * 4)
    np.testing.assert_allclose(
        sorted(np.abs(symbols) ** 2),
        expected_energies,
        atol=1e-14,
    )


def test_gray_adjacency_on_both_axes() -> None:
    ordered_labels = [label for label, _ in reversed(AXIS_LEVELS)]

    for first, second in pairwise(ordered_labels):
        assert sum(a != b for a, b in zip(first, second)) == 1


def test_noiseless_round_trip() -> None:
    bits = generate_bits(40_000, seed=42)
    recovered = qam16_demodulate(qam16_modulate(bits))
    np.testing.assert_array_equal(recovered, bits)


def test_four_bits_per_symbol() -> None:
    bits = generate_bits(400, seed=2026)
    symbols = qam16_modulate(bits)

    assert symbols.shape == (100,)
    assert qam16_demodulate(symbols).shape == (400,)


def test_demodulator_threshold_conventions() -> None:
    boundary = INNER_OUTER_BOUNDARY

    samples = np.array([
        0 + 0j,
        complex(boundary, boundary),
        complex(-boundary, -boundary),
    ])

    np.testing.assert_array_equal(
        qam16_demodulate(samples),
        np.array(
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize("bit_count", [1, 2, 3, 5, 7])
def test_reject_invalid_bit_count(bit_count: int) -> None:
    with pytest.raises(ValueError):
        qam16_modulate([0] * bit_count)


@pytest.mark.parametrize(
    "bad_bits",
    [
        [0, 1, 2, 0],
        [0, -1, 1, 0],
    ],
)
def test_reject_nonbinary_bits(bad_bits: list[int]) -> None:
    with pytest.raises(ValueError):
        qam16_modulate(bad_bits)


@pytest.mark.parametrize(
    "bad_bits",
    [
        [0.0, 1.0, 0.0, 1.0],
        ["0", "1", "0", "1"],
        [0j, 1 + 0j, 0j, 1 + 0j],
    ],
)
def test_reject_invalid_bit_dtypes(bad_bits: list[object]) -> None:
    with pytest.raises(TypeError):
        qam16_modulate(bad_bits)


def test_reject_empty_modulation_input() -> None:
    with pytest.raises(ValueError):
        qam16_modulate([])


def test_reject_multidimensional_modulation_input() -> None:
    with pytest.raises(ValueError):
        qam16_modulate([[0, 0, 1, 1]])


@pytest.mark.parametrize(
    "bad_samples",
    [
        np.array([np.nan + 1j]),
        np.array([complex(1, float("inf"))]),
    ],
)
def test_reject_nonfinite_samples(bad_samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        qam16_demodulate(bad_samples)


def test_reject_empty_demodulation_input() -> None:
    with pytest.raises(ValueError):
        qam16_demodulate([])


def test_reject_multidimensional_demodulation_input() -> None:
    with pytest.raises(ValueError):
        qam16_demodulate([[1 + 1j, -1 - 1j]])


def test_reject_text_samples() -> None:
    with pytest.raises(TypeError):
        qam16_demodulate(["1+1j"])
