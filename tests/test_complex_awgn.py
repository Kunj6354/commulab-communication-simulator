"""Tests for normalized complex AWGN and QPSK integration."""

import numpy as np
import pytest

from commulab.analysis.ber import theoretical_bpsk_ber
from commulab.channels.complex_awgn import (
    add_complex_awgn,
    complex_noise_std,
)
from commulab.modulation.bpsk import generate_bits
from commulab.modulation.qpsk import qpsk_demodulate, qpsk_modulate


@pytest.mark.parametrize(
    ("k", "expected_variance"),
    [(2, 0.25), (4, 0.125)],
)
def test_noise_variance_normalization(
    k: int, expected_variance: float
) -> None:
    assert complex_noise_std(
        0.0, bits_per_symbol=k
    ) ** 2 == pytest.approx(expected_variance, rel=1e-12)


def test_noise_variance_changes_with_snr() -> None:
    low = complex_noise_std(0.0, bits_per_symbol=2)
    high = complex_noise_std(10.0, bits_per_symbol=2)

    assert low**2 == pytest.approx(0.25)
    assert high**2 == pytest.approx(0.025)


def test_complex_awgn_is_reproducible() -> None:
    symbols = qpsk_modulate([0, 0, 0, 1, 1, 1, 1, 0])

    first = add_complex_awgn(
        symbols, 4.0, bits_per_symbol=2, seed=123
    )
    second = add_complex_awgn(
        symbols, 4.0, bits_per_symbol=2, seed=123
    )

    np.testing.assert_array_equal(first, second)


def test_different_seeds_change_noise() -> None:
    symbols = np.ones(100, dtype=np.complex128)

    first = add_complex_awgn(
        symbols, 4.0, bits_per_symbol=2, seed=123
    )
    second = add_complex_awgn(
        symbols, 4.0, bits_per_symbol=2, seed=124
    )

    assert np.any(first != second)


def test_noise_statistics_and_iq_independence() -> None:
    symbols = np.ones(200_000, dtype=np.complex128)

    received = add_complex_awgn(
        symbols, 0.0, bits_per_symbol=2, seed=42
    )

    noise = received - symbols
    ni = noise.real
    nq = noise.imag

    assert abs(float(np.mean(ni))) < 0.01
    assert abs(float(np.mean(nq))) < 0.01
    assert float(np.var(ni)) == pytest.approx(0.25, rel=0.03)
    assert float(np.var(nq)) == pytest.approx(0.25, rel=0.03)
    assert abs(float(np.corrcoef(ni, nq)[0, 1])) < 0.02


@pytest.mark.parametrize("ebn0_db", [0.0, 3.0, 6.0])
def test_qpsk_ber_agrees_with_bpsk_theory(ebn0_db: float) -> None:
    bits = generate_bits(200_000, seed=2026)
    symbols = qpsk_modulate(bits)

    received = add_complex_awgn(
        symbols,
        ebn0_db,
        bits_per_symbol=2,
        seed=3030,
    )
    detected = qpsk_demodulate(received)

    empirical = float(np.mean(bits != detected))
    theoretical = theoretical_bpsk_ber(ebn0_db)

    standard_error = np.sqrt(
        theoretical * (1.0 - theoretical) / bits.size
    )

    assert abs(empirical - theoretical) <= (
        5.0 * standard_error + 1.0 / bits.size
    )


def test_channel_output_contract() -> None:
    received = add_complex_awgn(
        np.array([1 + 1j, -1 - 1j]) / np.sqrt(2),
        5.0,
        bits_per_symbol=2,
        seed=7,
    )

    assert received.shape == (2,)
    assert received.dtype == np.complex128


@pytest.mark.parametrize("k", [0, -1])
def test_reject_invalid_k_values(k: int) -> None:
    with pytest.raises(ValueError):
        complex_noise_std(0.0, bits_per_symbol=k)


@pytest.mark.parametrize("k", [True, 1.5, "2"])
def test_reject_invalid_k_types(k: object) -> None:
    with pytest.raises(TypeError):
        complex_noise_std(0.0, bits_per_symbol=k)


@pytest.mark.parametrize(
    "bad_db", [float("nan"), float("inf"), -float("inf")]
)
def test_reject_nonfinite_snr(bad_db: float) -> None:
    with pytest.raises(ValueError):
        complex_noise_std(bad_db, bits_per_symbol=2)


@pytest.mark.parametrize("bad_db", [True, "4", 1 + 1j])
def test_reject_invalid_snr_types(bad_db: object) -> None:
    with pytest.raises(TypeError):
        complex_noise_std(bad_db, bits_per_symbol=2)


@pytest.mark.parametrize("bad_db", [-10000.0, 10000.0])
def test_reject_unrepresentable_snr(bad_db: float) -> None:
    with pytest.raises(ValueError):
        complex_noise_std(bad_db, bits_per_symbol=2)


@pytest.mark.parametrize("bad_seed", [True, 1.5, "42"])
def test_reject_invalid_seed_types(bad_seed: object) -> None:
    with pytest.raises(TypeError):
        add_complex_awgn(
            np.array([1 + 1j]),
            0.0,
            bits_per_symbol=2,
            seed=bad_seed,
        )


def test_reject_real_symbols() -> None:
    with pytest.raises(TypeError):
        add_complex_awgn(
            [1.0, -1.0], 0.0, bits_per_symbol=2, seed=1
        )


def test_reject_nonfinite_symbols() -> None:
    with pytest.raises(ValueError):
        add_complex_awgn(
            np.array([1 + 1j, complex(float("nan"), 0)]),
            0.0,
            bits_per_symbol=2,
            seed=1,
        )


def test_reject_empty_symbols() -> None:
    with pytest.raises(ValueError):
        add_complex_awgn(
            np.array([], dtype=np.complex128),
            0.0,
            bits_per_symbol=2,
            seed=1,
        )


def test_reject_multidimensional_symbols() -> None:
    with pytest.raises(ValueError):
        add_complex_awgn(
            np.array([[1 + 1j]]),
            0.0,
            bits_per_symbol=2,
            seed=1,
        )
