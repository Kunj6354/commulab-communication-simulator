"""Analytical BER verification for QPSK and Gray-coded 16-QAM."""

import math

import pytest

from commulab.analysis.ber import theoretical_bpsk_ber
from commulab.analysis.modulation_theory import (
    theoretical_qam16_ber,
    theoretical_qpsk_ber,
)


@pytest.mark.parametrize("db", [-3.0, 0.0, 3.0, 6.0, 10.0])
def test_qpsk_matches_bpsk_theory(db: float) -> None:
    assert theoretical_qpsk_ber(db) == pytest.approx(
        theoretical_bpsk_ber(db), rel=1e-14
    )


@pytest.mark.parametrize(
    ("db", "expected"),
    [
        (0.0, 0.14098163506684164),
        (4.0, 0.058623737283404406),
        (8.0, 0.009247213741474409),
        (10.0, 0.0017541506178927271),
    ],
)
def test_qam16_reference_values(db: float, expected: float) -> None:
    assert theoretical_qam16_ber(db) == pytest.approx(
        expected, rel=1e-11
    )


@pytest.mark.parametrize("db", [0.0, 3.0, 6.0, 10.0])
def test_qam16_ber_exceeds_qpsk_at_equal_ebn0(db: float) -> None:
    assert theoretical_qam16_ber(db) > theoretical_qpsk_ber(db)


def test_qam16_ber_decreases_with_snr() -> None:
    results = [theoretical_qam16_ber(db) for db in (0, 3, 6, 9)]

    assert results == sorted(results, reverse=True)


def test_theoretical_probabilities_are_valid() -> None:
    for db in (-5.0, 0.0, 5.0, 10.0):
        assert 0.0 <= theoretical_qpsk_ber(db) <= 0.5
        assert 0.0 <= theoretical_qam16_ber(db) <= 0.5


@pytest.mark.parametrize("db", [math.nan, math.inf, -math.inf])
def test_reject_nonfinite_snr(db: float) -> None:
    with pytest.raises(ValueError):
        theoretical_qam16_ber(db)


@pytest.mark.parametrize("db", [True, "4", 1 + 1j])
def test_reject_invalid_snr_types(db: object) -> None:
    with pytest.raises(TypeError):
        theoretical_qam16_ber(db)
