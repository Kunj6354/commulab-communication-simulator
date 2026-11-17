"""Tests for Wilson confidence intervals used in BER experiments."""

import math

import pytest

from commulab.analysis.confidence import wilson_ber_interval


def test_zero_errors_has_positive_upper_bound() -> None:
    lower, upper = wilson_ber_interval(0, 1000)

    assert lower == pytest.approx(0.0, abs=1e-15)
    assert upper == pytest.approx(0.00382676, rel=1e-4)


def test_all_errors_has_lower_bound_below_one() -> None:
    lower, upper = wilson_ber_interval(1000, 1000)

    assert lower < 1.0
    assert upper == pytest.approx(1.0)


def test_interval_contains_observed_ber() -> None:
    for errors, total in [(10, 100), (79, 1000), (0, 1000)]:
        lower, upper = wilson_ber_interval(errors, total)

        assert 0.0 <= lower <= errors / total <= upper <= 1.0


def test_interval_narrows_with_more_samples() -> None:
    small = wilson_ber_interval(10, 100)
    large = wilson_ber_interval(100, 1000)

    assert large[1] - large[0] < small[1] - small[0]


def test_higher_confidence_widens_interval() -> None:
    low_confidence = wilson_ber_interval(50, 1000, confidence=0.90)
    high_confidence = wilson_ber_interval(50, 1000, confidence=0.99)

    assert high_confidence[0] < low_confidence[0]
    assert high_confidence[1] > low_confidence[1]


def test_symmetry_of_complementary_counts() -> None:
    lower, upper = wilson_ber_interval(20, 100)
    complementary_lower, complementary_upper = wilson_ber_interval(
        80, 100
    )

    assert lower == pytest.approx(1.0 - complementary_upper)
    assert upper == pytest.approx(1.0 - complementary_lower)


@pytest.mark.parametrize("total", [0, -1])
def test_invalid_sample_count(total: int) -> None:
    with pytest.raises(ValueError):
        wilson_ber_interval(0, total)


@pytest.mark.parametrize("errors", [-1, 101])
def test_invalid_error_count(errors: int) -> None:
    with pytest.raises(ValueError):
        wilson_ber_interval(errors, 100)


@pytest.mark.parametrize("errors", [True, 1.5, "5"])
def test_invalid_error_types(errors: object) -> None:
    with pytest.raises(TypeError):
        wilson_ber_interval(errors, 100)


@pytest.mark.parametrize("total", [True, 100.5, "100"])
def test_invalid_sample_types(total: object) -> None:
    with pytest.raises(TypeError):
        wilson_ber_interval(5, total)


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.1, math.nan, math.inf],
)
def test_invalid_confidence_values(confidence: float) -> None:
    with pytest.raises(ValueError):
        wilson_ber_interval(5, 100, confidence=confidence)


@pytest.mark.parametrize("confidence", [True, "0.95"])
def test_invalid_confidence_types(confidence: object) -> None:
    with pytest.raises(TypeError):
        wilson_ber_interval(5, 100, confidence=confidence)
