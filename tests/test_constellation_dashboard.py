"""Streamlit integration tests for the Constellation Explorer."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

PAGE = (
    Path(__file__).resolve().parents[1]
    / "dashboard"
    / "pages"
    / "1_Constellation_Explorer.py"
)


def launch() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=20).run()
    assert not app.exception
    return app


def generate(
    app: AppTest,
    modulation: str,
    *,
    symbol_count: int = 1_000,
    ebn0_db: float = 6.0,
    seed: int = 42,
) -> AppTest:
    app.selectbox[0].set_value(modulation)
    app.slider[0].set_value(ebn0_db)
    app.number_input[0].set_value(symbol_count)
    app.number_input[1].set_value(seed)
    app.form_submit_button[0].click().run()
    assert not app.exception
    return app


def test_page_loads_before_generation() -> None:
    app = launch()

    assert app.title[0].value == "Constellation Explorer"
    assert len(app.form_submit_button) == 1
    assert len(app.metric) == 0
    assert "m05_constellation_result" not in app.session_state


@pytest.mark.parametrize(
    ("modulation", "bits_per_symbol"),
    [
        ("BPSK", 1),
        ("QPSK", 2),
        ("16-QAM", 4),
    ],
)
def test_each_modulation_generates_metrics(
    modulation: str,
    bits_per_symbol: int,
) -> None:
    app = generate(launch(), modulation)

    selected, result = app.session_state["m05_constellation_result"]

    assert selected == modulation
    assert result.symbol_count == 1_000
    assert result.bits_per_symbol == bits_per_symbol
    assert result.transmitted_bits == 1_000 * bits_per_symbol
    assert result.ideal_symbols.shape == (1_000,)
    assert result.received_symbols.shape == (1_000,)
    assert len(app.metric) == 4
    assert app.metric[0].value == modulation
    assert app.metric[1].value == "1,000"
    assert app.metric[2].value == f"{result.bit_errors:,}"
    assert app.metric[3].value == f"{result.observed_ber:.6e}"


def test_repeatability_for_same_configuration() -> None:
    app = generate(launch(), "16-QAM", seed=123)
    first = app.session_state["m05_constellation_result"][1]

    app.form_submit_button[0].click().run()

    assert not app.exception
    second = app.session_state["m05_constellation_result"][1]

    assert np.array_equal(first.ideal_symbols, second.ideal_symbols)
    assert np.array_equal(first.received_symbols, second.received_symbols)
    assert first.bit_errors == second.bit_errors


def test_last_result_survives_page_rerun() -> None:
    app = generate(launch(), "QPSK")
    selected, previous = app.session_state["m05_constellation_result"]

    app.run()

    assert not app.exception
    current_selected, current = app.session_state["m05_constellation_result"]

    assert current_selected == selected
    assert np.array_equal(previous.received_symbols, current.received_symbols)
    assert len(app.metric) == 4


def test_high_snr_produces_no_observed_errors() -> None:
    app = generate(
        launch(),
        "BPSK",
        symbol_count=1_000,
        ebn0_db=20.0,
    )

    result = app.session_state["m05_constellation_result"][1]

    assert result.bit_errors == 0
    assert result.observed_ber == 0.0
    assert len(app.metric) == 4
