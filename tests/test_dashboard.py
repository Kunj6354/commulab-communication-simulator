"""Automated integration tests for the CommuLab BER Explorer."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def launch() -> AppTest:
    """Start the dashboard with a fresh Streamlit session."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    return app


def submit(
    app: AppTest,
    modulation: str,
    *,
    bits: int = 4_000,
    ebn0_db: float = 6.0,
    seed: int = 42,
) -> AppTest:
    """Submit the actual sidebar form."""
    app.selectbox[0].set_value(modulation)
    app.slider[0].set_value(ebn0_db)
    app.number_input[0].set_value(bits)
    app.number_input[1].set_value(seed)

    app.form_submit_button[0].click().run()
    assert not app.exception
    return app


def test_initial_dashboard_loads_without_simulation() -> None:
    app = launch()

    assert app.title[0].value == "CommuLab"
    assert len(app.selectbox) == 1
    assert len(app.slider) == 1
    assert len(app.number_input) == 2
    assert len(app.form_submit_button) == 1
    assert len(app.metric) == 0
    assert "m05_ber_result" not in app.session_state


@pytest.mark.parametrize(
    ("modulation", "expected_type"),
    [
        ("BPSK", "BerResult"),
        ("QPSK", "ModulationBerResult"),
        ("16-QAM", "ModulationBerResult"),
    ],
)
def test_each_modulation_displays_simulation_metrics(
    modulation: str,
    expected_type: str,
) -> None:
    app = submit(launch(), modulation)
    saved_modulation, result = app.session_state["m05_ber_result"]

    assert saved_modulation == modulation
    assert type(result).__name__ == expected_type
    assert result.bits_transmitted == 4_000
    assert result.seed == 42
    assert 0 <= result.bit_errors <= 4_000
    assert result.confidence_lower <= result.simulated_ber
    assert result.simulated_ber <= result.confidence_upper

    assert len(app.metric) == 4
    assert app.metric[0].value == f"{result.theoretical_ber:.6e}"
    assert app.metric[1].value == f"{result.simulated_ber:.6e}"
    assert app.metric[2].value == f"{result.bit_errors:,}"
    assert app.metric[3].value == "4,000"


def test_same_configuration_is_reproducible() -> None:
    app = submit(launch(), "16-QAM", bits=8_000, seed=123)
    first = app.session_state["m05_ber_result"][1]

    app.form_submit_button[0].click().run()

    assert not app.exception
    second = app.session_state["m05_ber_result"][1]
    assert first == second


def test_result_persists_across_rerun_without_submission() -> None:
    app = submit(launch(), "BPSK")
    previous = app.session_state["m05_ber_result"]

    app.run()

    assert not app.exception
    assert app.session_state["m05_ber_result"] == previous
    assert len(app.metric) == 4


@pytest.mark.parametrize(
    ("modulation", "bits"),
    [
        ("QPSK", 1_001),
        ("16-QAM", 1_002),
    ],
)
def test_invalid_symbol_alignment_clears_stale_result(
    modulation: str,
    bits: int,
) -> None:
    app = submit(launch(), "BPSK")
    assert len(app.metric) == 4

    submit(app, modulation, bits=bits)

    assert "m05_ber_result" not in app.session_state
    assert len(app.metric) == 0
    assert len(app.error) == 1
    assert "Simulation rejected" in app.error[0].body


def test_zero_error_result_has_positive_wilson_upper_bound() -> None:
    app = submit(
        launch(),
        "BPSK",
        bits=1_000,
        ebn0_db=12.0,
        seed=42,
    )

    result = app.session_state["m05_ber_result"][1]

    assert result.bit_errors == 0
    assert result.simulated_ber == 0.0
    assert result.confidence_lower == 0.0
    assert result.confidence_upper > 0.0
    assert len(app.metric) == 4
    assert not app.exception
