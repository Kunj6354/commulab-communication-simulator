"""Streamlit integration tests for the Numerical Q-function Lab."""

import math
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = (
    Path(__file__).resolve().parents[1]
    / "dashboard"
    / "pages"
    / "2_Numerical_Q_Function_Lab.py"
)


def launch() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=20).run()
    assert not app.exception
    return app


def evaluate(
    app: AppTest,
    *,
    x: float = 1.0,
    method: str = "simpson",
    intervals: int = 128,
) -> AppTest:
    app.number_input[0].set_value(x)
    app.selectbox[0].set_value(method)
    app.number_input[1].set_value(intervals)
    app.form_submit_button[0].click().run()

    assert not app.exception
    return app


def test_initial_page_loads() -> None:
    app = launch()

    assert app.title[0].value == "Numerical Q-function Lab"
    assert len(app.form_submit_button) == 1
    assert len(app.metric) == 0
    assert "m05_q_lab_result" not in app.session_state


@pytest.mark.parametrize(
    "method",
    ["left", "right", "midpoint", "trapezoidal", "simpson"],
)
def test_each_integration_method_runs(method: str) -> None:
    app = evaluate(launch(), method=method)
    saved = app.session_state["m05_q_lab_result"]

    assert saved["method"] == method
    assert saved["intervals"] == 128
    assert len(saved["convergence"]) == 40
    assert len(app.metric) == 4
    assert saved["absolute_error"] >= 0.0
    assert math.isclose(
        saved["reference"],
        0.15865525393145707,
        rel_tol=1e-12,
    )


def test_simpson_high_resolution_matches_reference() -> None:
    app = evaluate(launch(), x=1.0, intervals=1024)
    saved = app.session_state["m05_q_lab_result"]

    assert saved["method"] == "simpson"
    assert saved["absolute_error"] < 1e-6


def test_q_at_zero_is_one_half() -> None:
    app = evaluate(launch(), x=0.0)
    saved = app.session_state["m05_q_lab_result"]

    assert saved["reference"] == pytest.approx(0.5)
    assert saved["approximation"] == pytest.approx(0.5, abs=1e-12)


def test_result_persists_without_new_submission() -> None:
    app = evaluate(launch(), x=2.0, method="midpoint")
    previous = app.session_state["m05_q_lab_result"]

    app.run()

    assert not app.exception
    assert app.session_state["m05_q_lab_result"] == previous
    assert len(app.metric) == 4


def test_simpson_invalid_odd_count_rejected() -> None:
    app = evaluate(launch(), x=1.0, method="simpson")
    assert len(app.metric) == 4

    evaluate(app, x=1.0, method="simpson", intervals=99)

    assert "m05_q_lab_result" not in app.session_state
    assert len(app.metric) == 0
    assert len(app.error) == 1
    assert "Numerical evaluation rejected" in app.error[0].body


def test_repeatability_of_numerical_results() -> None:
    app = evaluate(launch(), x=3.0, method="trapezoidal")
    first = app.session_state["m05_q_lab_result"]

    app.form_submit_button[0].click().run()

    assert not app.exception
    assert app.session_state["m05_q_lab_result"] == first
