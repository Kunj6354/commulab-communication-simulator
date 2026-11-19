"""Streamlit integration tests for M01-M04 experiment analytics."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGE = (
    Path(__file__).resolve().parents[1]
    / "dashboard"
    / "pages"
    / "3_Experiment_Analytics.py"
)

EXPECTED = {
    "M01": 175,
    "M02": 7,
    "M03": 14,
    "M04 Convergence": 48,
    "M04 Throughput": 6,
    "M04 Rare errors": 3,
}


def launch() -> AppTest:
    app = AppTest.from_file(str(PAGE), default_timeout=20).run()
    assert not app.exception
    return app


def test_analytics_page_loads() -> None:
    app = launch()

    assert app.title[0].value == "Experiment Analytics"
    assert len(app.selectbox) == 1
    assert app.selectbox[0].value == "M01"
    assert app.metric[0].value == "175"


@pytest.mark.parametrize(("name", "count"), EXPECTED.items())
def test_each_dataset_loads_without_exception(
    name: str,
    count: int,
) -> None:
    app = launch()
    app.selectbox[0].set_value(name).run()

    assert not app.exception
    assert len(app.error) == 0
    assert app.metric[0].value == f"{count:,}"


def test_selection_persists_after_rerun() -> None:
    app = launch()
    app.selectbox[0].set_value("M03").run()
    app.run()

    assert not app.exception
    assert app.selectbox[0].value == "M03"
    assert app.metric[0].value == "14"


def test_analytics_does_not_modify_source_csv() -> None:
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "m02"
        / "bpsk_ber_sweep.csv"
    )

    original = csv_path.read_bytes()

    app = launch()
    app.selectbox[0].set_value("M02").run()

    assert not app.exception
    assert csv_path.read_bytes() == original
