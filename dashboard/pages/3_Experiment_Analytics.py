"""Read-only analytics of CommuLab's published scientific experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parents[2] / "experiments" / "results"

DATASETS = {
    "M01": ("m01/quadrature_convergence.csv", 175),
    "M02": ("m02/bpsk_ber_sweep.csv", 7),
    "M03": ("m03/modulation_ber_comparison.csv", 14),
    "M04 Convergence": ("m04/bpsk_convergence_runs.csv", 48),
    "M04 Throughput": ("m04/modulation_throughput_summary.csv", 6),
    "M04 Rare errors": ("m04/rare_error_summary.csv", 3),
}

LABELS = {
    "bpsk": "BPSK",
    "qpsk": "QPSK",
    "16qam": "16-QAM",
}


@st.cache_data(show_spinner=False)
def read_dataset(name: str) -> list[dict[str, str]]:
    """Load a known, repository-local CSV with basic integrity checks."""
    if name not in DATASETS:
        raise ValueError("Unknown experiment dataset")

    relative_path, expected_rows = DATASETS[name]
    path = RESULTS / relative_path

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {relative_path}")
        rows = list(reader)

    if len(rows) != expected_rows:
        raise ValueError(
            f"{relative_path}: expected {expected_rows} rows, "
            f"found {len(rows)}"
        )
    return rows


def number(row: dict[str, str], key: str) -> float:
    """Convert a numeric CSV value and reject non-finite results."""
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in {key}")
    return value


def ber_trace(
    fig: go.Figure,
    rows: list[dict[str, str]],
    label: str,
) -> None:
    """Show nonzero BER observations and distinct zero-error bounds."""
    positive = [
        row for row in rows if number(row, "bit_errors") > 0
    ]
    zero = [
        row for row in rows if number(row, "bit_errors") == 0
    ]

    if positive:
        fig.add_trace(
            go.Scatter(
                x=[number(row, "ebn0_db") for row in positive],
                y=[number(row, "simulated_ber") for row in positive],
                mode="markers",
                name=f"{label} simulated",
                marker={"size": 9},
            )
        )

    if zero:
        fig.add_trace(
            go.Scatter(
                x=[number(row, "ebn0_db") for row in zero],
                y=[number(row, "confidence_upper") for row in zero],
                mode="markers",
                name=f"{label} zero-error upper bounds",
                marker={
                    "size": 11,
                    "symbol": "triangle-down-open",
                },
            )
        )


def build_chart(name: str, rows: list[dict[str, str]]) -> go.Figure:
    """Construct an interactive graph for one experiment family."""
    fig = go.Figure()

    if name == "M01":
        for method in sorted({row["method"] for row in rows}):
            subset = [
                row for row in rows
                if row["method"] == method and number(row, "x") == 1.0
            ]
            fig.add_trace(
                go.Scatter(
                    x=[number(row, "intervals") for row in subset],
                    y=[
                        max(number(row, "relative_error"), 1e-18)
                        for row in subset
                    ],
                    mode="lines+markers",
                    name=method,
                )
            )
        fig.update_layout(
            title="Q-function quadrature convergence at x = 1",
            xaxis_title="Integration intervals",
            yaxis_title="Relative error (display floor: 1e-18)",
            xaxis_type="log",
            yaxis_type="log",
        )

    elif name in ("M02", "M03"):
        modulations = (
            ["bpsk"] if name == "M02"
            else ["qpsk", "16qam"]
        )

        for modulation in modulations:
            subset = (
                rows if name == "M02"
                else [row for row in rows if row["modulation"] == modulation]
            )
            subset = sorted(
                subset, key=lambda row: number(row, "ebn0_db")
            )
            label = LABELS[modulation]

            fig.add_trace(
                go.Scatter(
                    x=[number(row, "ebn0_db") for row in subset],
                    y=[number(row, "theoretical_ber") for row in subset],
                    mode="lines",
                    name=f"{label} theoretical",
                )
            )
            ber_trace(fig, subset, label)

        fig.update_layout(
            title="Observed and theoretical BER over AWGN",
            xaxis_title="Eb/N0 (dB)",
            yaxis_title="BER / zero-error confidence bound",
            yaxis_type="log",
        )

    elif name == "M04 Convergence":
        for count in sorted({int(row["bit_count"]) for row in rows}):
            subset = [
                row for row in rows if int(row["bit_count"]) == count
            ]
            fig.add_trace(
                go.Box(
                    x=[str(count)] * len(subset),
                    y=[number(row, "absolute_error") for row in subset],
                    name=f"{count:,} bits",
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                )
            )
        fig.update_layout(
            title="Monte Carlo absolute BER error by sample size",
            xaxis_title="Bits per simulation",
            yaxis_title="Absolute BER estimation error",
            yaxis_type="log",
            showlegend=False,
        )

    elif name == "M04 Throughput":
        for modulation in ("bpsk", "qpsk", "16qam"):
            subset = [
                row for row in rows if row["modulation"] == modulation
            ]
            fig.add_trace(
                go.Bar(
                    x=[f"{int(row['bit_count']):,}" for row in subset],
                    y=[
                        number(
                            row, "median_throughput_bits_per_second"
                        ) / 1e6
                        for row in subset
                    ],
                    name=LABELS[modulation],
                )
            )
        fig.update_layout(
            title="Implementation throughput (machine-specific)",
            xaxis_title="Bits per timed simulation",
            yaxis_title="Median million bits per second",
            barmode="group",
        )

    else:
        rows = sorted(rows, key=lambda row: int(row["bit_count"]))
        labels = [f"{int(row['bit_count']):,}" for row in rows]

        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[
                    number(row, "theoretical_zero_probability") * 100
                    for row in rows
                ],
                mode="lines+markers",
                name="Theoretical P(zero errors)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=[
                    number(row, "observed_zero_fraction") * 100
                    for row in rows
                ],
                mode="lines+markers",
                name="Observed zero-error fraction",
            )
        )
        fig.update_layout(
            title="Rare-error study: BPSK at 10 dB",
            xaxis_title="Bits per simulation",
            yaxis_title="Zero-error probability / fraction (%)",
            yaxis_range=[0, 105],
        )

    fig.update_layout(
        height=520,
        hovermode="closest",
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 45, "r": 30, "t": 65, "b": 90},
    )
    return fig


st.set_page_config(
    page_title="CommuLab | Experiment Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Experiment Analytics")
st.caption("Explore reproducible M01-M04 scientific results")

st.write(
    "These charts use saved experimental measurements. "
    "Opening this page does not rerun the simulations."
)

with st.sidebar:
    st.header("Dataset selection")
    selected = st.selectbox("Experiment", tuple(DATASETS))

try:
    records = read_dataset(selected)
    figure = build_chart(selected, records)
except (OSError, KeyError, TypeError, ValueError) as exc:
    st.error(f"Unable to load experiment: {exc}")
else:
    st.metric("Dataset records", f"{len(records):,}")
    st.caption(f"Source: experiments/results/{DATASETS[selected][0]}")

    st.plotly_chart(
        figure,
        width="stretch",
        key="m05_experiment_analytics_plot",
    )

    if selected in ("M02", "M03"):
        st.info(
            "Hollow downward markers indicate simulations with zero "
            "observed errors. Their vertical coordinates are Wilson "
            "confidence upper bounds, not observed BER measurements."
        )
    elif selected == "M04 Throughput":
        st.info(
            "Performance values are specific to the benchmark environment "
            "and software implementation. They are not radio link rates."
        )
    elif selected == "M04 Convergence":
        st.info(
            "Each box contains independently seeded repetitions. "
            "Individual estimation errors need not decrease monotonically."
        )
    elif selected == "M04 Rare errors":
        st.info(
            "Observed zero-error fractions are based on 24 independent "
            "runs per sample size. Zero observed errors do not imply zero BER."
        )
    else:
        st.info(
            "Relative errors smaller than 1e-18 are displayed at the "
            "plotting floor; the underlying CSV values are unchanged."
        )

    with st.expander("Inspect underlying CSV records"):
        st.dataframe(records, width="stretch", hide_index=True)

    csv_path = RESULTS / DATASETS[selected][0]
    st.download_button(
        "Download original CSV",
        data=csv_path.read_bytes(),
        file_name=csv_path.name,
        mime="text/csv",
    )
