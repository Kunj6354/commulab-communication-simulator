"""Interactive I/Q constellation visualization over AWGN."""

from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from commulab.analysis.constellation import (
    ConstellationResult,
    simulate_constellation,
)

MODULATIONS: dict[str, Literal["bpsk", "qpsk", "16qam"]] = {
    "BPSK": "bpsk",
    "QPSK": "qpsk",
    "16-QAM": "16qam",
}

st.set_page_config(
    page_title="CommuLab | Constellation Explorer",
    page_icon="📡",
    layout="wide",
)

st.title("Constellation Explorer")
st.caption("Ideal and noisy digital modulation symbols over AWGN")

st.write(
    "Compare transmitted constellation points with received "
    "symbols after additive white Gaussian noise. "
    "All runs use a reproducible random seed."
)

with st.sidebar:
    st.header("Constellation configuration")

    with st.form("constellation_configuration"):
        modulation = st.selectbox(
            "Modulation",
            tuple(MODULATIONS),
        )

        ebn0_db = st.slider(
            "Eb/N0 (dB)",
            min_value=0.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
        )

        symbol_count = int(
            st.number_input(
                "Number of symbols",
                min_value=100,
                max_value=5_000,
                value=1_000,
                step=100,
            )
        )

        seed = int(
            st.number_input(
                "Random seed",
                min_value=0,
                max_value=2_147_483_647,
                value=42,
                step=1,
            )
        )

        submitted = st.form_submit_button(
            "Generate constellation",
            type="primary",
        )

    st.caption(
        "The visualization is limited to 5,000 symbols "
        "to keep the interactive plot responsive."
    )

if submitted:
    with st.spinner("Generating constellation samples..."):
        try:
            result = simulate_constellation(
                MODULATIONS[modulation],
                ebn0_db,
                symbol_count=symbol_count,
                seed=seed,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop("m05_constellation_result", None)
            st.error(f"Constellation rejected: {exc}")
        else:
            st.session_state["m05_constellation_result"] = (
                modulation,
                result,
            )

saved = st.session_state.get("m05_constellation_result")

if saved is None:
    st.info(
        "Choose your configuration in the sidebar and select "
        "'Generate constellation'."
    )
else:
    selected_modulation, result = saved
    assert isinstance(result, ConstellationResult)

    st.markdown(
        f"**Last generated:** {selected_modulation} | "
        f"Eb/N0 = {result.ebn0_db:g} dB | "
        f"Seed = {result.seed}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Modulation", selected_modulation)
    c2.metric("Symbols", f"{result.symbol_count:,}")
    c3.metric("Bit errors", f"{result.bit_errors:,}")
    c4.metric("Observed BER", f"{result.observed_ber:.6e}")

    ideal = result.ideal_symbols
    received = result.received_symbols
    reference = np.unique(ideal)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Transmitted ideal symbols",
            "Received symbols with AWGN",
        ),
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=reference.real,
            y=reference.imag,
            mode="markers",
            name="Ideal constellation",
            marker={
                "size": 13,
                "color": "#2563eb",
                "symbol": "diamond",
            },
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scattergl(
            x=received.real,
            y=received.imag,
            mode="markers",
            name="Received samples",
            marker={
                "size": 5,
                "color": "#ef4444",
                "opacity": 0.42,
            },
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=reference.real,
            y=reference.imag,
            mode="markers",
            name="Ideal reference",
            marker={
                "size": 12,
                "color": "#1d4ed8",
                "symbol": "x",
            },
        ),
        row=1,
        col=2,
    )

    extent = max(
        1.5,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        (
                            ideal.real,
                            ideal.imag,
                            received.real,
                            received.imag,
                        )
                    )
                )
            )
        )
        * 1.1,
    )

    for column in (1, 2):
        xaxis = "x" if column == 1 else "x2"
        fig.update_xaxes(
            title_text="In-phase (I)",
            range=[-extent, extent],
            zeroline=True,
            row=1,
            col=column,
        )
        fig.update_yaxes(
            title_text="Quadrature (Q)",
            range=[-extent, extent],
            scaleanchor=xaxis,
            scaleratio=1,
            zeroline=True,
            row=1,
            col=column,
        )

    fig.update_layout(
        title=f"{selected_modulation} constellation at {result.ebn0_db:g} dB",
        height=540,
        showlegend=True,
        legend={"orientation": "h", "y": -0.19},
        margin={"l": 30, "r": 30, "t": 75, "b": 80},
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="m05_constellation_plot",
    )

    st.caption(
        "The left plot shows distinct ideal symbol locations; "
        "the right plot shows all received samples and ideal references. "
        "BPSK uses a real-valued AWGN channel, so its Q component is zero."
    )

    st.info(
        "Observed BER is calculated from this symbol sample. "
        "With small sample sizes or high Eb/N0, zero observed errors "
        "do not establish a zero underlying bit-error probability."
    )
