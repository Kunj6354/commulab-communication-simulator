"""Interactive Gaussian Q-function numerical integration laboratory."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from commulab.numerical.q_function import log_q_function, q_function
from commulab.numerical.quadrature import craig_quadrature

METHODS = ("left", "right", "midpoint", "trapezoidal", "simpson")
INTERVAL_GRID = (8, 16, 32, 64, 128, 256, 512, 1024)


def calculate(x: float, method: str, intervals: int) -> dict[str, object]:
    """Compute an M01 reference comparison and convergence dataset."""
    if not math.isfinite(x) or x < 0:
        raise ValueError("Craig's quadrature requires finite x >= 0")
    if method not in METHODS:
        raise ValueError("Unknown quadrature method")
    if intervals < 1:
        raise ValueError("Intervals must be positive")
    if method == "simpson" and intervals % 2:
        raise ValueError("Simpson's rule requires even intervals")

    reference = float(q_function(x))
    approximation = float(
        craig_quadrature(x, method=method, intervals=intervals)
    )
    log_reference = float(log_q_function(x))
    absolute_error = abs(approximation - reference)
    relative_error = absolute_error / reference

    convergence = []
    for candidate in METHODS:
        for count in INTERVAL_GRID:
            estimate = float(
                craig_quadrature(x, method=candidate, intervals=count)
            )
            convergence.append(
                {
                    "method": candidate,
                    "intervals": count,
                    "estimate": estimate,
                    "absolute_error": abs(estimate - reference),
                }
            )

    return {
        "x": x,
        "method": method,
        "intervals": intervals,
        "reference": reference,
        "approximation": approximation,
        "log_reference": log_reference,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "convergence": convergence,
    }


st.set_page_config(
    page_title="CommuLab | Numerical Q-function Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Numerical Q-function Lab")
st.caption("Craig's integral, quadrature accuracy, and numerical convergence")

st.markdown(
    "Compare the reference Gaussian Q-function against five numerical "
    "integration methods from M01. The integration domain is restricted "
    "to nonnegative x."
)

with st.sidebar:
    st.header("Numerical integration")

    with st.form("q_function_configuration"):
        x = st.number_input(
            "Q-function argument x",
            min_value=0.0,
            max_value=8.0,
            value=1.0,
            step=0.1,
            format="%.2f",
        )

        method = st.selectbox(
            "Quadrature method",
            METHODS,
            index=4,
        )

        intervals = int(
            st.number_input(
                "Integration intervals",
                min_value=2,
                max_value=2048,
                value=128,
                step=2,
            )
        )

        submitted = st.form_submit_button(
            "Evaluate Q-function",
            type="primary",
        )

    st.caption(
        "Interval count uses even increments, so every selectable "
        "configuration is also valid for Simpson's rule."
    )

if submitted:
    with st.spinner("Evaluating numerical integration..."):
        try:
            result = calculate(float(x), method, intervals)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop("m05_q_lab_result", None)
            st.error(f"Numerical evaluation rejected: {exc}")
        else:
            st.session_state["m05_q_lab_result"] = result

saved = st.session_state.get("m05_q_lab_result")

if saved is None:
    st.info(
        "Configure the argument and integration method, "
        "then select 'Evaluate Q-function'."
    )
else:
    st.markdown(
        f"**Last evaluation:** x = {saved['x']:g} | "
        f"Method = {saved['method']} | "
        f"Intervals = {saved['intervals']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reference Q(x)", f"{saved['reference']:.8e}")
    col2.metric("Quadrature Q(x)", f"{saved['approximation']:.8e}")
    col3.metric("Absolute error", f"{saved['absolute_error']:.3e}")
    col4.metric("Relative error", f"{saved['relative_error']:.3e}")

    st.caption(f"Reference log(Q(x)) = {saved['log_reference']:.8f}")

    st.subheader("Quadrature convergence")

    fig = go.Figure()

    for candidate in METHODS:
        subset = [
            row
            for row in saved["convergence"]
            if row["method"] == candidate
        ]

        fig.add_trace(
            go.Scatter(
                x=[row["intervals"] for row in subset],
                y=[max(row["absolute_error"], 1e-18) for row in subset],
                mode="lines+markers",
                name=candidate,
            )
        )

    fig.update_layout(
        title=f"Absolute integration error at x = {saved['x']:g}",
        xaxis_title="Number of integration intervals",
        yaxis_title="Absolute error",
        xaxis_type="log",
        yaxis_type="log",
        height=520,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        key="m05_q_convergence_plot",
    )

    st.caption(
        "Errors below 1e-18 are displayed at 1e-18 for the logarithmic "
        "plot. This plotting floor is not the measured numerical error."
    )

    st.subheader("Selected evaluation")

    st.write(
        {
            "x": saved["x"],
            "method": saved["method"],
            "intervals": saved["intervals"],
            "reference": saved["reference"],
            "estimate": saved["approximation"],
            "absolute_error": saved["absolute_error"],
            "relative_error": saved["relative_error"],
        }
    )

    st.info(
        "A smaller numerical error means the selected quadrature "
        "configuration more closely matches the reference Q-function. "
        "The convergence chart compares all five M01 methods."
    )

st.divider()
st.caption("M05-C | Numerical Q-function Lab | M01 validated numerical engine")
