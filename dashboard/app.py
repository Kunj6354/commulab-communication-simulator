"""CommuLab interactive digital communication laboratory."""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from commulab.analysis.ber import BerResult, simulate_bpsk_ber, theoretical_bpsk_ber
from commulab.analysis.modulation_ber import ModulationBerResult, simulate_modulation_ber
from commulab.analysis.modulation_theory import theoretical_qam16_ber, theoretical_qpsk_ber

THEORETICAL_BER = {
    "BPSK": theoretical_bpsk_ber,
    "QPSK": theoretical_qpsk_ber,
    "16-QAM": theoretical_qam16_ber,
}

st.set_page_config(page_title="CommuLab | Communication Simulator", page_icon="📡", layout="wide")
st.title("CommuLab")
st.caption("Digital Communication & BER Simulation Laboratory")

with st.sidebar:
    st.header("Simulation configuration")
    with st.form("ber_configuration"):
        modulation = st.selectbox("Modulation", tuple(THEORETICAL_BER))
        ebn0_db = st.slider("Eb/N0 (dB)", 0.0, 12.0, 6.0, 0.5)
        bit_count = st.number_input(
            "Transmitted bits",
            min_value=1_000,
            max_value=2_000_000,
            value=100_000,
            step=1_000,
            help="Use an even count for QPSK and a multiple of four for 16-QAM.",
        )
        seed = st.number_input("Random seed", 0, 2_147_483_647, 42, 1)
        submitted = st.form_submit_button("Run BER simulation", type="primary")

st.subheader("BER Explorer")

if submitted:
    simulation: BerResult | ModulationBerResult
    with st.spinner("Running Monte Carlo BER simulation..."):
        try:
            if modulation == "BPSK":
                simulation = simulate_bpsk_ber(
                    ebn0_db, bit_count=bit_count, seed=seed, batch_size=50_000
                )
            else:
                key: Literal["qpsk", "16qam"] = "qpsk" if modulation == "QPSK" else "16qam"
                simulation = simulate_modulation_ber(
                    key, ebn0_db, bit_count=bit_count, seed=seed, batch_symbols=25_000
                )
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop("m05_ber_result", None)
            st.error(f"Simulation rejected: {exc}")
        else:
            st.session_state["m05_ber_result"] = (modulation, simulation)

saved = st.session_state.get("m05_ber_result")
if saved is None:
    st.info("Set the parameters in the sidebar and select ‘Run BER simulation’ to begin.")
else:
    selected_modulation, result = saved
    st.markdown(
        f"Last completed simulation · {selected_modulation} · "
        f"Eb/N0 = {result.ebn0_db:g} dB · Seed = {result.seed}"
    )
    theory, observed, errors, bits = st.columns(4)
    theory.metric("Theoretical BER", f"{result.theoretical_ber:.6e}")
    observed.metric("Simulated BER", f"{result.simulated_ber:.6e}")
    errors.metric("Bit errors", f"{result.bit_errors:,}")
    bits.metric("Transmitted bits", f"{result.bits_transmitted:,}")

    st.subheader("BER versus Eb/N0")
    snr_values = np.linspace(0.0, 12.0, 61)
    theoretical_ber = THEORETICAL_BER[selected_modulation]
    theory_values = [theoretical_ber(float(snr)) for snr in snr_values]

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    try:
        ax.semilogy(snr_values, theory_values, label=f"{selected_modulation} theoretical BER")
        if result.bit_errors:
            ax.semilogy(
                [result.ebn0_db], [result.simulated_ber], "o", label="Monte Carlo observation"
            )
        else:
            ax.semilogy(
                [result.ebn0_db],
                [result.confidence_upper],
                "v",
                markerfacecolor="none",
                label="Zero errors · Wilson upper bound",
            )
        ax.set(
            xlabel="Eb/N0 (dB)", ylabel="Bit-error rate", title=f"{selected_modulation} over AWGN"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    finally:
        plt.close(fig)

    st.subheader("95% Wilson confidence interval")
    st.write(f"{result.confidence_lower:.6e} to {result.confidence_upper:.6e}")
