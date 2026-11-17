"""Reproducible comparison of QPSK and 16-QAM BER over AWGN."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.analysis.modulation_ber import simulate_modulation_ber

MODULATIONS = ("qpsk", "16qam")
EBN0_VALUES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
BIT_COUNT = 500_000
BATCH_SYMBOLS = 25_000
BASE_SEED = 2026


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output = root / "experiments" / "results" / "m03"
    output.mkdir(parents=True, exist_ok=True)

    records = []

    print("===== M03 QPSK / 16-QAM BER COMPARISON =====")

    for mod_index, modulation in enumerate(MODULATIONS):
        for snr_index, ebn0_db in enumerate(EBN0_VALUES):
            seed = BASE_SEED + mod_index * 100 + snr_index

            result = simulate_modulation_ber(
                modulation,
                ebn0_db,
                bit_count=BIT_COUNT,
                batch_symbols=BATCH_SYMBOLS,
                seed=seed,
            )

            records.append({
                "modulation": result.modulation,
                "ebn0_db": result.ebn0_db,
                "bits_per_symbol": result.bits_per_symbol,
                "bits_transmitted": result.bits_transmitted,
                "symbols_transmitted": result.symbols_transmitted,
                "bit_errors": result.bit_errors,
                "simulated_ber": result.simulated_ber,
                "theoretical_ber": result.theoretical_ber,
                "confidence_lower": result.confidence_lower,
                "confidence_upper": result.confidence_upper,
                "seed": result.seed,
                "batch_symbols": BATCH_SYMBOLS,
            })

            print(
                f"{modulation:5s} | "
                f"Eb/N0={ebn0_db:4.1f} dB | "
                f"errors={result.bit_errors:6d} | "
                f"simulated={result.simulated_ber:.6e} | "
                f"theoretical={result.theoretical_ber:.6e}"
            )

    csv_path = output / "modulation_ber_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for modulation, marker in (
        ("qpsk", "o"),
        ("16qam", "s"),
    ):
        subset = [
            row for row in records
            if row["modulation"] == modulation
        ]

        ax.semilogy(
            [row["ebn0_db"] for row in subset],
            [row["theoretical_ber"] for row in subset],
            linewidth=2,
            label=f"{modulation.upper()} theory",
        )

        observed = [
            row for row in subset
            if row["bit_errors"] > 0
        ]

        if observed:
            ax.errorbar(
                [row["ebn0_db"] for row in observed],
                [row["simulated_ber"] for row in observed],
                yerr=[
                    [
                        row["simulated_ber"] - row["confidence_lower"]
                        for row in observed
                    ],
                    [
                        row["confidence_upper"] - row["simulated_ber"]
                        for row in observed
                    ],
                ],
                fmt=marker,
                linestyle="none",
                capsize=3,
                label=f"{modulation.upper()} Monte Carlo (95% CI)",
            )

        zero_error_rows = [
            row for row in subset
            if row["bit_errors"] == 0
        ]

        if zero_error_rows:
            ax.semilogy(
                [row["ebn0_db"] for row in zero_error_rows],
                [row["confidence_upper"] for row in zero_error_rows],
                "v",
                markersize=8,
                fillstyle="none",
                label=f"{modulation.upper()} zero errors: CI upper bound",
            )

    ax.set_title("QPSK and Gray-coded 16-QAM over AWGN")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_xticks(EBN0_VALUES)
    ax.set_ylim(1e-9, 0.3)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    png_path = output / "modulation_ber_comparison.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    assert len(records) == 14
    assert sum(row["bits_transmitted"] for row in records) == 7_000_000

    print("\n===== EXPERIMENT OUTPUT =====")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"Configurations: {len(records)}")
    print("Total transmitted bits: 7,000,000")
    print("PASS: M03-E1 experiment completed")


if __name__ == "__main__":
    main()
