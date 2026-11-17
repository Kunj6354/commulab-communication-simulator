"""Reproducible BPSK BER sweep over an AWGN channel."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.analysis.ber import simulate_bpsk_ber

EBN0_DB_VALUES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
BIT_COUNT = 1_000_000
BATCH_SIZE = 100_000
BASE_SEED = 2026


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output = root / "experiments" / "results" / "m02"
    output.mkdir(parents=True, exist_ok=True)

    records = []

    print("===== M02 BPSK BER SWEEP =====")

    for index, ebn0_db in enumerate(EBN0_DB_VALUES):
        result = simulate_bpsk_ber(
            ebn0_db,
            bit_count=BIT_COUNT,
            batch_size=BATCH_SIZE,
            seed=BASE_SEED + index,
        )

        record = {
            "ebn0_db": result.ebn0_db,
            "bits_transmitted": result.bits_transmitted,
            "bit_errors": result.bit_errors,
            "simulated_ber": result.simulated_ber,
            "theoretical_ber": result.theoretical_ber,
            "confidence_lower": result.confidence_lower,
            "confidence_upper": result.confidence_upper,
            "seed": result.seed,
            "batch_size": BATCH_SIZE,
        }

        records.append(record)

        print(
            f"Eb/N0={ebn0_db:4.1f} dB | "
            f"errors={result.bit_errors:7d} | "
            f"simulated={result.simulated_ber:.6e} | "
            f"theoretical={result.theoretical_ber:.6e} | "
            f"CI=[{result.confidence_lower:.6e}, "
            f"{result.confidence_upper:.6e}]"
        )

    csv_path = output / "bpsk_ber_sweep.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.semilogy(
        [row["ebn0_db"] for row in records],
        [row["theoretical_ber"] for row in records],
        "-",
        linewidth=2,
        label="Theoretical BPSK BER",
    )

    observed = [
        row for row in records if row["bit_errors"] > 0
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
            fmt="o",
            capsize=4,
            label="Simulated BER (95% Wilson CI)",
        )

    zero_errors = [
        row for row in records if row["bit_errors"] == 0
    ]

    if zero_errors:
        ax.semilogy(
            [row["ebn0_db"] for row in zero_errors],
            [row["confidence_upper"] for row in zero_errors],
            "v",
            markersize=8,
            fillstyle="none",
            label="Zero errors: 95% upper bound",
        )

    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title("BPSK over AWGN — theoretical and Monte Carlo BER")
    ax.set_xticks(EBN0_DB_VALUES)
    ax.set_ylim(1e-10, 0.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = output / "bpsk_ber_sweep.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    assert len(records) == len(EBN0_DB_VALUES)
    assert sum(row["bits_transmitted"] for row in records) == (
        BIT_COUNT * len(EBN0_DB_VALUES)
    )

    print("\n===== OUTPUT =====")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"SNR points: {len(records)}")
    print(f"Total transmitted bits: {BIT_COUNT * len(records):,}")
    print("PASS: M02-E2 experiment completed")


if __name__ == "__main__":
    main()
