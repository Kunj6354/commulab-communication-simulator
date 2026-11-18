"""Rare-error reliability and zero-error BER experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.analysis.ber import simulate_bpsk_ber, theoretical_bpsk_ber
from commulab.analysis.confidence import wilson_ber_interval

EBN0_DB = 10.0
BIT_COUNTS = (10_000, 100_000, 1_000_000)
REPETITIONS = 24
BATCH_SIZE = 50_000
BASE_SEED = 70_000


def main() -> None:
    output = Path(__file__).resolve().parent / "results" / "m04"
    output.mkdir(parents=True, exist_ok=True)

    theory = theoretical_bpsk_ber(EBN0_DB)

    records: list[dict[str, int | float | bool]] = []
    summaries: list[dict[str, int | float]] = []

    print("===== M04-C1 RARE-ERROR RELIABILITY =====")
    print(f"THEORETICAL_BER = {theory:.12e}")

    for index, bit_count in enumerate(BIT_COUNTS):
        zero_runs = 0
        covered_runs = 0
        total_errors = 0

        for repetition in range(REPETITIONS):
            seed = BASE_SEED + index * 100 + repetition

            result = simulate_bpsk_ber(
                EBN0_DB,
                bit_count=bit_count,
                batch_size=BATCH_SIZE,
                seed=seed,
            )

            is_zero = result.bit_errors == 0
            is_covered = (
                result.confidence_lower
                <= theory
                <= result.confidence_upper
            )

            zero_runs += int(is_zero)
            covered_runs += int(is_covered)
            total_errors += result.bit_errors

            records.append({
                "ebn0_db": EBN0_DB,
                "bit_count": bit_count,
                "repetition": repetition,
                "seed": seed,
                "batch_size": BATCH_SIZE,
                "bit_errors": result.bit_errors,
                "simulated_ber": result.simulated_ber,
                "theoretical_ber": theory,
                "confidence_lower": result.confidence_lower,
                "confidence_upper": result.confidence_upper,
                "zero_errors": is_zero,
                "theory_covered": is_covered,
            })

        expected_zero_probability = math.exp(
            bit_count * math.log1p(-theory)
        )
        zero_ci_upper = wilson_ber_interval(0, bit_count)[1]

        summaries.append({
            "bit_count": bit_count,
            "repetitions": REPETITIONS,
            "theoretical_ber": theory,
            "expected_errors_per_run": bit_count * theory,
            "observed_total_errors": total_errors,
            "zero_error_runs": zero_runs,
            "observed_zero_fraction": zero_runs / REPETITIONS,
            "theoretical_zero_probability": expected_zero_probability,
            "zero_error_wilson_upper": zero_ci_upper,
            "theory_covered_runs": covered_runs,
            "observed_coverage_fraction": covered_runs / REPETITIONS,
        })

        print(
            f"bits={bit_count:>9,d} | "
            f"zero runs={zero_runs:2d}/{REPETITIONS} | "
            f"expected zero probability={expected_zero_probability:.4f} | "
            f"observed coverage={covered_runs:2d}/{REPETITIONS} | "
            f"zero-error CI upper={zero_ci_upper:.6e}"
        )

    detail_path = output / "rare_error_runs.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output / "rare_error_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, ax = plt.subplots(figsize=(9.5, 6))

    ax.plot(
        BIT_COUNTS,
        [row["theoretical_zero_probability"] for row in summaries],
        marker="o",
        linewidth=2,
        label="Theoretical zero-error probability",
    )
    ax.plot(
        BIT_COUNTS,
        [row["observed_zero_fraction"] for row in summaries],
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Observed zero-error fraction (24 runs)",
    )

    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(BIT_COUNTS)
    ax.set_xticklabels([f"{count:,}" for count in BIT_COUNTS])
    ax.set_xlabel("Transmitted bits per simulation")
    ax.set_ylabel("Probability / observed fraction")
    ax.set_title("BPSK zero-error probability at 10 dB Eb/N0")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output / "rare_error_reliability.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    assert len(records) == 72
    assert len(summaries) == 3
    assert sum(int(row["bit_count"]) for row in records) == 26_640_000

    print("\n===== EXPERIMENT ACCEPTANCE =====")
    print("RUNS = 72")
    print("TOTAL_BITS = 26640000")
    print(f"DETAIL_CSV = {detail_path}")
    print(f"SUMMARY_CSV = {summary_path}")
    print(f"PNG = {plot_path}")
    print("M04_C1 = PASS")


if __name__ == "__main__":
    main()
