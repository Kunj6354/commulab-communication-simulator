"""Independent-repetition Monte Carlo BER convergence experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.analysis.ber import simulate_bpsk_ber, theoretical_bpsk_ber

EBN0_DB = 6.0
BIT_COUNTS = (1_000, 10_000, 100_000, 1_000_000)
REPETITIONS = 12
BATCH_SIZE = 50_000
BASE_SEED = 40_000


def main() -> None:
    output = Path(__file__).resolve().parent / "results" / "m04"
    output.mkdir(parents=True, exist_ok=True)

    theory = theoretical_bpsk_ber(EBN0_DB)
    records: list[dict[str, int | float | bool]] = []
    summaries: list[dict[str, int | float]] = []

    print("===== M04-A1 BPSK BER CONVERGENCE =====")
    print(f"Theoretical BER = {theory:.12e}")

    for index, bit_count in enumerate(BIT_COUNTS):
        group: list[dict[str, int | float | bool]] = []

        for repetition in range(REPETITIONS):
            seed = BASE_SEED + index * 100 + repetition

            result = simulate_bpsk_ber(
                EBN0_DB,
                bit_count=bit_count,
                batch_size=BATCH_SIZE,
                seed=seed,
            )

            covered = (
                result.confidence_lower
                <= theory
                <= result.confidence_upper
            )

            row: dict[str, int | float | bool] = {
                "modulation": "bpsk",
                "ebn0_db": EBN0_DB,
                "bit_count": bit_count,
                "repetition": repetition,
                "seed": seed,
                "batch_size": BATCH_SIZE,
                "bit_errors": result.bit_errors,
                "simulated_ber": result.simulated_ber,
                "theoretical_ber": theory,
                "absolute_error": abs(result.simulated_ber - theory),
                "confidence_lower": result.confidence_lower,
                "confidence_upper": result.confidence_upper,
                "confidence_width": (
                    result.confidence_upper - result.confidence_lower
                ),
                "theory_covered": covered,
                "zero_errors": result.bit_errors == 0,
            }

            records.append(row)
            group.append(row)

        estimates = [float(row["simulated_ber"]) for row in group]
        absolute_errors = [float(row["absolute_error"]) for row in group]
        widths = [float(row["confidence_width"]) for row in group]
        coverage_count = sum(bool(row["theory_covered"]) for row in group)
        zero_count = sum(bool(row["zero_errors"]) for row in group)

        summary: dict[str, int | float] = {
            "bit_count": bit_count,
            "repetitions": REPETITIONS,
            "theoretical_ber": theory,
            "mean_simulated_ber": mean(estimates),
            "std_simulated_ber": stdev(estimates),
            "median_absolute_error": median(absolute_errors),
            "mean_absolute_error": mean(absolute_errors),
            "median_confidence_width": median(widths),
            "theory_covered_count": coverage_count,
            "observed_coverage_fraction": coverage_count / REPETITIONS,
            "zero_error_runs": zero_count,
        }
        summaries.append(summary)

        print(
            f"bits={bit_count:>9,d} | "
            f"mean BER={mean(estimates):.8f} | "
            f"median abs error={median(absolute_errors):.6e} | "
            f"median CI width={median(widths):.6e} | "
            f"coverage={coverage_count}/{REPETITIONS} | "
            f"zero-error runs={zero_count}"
        )

    detail_path = output / "bpsk_convergence_runs.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output / "bpsk_convergence_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, ax = plt.subplots(figsize=(9.5, 6))

    for index, count in enumerate(BIT_COUNTS):
        group = [
            row for row in records
            if row["bit_count"] == count
        ]
        ax.scatter(
            [count] * REPETITIONS,
            [row["absolute_error"] for row in group],
            alpha=0.35,
            color="tab:blue",
            label="Individual repetitions" if index == 0 else None,
        )

    ax.plot(
        BIT_COUNTS,
        [row["median_absolute_error"] for row in summaries],
        marker="o",
        color="tab:orange",
        linewidth=2,
        label="Median absolute estimation error",
    )

    ax.plot(
        BIT_COUNTS,
        [(theory * (1.0 - theory) / count) ** 0.5
         for count in BIT_COUNTS],
        linestyle="--",
        color="tab:green",
        label="Theoretical BER standard error",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("BPSK Monte Carlo BER convergence at 6 dB")
    ax.set_xlabel("Transmitted bits per repetition")
    ax.set_ylabel("Absolute BER estimation error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = output / "bpsk_convergence.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    assert len(records) == 48
    assert len(summaries) == 4
    assert sum(int(row["bit_count"]) for row in records) == 13_332_000

    print("\n===== GENERATED ARTIFACTS =====")
    print(f"DETAIL_CSV = {detail_path}")
    print(f"SUMMARY_CSV = {summary_path}")
    print(f"PNG = {plot_path}")
    print("TOTAL_RUNS = 48")
    print("TOTAL_BITS = 13332000")
    print("M04_A1 = PASS")


if __name__ == "__main__":
    main()
