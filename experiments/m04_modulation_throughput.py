"""Wall-clock throughput benchmark for CommuLab BER simulators."""

from __future__ import annotations

import csv
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.analysis.ber import simulate_bpsk_ber
from commulab.analysis.modulation_ber import simulate_modulation_ber

MODULATIONS = ("bpsk", "qpsk", "16qam")
BIT_COUNTS = (100_000, 1_000_000)
REPETITIONS = 3
EBN0_DB = 6.0
BASE_SEED = 60_000
BATCH_BITS = 100_000


def run_simulation(
    modulation: str,
    bit_count: int,
    seed: int,
) -> int:
    """Run one BER simulation and return the observed error count."""
    if modulation == "bpsk":
        result = simulate_bpsk_ber(
            EBN0_DB,
            bit_count=bit_count,
            batch_size=BATCH_BITS,
            seed=seed,
        )
    else:
        bits_per_symbol = 2 if modulation == "qpsk" else 4
        result = simulate_modulation_ber(
            modulation,
            EBN0_DB,
            bit_count=bit_count,
            batch_symbols=BATCH_BITS // bits_per_symbol,
            seed=seed,
        )

    return result.bit_errors


def main() -> None:
    output = Path(__file__).resolve().parent / "results" / "m04"
    output.mkdir(parents=True, exist_ok=True)

    print("===== M04-B1 BER THROUGHPUT BENCHMARK =====")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor() or 'not reported'}")

    records: list[dict[str, int | float | str]] = []
    summaries: list[dict[str, int | float | str]] = []

    for mod_index, modulation in enumerate(MODULATIONS):
        # Untimed warm-up for each simulation path.
        run_simulation(modulation, 4_000, BASE_SEED + mod_index)

        for count_index, bit_count in enumerate(BIT_COUNTS):
            group: list[dict[str, int | float | str]] = []

            for repetition in range(REPETITIONS):
                seed = (
                    BASE_SEED
                    + mod_index * 1_000
                    + count_index * 100
                    + repetition
                )

                start = perf_counter()
                errors = run_simulation(modulation, bit_count, seed)
                elapsed = perf_counter() - start

                if elapsed <= 0.0:
                    raise RuntimeError("Nonpositive measured runtime")

                record: dict[str, int | float | str] = {
                    "modulation": modulation,
                    "ebn0_db": EBN0_DB,
                    "bit_count": bit_count,
                    "repetition": repetition,
                    "seed": seed,
                    "batch_bits": BATCH_BITS,
                    "bit_errors": errors,
                    "elapsed_seconds": elapsed,
                    "throughput_bits_per_second": bit_count / elapsed,
                    "python_version": sys.version.split()[0],
                    "platform": platform.platform(),
                }

                records.append(record)
                group.append(record)

            durations = [
                float(row["elapsed_seconds"])
                for row in group
            ]
            throughputs = [
                float(row["throughput_bits_per_second"])
                for row in group
            ]

            summary: dict[str, int | float | str] = {
                "modulation": modulation,
                "bit_count": bit_count,
                "repetitions": REPETITIONS,
                "median_seconds": median(durations),
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "median_throughput_bits_per_second": median(
                    throughputs
                ),
            }
            summaries.append(summary)

            print(
                f"{modulation:5s} | bits={bit_count:>9,d} | "
                f"median runtime={median(durations):.4f} s | "
                f"median throughput={median(throughputs):,.0f} bits/s"
            )

    detail_path = output / "modulation_throughput_runs.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = output / "modulation_throughput_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    large_runs = [
        row for row in summaries
        if row["bit_count"] == 1_000_000
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    labels = [str(row["modulation"]).upper() for row in large_runs]
    throughput_millions = [
        float(row["median_throughput_bits_per_second"]) / 1_000_000
        for row in large_runs
    ]

    bars = ax.bar(labels, throughput_millions)
    ax.bar_label(bars, fmt="%.2f", padding=4)

    ax.set_title("CommuLab Monte Carlo throughput — 1 million bits")
    ax.set_ylabel("Median throughput (million bits/second)")
    ax.set_ylim(0, max(throughput_millions) * 1.20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    plot_path = output / "modulation_throughput.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    assert len(records) == 18
    assert len(summaries) == 6
    assert sum(int(row["bit_count"]) for row in records) == 9_900_000

    print("\n===== BENCHMARK OUTPUT =====")
    print(f"DETAIL_CSV = {detail_path}")
    print(f"SUMMARY_CSV = {summary_path}")
    print(f"PNG = {plot_path}")
    print("TIMED_RUNS = 18")
    print("TIMED_BITS = 9900000")
    print("M04_B1 = PASS")


if __name__ == "__main__":
    main()
