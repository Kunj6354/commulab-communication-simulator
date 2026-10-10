"""Reproducible quadrature convergence experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from commulab.numerical.q_function import q_function
from commulab.numerical.quadrature import craig_quadrature

METHODS = ("left", "right", "midpoint", "trapezoidal", "simpson")
X_VALUES = (0.0001, 0.01, 0.1, 1.0, 5.0)
INTERVALS = (16, 32, 64, 128, 256, 512, 1024)


def main() -> None:
    output = Path("experiments/results/m01")
    output.mkdir(parents=True, exist_ok=True)

    records = []

    for x in X_VALUES:
        reference = float(q_function(x))

        for method in METHODS:
            for intervals in INTERVALS:
                estimate = float(
                    craig_quadrature(
                        x,
                        method=method,
                        intervals=intervals,
                    )
                )

                absolute_error = abs(estimate - reference)
                relative_error = absolute_error / reference

                records.append({
                    "x": x,
                    "method": method,
                    "intervals": intervals,
                    "reference": reference,
                    "estimate": estimate,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                })

    csv_path = output / "quadrature_convergence.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHODS:
        subset = [
            row for row in records
            if row["x"] == 0.01 and row["method"] == method
        ]

        ax.plot(
            [row["intervals"] for row in subset],
            [max(row["relative_error"], 1e-16) for row in subset],
            marker="o",
            label=method,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of integration intervals")
    ax.set_ylabel("Relative error")
    ax.set_title("Craig Q-function quadrature convergence at x = 0.01")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = output / "quadrature_convergence.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    expected_records = len(METHODS) * len(X_VALUES) * len(INTERVALS)
    assert len(records) == expected_records

    print(f"PASS: {len(records)} experiment records generated")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print("PASS: M01-D experiment completed")


if __name__ == "__main__":
    main()
