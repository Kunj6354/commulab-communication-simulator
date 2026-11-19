# M05 ? Interactive Communication Simulator Dashboard

## Status

Implemented and validated locally on
`feature/m05-interactive-dashboard`.

GitHub integration pending.

## Objective

Provide a user-facing interface for the validated CommuLab
numerical and digital communication algorithms.

The dashboard uses Streamlit and Plotly. Existing numerical
models and Monte Carlo BER implementations remain reusable.

## Installation

Requires Python 3.12 or later.

For the dashboard:

    python -m pip install -e ".[dashboard]"

For dashboard development and the full test suite:

    python -m pip install -e ".[dev,dashboard]"

## Launch

From the repository root on Windows:

    .\.venv\Scripts\python.exe -m streamlit run dashboard/app.py

Open the local URL reported by Streamlit, usually
http://localhost:8501.

## Page 1 ? BER Explorer

Supports:

- Coherent BPSK over real AWGN
- Gray-coded QPSK over complex AWGN
- Gray-coded 16-QAM over complex AWGN
- Configurable Eb/N0, bit count and deterministic seed
- Theoretical and simulated BER comparison
- Observed bit-error counts
- Two-sided 95% Wilson confidence intervals
- Zero-error observations indicated using their positive
  Wilson upper confidence bound

Results persist across ordinary interface reruns. Rejected
simulation submissions clear stale displayed results.

The BER Explorer uses the existing bounded-memory simulator.

## Page 2 ? Constellation Explorer

Displays ideal symbol positions alongside noisy received
samples for BPSK, QPSK and 16-QAM.

Users can configure Eb/N0, symbol count and random seed.
The plot uses the existing symbol mapping and demodulation
implementations.

BPSK uses real-valued AWGN; QPSK and 16-QAM use complex
AWGN. Results are reproducible for the same configuration.

The visualized symbol count is limited to 5,000.

## Page 3 ? Numerical Q-function Lab

Compares the Gaussian Q-function reference implementation
with numerical integration of Craig's integral.

Methods:

- Left rectangle
- Right rectangle
- Midpoint
- Trapezoidal
- Simpson

The page reports reference and estimated values, absolute
error, relative error and convergence across interval counts.

Craig's integral requires nonnegative x. Simpson's method
requires an even interval count.

## Page 4 ? Experiment Analytics

Reads existing M01?M04 CSV datasets without rerunning
the scientific experiments.

Views include:

- M01 quadrature convergence
- M02 BPSK BER sweep
- M03 QPSK and 16-QAM BER comparison
- M04 Monte Carlo BER convergence
- M04 implementation throughput
- M04 rare-error reliability

The analytics view provides interactive charts, underlying
CSV records and downloads of the source CSV files.

For BER plots, a zero-error Wilson upper-bound marker is
distinct from nonzero observed BER measurements.

## Reproducibility

The Monte Carlo and constellation interfaces accept explicit
random seeds. Repeating an identical configuration reproduces
the corresponding simulation result.

The numerical integration page is deterministic.

Saved benchmark timing results remain machine-specific; they
should not be interpreted as physical communication data rates.

## Verification

Local M05-D acceptance:

- 353 automated tests passed
- Ruff passed
- mypy passed for 17 source files
- pip check passed

Dashboard-specific tests exercise Streamlit widget submission,
result persistence, invalid configurations and reproducibility.

## Limitations

The communication models assume ideal, uncoded AWGN channels.
No RF hardware, synchronization impairments, fading or channel
coding is modeled.

A measured BER of zero does not prove that the underlying BER
is zero. Wilson confidence bounds quantify sampling uncertainty.

The dashboard is intended for engineering exploration and
portfolio demonstration, not certified telecommunications
performance assessment.
