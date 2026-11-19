# CommuLab

A Python-based digital communication systems and numerical BER simulation laboratory.

## Project origin

This project is a redevelopment and extension of an unfinished 2023 academic
Communication Systems project investigating numerical approximations of the
Gaussian Q-function.

## Planned capabilities

- Numerical integration and Gaussian Q-function approximation
- Convergence, accuracy, and performance benchmarking
- BPSK, QPSK, and 16-QAM simulation
- AWGN channel modeling
- Theoretical and Monte Carlo BER comparison
- Interactive experiment dashboard

## Status

M00: Engineering foundation complete.

M01: Gaussian Q-function numerical analysis complete.

M02: BPSK/AWGN communication simulation implemented and validated with 159 automated tests.

M03: Gray-coded QPSK and 16-QAM BER simulation complete and merged into GitHub main; 299 tests passed.

M04: Statistical convergence, simulator throughput, and rare-error reliability studies validated locally. GitHub integration pending.

## Python

Python 3.12 or later is required.


## M02 - BPSK over AWGN

The communication simulator supports seeded BPSK modulation, real AWGN,
bounded-memory Monte Carlo BER analysis, theoretical comparison, and
95% Wilson confidence intervals.

Run the complete test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Generate the BPSK BER experiment:

```powershell
.\.venv\Scripts\python.exe experiments\m02_bpsk_ber_sweep.py
```

See [M02 methodology and results](docs/M02_BPSK_AWGN_BER.md).

![BPSK BER comparison](experiments/results/m02/bpsk_ber_sweep.png)

## M03 - QPSK and 16-QAM over AWGN

The simulator also supports Gray-coded QPSK and 16-QAM with
unit-average-symbol-energy normalization, complex AWGN, ideal
hard-decision demodulation, theoretical BER models, and reproducible
bounded-memory Monte Carlo estimation.

At equal Eb/N0, the comparison shows the BER and bits-per-symbol
tradeoff between QPSK and 16-QAM.

Generate the modulation comparison experiment:

```powershell
.\.venv\Scripts\python.exe experiments\m03_modulation_ber_comparison.py
```

The experiment processes 7,000,000 transmitted bits across 14
configurations and saves a CSV dataset and semilogarithmic BER plot.

See [M03 methodology and results](docs/M03_QPSK_16QAM_BER.md).

![QPSK and 16-QAM BER comparison](experiments/results/m03/modulation_ber_comparison.png)

## M04 - Statistical Reliability and Performance

M04 investigates Monte Carlo convergence, runtime throughput for
BPSK/QPSK/16-QAM, and the interpretation of rare or zero observed errors.

See [M04 methodology and results](docs/M04_STATISTICAL_BENCHMARKING.md).

![Monte Carlo BER convergence](experiments/results/m04/bpsk_convergence.png)

![Simulator throughput](experiments/results/m04/modulation_throughput.png)

![Rare-error reliability](experiments/results/m04/rare_error_reliability.png)

## M05 - Interactive Communication Simulator

CommuLab includes a four-page Streamlit application:

- BER Explorer
- Constellation Explorer
- Numerical Q-function Lab
- Experiment Analytics for M01-M04

Install the dashboard and development dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard]"
```

Launch the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

See [M05 dashboard documentation](docs/M05_INTERACTIVE_DASHBOARD.md).
