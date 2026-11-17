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
