# M03 — Gray-coded QPSK and 16-QAM over AWGN

## Status

Implementation and scientific experiment validated locally.
GitHub integration remains pending.

## Objective

Extend CommuLab's BPSK/AWGN simulator to support complex-valued,
Gray-coded QPSK and 16-QAM, theoretical BER models, and bounded-memory
Monte Carlo comparison at equal Eb/N0.

## 1. Modulation design

### QPSK

Each symbol carries two bits:

    s = ((1 - 2*bI) + j*(1 - 2*bQ)) / sqrt(2)

The resulting symbols have energy Es = 1 and Eb = 1/2.

The first bit controls the in-phase sign; the second controls the
quadrature sign. Adjacent constellation symbols differ by one bit.

### 16-QAM

Each symbol carries four bits. Each axis uses the Gray-labeled
four-level mapping:

| Bits | Unnormalized amplitude |
| --- | ---: |
| 00 | +3 |
| 01 | +1 |
| 11 | -1 |
| 10 | -3 |

The first pair controls I and the second controls Q. Both coordinates
are divided by sqrt(10).

The 16-point constellation has average Es = 1 for equiprobable
symbols, so Eb = 1/4.

Individual symbol energies vary: 0.2, 1.0, or 1.8. Unit average
symbol energy does not mean unit energy for every 16-QAM symbol.

## 2. Complex AWGN

Received symbols follow:

    r = s + nI + j*nQ

The independent noise components have distribution:

    nI, nQ ~ Normal(0, sigma^2)

For k bits per symbol, average Es = 1 and
gamma_b = 10^(EbN0_dB/10):

    sigma^2 = 1 / (2*k*gamma_b)

Thus the variance of each real noise component is 0.25 for QPSK and
0.125 for 16-QAM at Eb/N0 = 0 dB.

The channel does not normalize input symbols automatically.

## 3. Theoretical BER

QPSK uses the coherent Gray-coded AWGN expression:

    Pb_QPSK = Q(sqrt(2*gamma_b))

This equals the coherent BPSK BER at the same Eb/N0.

For the implemented Gray-coded 16-QAM labeling and ideal
hard-decision detection, let:

    a = sqrt(4*gamma_b/5)

The exact average bit-error probability is:

    Pb_16QAM = 0.75*Q(a) + 0.5*Q(3*a) - 0.25*Q(5*a)

The high-SNR approximation is 0.75*Q(a).

These formulas assume equally likely bits, independent complex AWGN,
ideal synchronization, and the specified Gray labeling.

## 4. Monte Carlo implementation

The simulator:

- Generates reproducible random bits.
- Groups bits into complete modulation symbols.
- Maps symbols with unit average constellation energy.
- Adds independent I/Q Gaussian noise.
- Applies hard-decision demodulation.
- Counts bit errors in bounded symbol batches.
- Calculates simulated BER and a two-sided 95% Wilson interval.

Batch size is measured in symbols. Memory use depends on the batch
size instead of total transmitted bits.

The simulation uses seeded random streams for bit generation and noise.

## 5. Reproducible comparison experiment

Command:

    python experiments/m03_modulation_ber_comparison.py

Configuration:

| Parameter | Value |
| --- | --- |
| Modulations | QPSK, 16-QAM |
| Eb/N0 values (dB) | 0, 2, 4, 6, 8, 10, 12 |
| Bits per configuration | 500,000 |
| Batch size | 25,000 symbols |
| Base seed | 2026 |
| Total configurations | 14 |
| Total bits | 7,000,000 |

Each configuration receives a recorded deterministic seed.

Generated artifacts:

- experiments/results/m03/modulation_ber_comparison.csv
- experiments/results/m03/modulation_ber_comparison.png

## 6. Selected numerical observations

| Eb/N0 (dB) | Scheme | Errors | Simulated BER | Theoretical BER |
| ---: | --- | ---: | ---: | ---: |
| 0 | QPSK | 39,269 | 0.078538 | 0.07864960 |
| 0 | 16-QAM | 70,198 | 0.140396 | 0.1409816 |
| 4 | QPSK | 6,246 | 0.012492 | 0.01250082 |
| 4 | 16-QAM | 29,176 | 0.058352 | 0.05862374 |
| 6 | QPSK | 1,111 | 0.002222 | 0.002388291 |
| 8 | QPSK | 113 | 0.000226 | 0.0001909078 |
| 8 | 16-QAM | 4,690 | 0.009380 | 0.009247214 |
| 10 | QPSK | 4 | 0.000008 | 0.000003872108 |
| 12 | QPSK | 0 | 0 | 0.00000000900601 |
| 12 | 16-QAM | 68 | 0.000136 | 0.0001386587 |

The full CSV contains all 14 results and their Wilson interval
bounds.

### Statistical interpretation

Monte Carlo BER is an estimate. A 95% Wilson interval does not
guarantee that every independent experiment contains the theoretical
error probability.

For example, the 6 dB QPSK experiment recorded 1,111 errors compared
with approximately 1,194 expected errors. Its individual 95% interval
does not contain the theoretical BER. This is a statistical deviation,
not sufficient evidence by itself of a faulty channel model.

At 12 dB, QPSK observed zero errors among 500,000 bits. This does
not establish a true BER of zero. The graph plots the Wilson upper
confidence bound separately from the theoretical curve.

For very low BER, a larger number of transmitted bits or a different
rare-event estimation method would be needed for precise measurement.

## 7. Interpretation and limitations

At equal Eb/N0, QPSK achieves lower BER than 16-QAM under the tested
uncoded AWGN assumptions.

16-QAM carries four bits per symbol versus two for QPSK. This can
improve spectral efficiency for a fixed symbol rate, at the cost of
greater sensitivity to noise.

These results do not model fading, interference, coding, imperfect
carrier synchronization, nonlinear amplifiers, or hardware distortion.

## 8. Validation

Local M03-E1 acceptance:

- 299 automated tests passed.
- Ruff passed for src, tests, and experiments.
- mypy passed for 16 source files.
- Experiment generated 14 CSV records and a PNG figure.
- The experiment processed 7,000,000 transmitted bits.
- The dependency integrity gate passed at M03-D2.

GitHub merge and post-merge regression remain to be completed.
