# M01 ? Gaussian Q-Function Numerical Analysis

## 1. Background

CommuLab extends an unfinished 2023 Communication Systems academic project
investigating numerical approximations to the Gaussian Q-function.

The original academic materials are preserved separately. The present
implementation is a Python redevelopment with automated tests and
reproducible numerical experiments.

## 2. Mathematical definition

For a standard normally distributed random variable Z:

\[
Q(x) = P(Z > x)
     = \frac{1}{\sqrt{2\pi}}
       \int_x^\infty \exp(-t^2/2)\,dt.
\]

Craig's integral representation, valid for x >= 0, is:

\[
Q(x) = \frac{1}{\pi}
       \int_0^{\pi/2}
       \exp\left(-\frac{x^2}{2\sin^2\theta}\right)d\theta.
\]

For reference evaluation, the implementation uses SciPy's ndtr(-x).
Log-domain evaluation uses log_ndtr(-x).

## 3. Implemented algorithms

Five composite quadrature methods are supported:

1. Left Riemann sum
2. Right Riemann sum
3. Midpoint rule
4. Trapezoidal rule
5. Simpson's rule

Simpson's rule requires an even, positive interval count.

The reference Q-function supports finite real inputs.
The Craig quadrature API accepts finite, nonnegative inputs.

## 4. Endpoint behavior

For x > 0, the integrand approaches zero as theta approaches zero.

For x = 0, the integrand equals one throughout the integration
interval, including the explicitly defined endpoint value.

The implementation handles this corner case without dividing by zero.

## 5. Boundary-layer limitation

For small positive x, the integrand changes sharply near theta = 0.
Fixed-grid methods may fail to resolve this narrow region.

For x = 1e-8, the midpoint method with 4096 intervals had a
relative Q-function error of approximately 7.978e-9.

This apparently small relative error does not mean the method accurately
resolves the difference between Q(x) and 0.5.

Fixed-grid convergence is not uniform across all x.

## 6. Adaptive integration observation

SciPy adaptive quadrature was also evaluated.

At x = 1e-8, it returned approximately 0.5, missing the
small endpoint contribution even though its reported quadrature
error estimate was much smaller than the observed error.

Therefore, adaptive quadrature error estimates are not treated
as independent guarantees of accuracy.

## 7. Reproducible experiment

Script:

`experiments/m01_quadrature_convergence.py`

Parameters:

- Five integration methods
- Five x values: 0.0001, 0.01, 0.1, 1.0, 5.0
- Seven interval counts: 16, 32, 64, 128, 256, 512, 1024
- Total: 175 experiment configurations

Generated outputs:

- `experiments/results/m01/quadrature_convergence.csv`
- `experiments/results/m01/quadrature_convergence.png`

The CSV includes the reference value, estimate, absolute error,
and relative error for every configuration.

The plotted convergence experiment uses x = 0.01.

## 8. Verification evidence

At the time of M01-D execution:

- 44 automated tests passed
- Ruff checks passed
- mypy checks passed
- All 175 CSV records passed recomputation and uniqueness checks
- The generated plot was visually reviewed

These results demonstrate correctness for the tested configurations;
they do not establish uniform accuracy across every possible input.

## 9. Scientific conclusions

The midpoint, trapezoidal, and Simpson methods achieve high accuracy
for many tested moderate and large positive x values.

Left and right Riemann sums generally exhibit larger discretization
errors at the tested resolutions.

No single method should be described as uniformly most accurate
without specifying the input range, grid resolution, and error metric.

The communication-system BER module should use the trusted
reference Q-function by default, while the quadrature engine
remains available for numerical research and comparison.

## 10. Limitations and future work

- Analyze very small positive x values more extensively
- Investigate a transformed-integral or adaptive-grid formulation
- Benchmark execution time separately from numerical accuracy
- Validate extreme tails with log-domain comparisons
- Add stronger property-based and convergence-order testing
- Connect Q-function evaluation to modulation BER equations
- Compare original 2023 MATLAB approximations in a separate study
