# Accuracy

Accuracy is the point of this package: it is measured, published and
enforced by `go test`, not asserted.

## How the error is measured

**Oracle.** Every value of the ULP fixtures (`testdata/fixtures`, written by
`testdata/gen/gen.py`) is computed by [mpmath](https://mpmath.org) 1.4.1
twice, at 192 and 320 bits; a point whose two evaluations differ by more than
2^-80 relative, or where mpmath cannot decide (no convergence, no bracket for
a quantile), is dropped and counted in the fixture header. The incomplete gamma
and beta oracles never trust a series blindly: the smaller tail is computed
directly and the other as its complement with the cancellation paid for in
extra bits, and large parameters integrate the tail numerically
(`testdata/gen/mpincomplete.py`, shared with go-numerics/special). A quantile's
oracle is the exact root of the exact distribution function at the float64
probability, found by a safeguarded Newton iteration on the logarithm of the
cdf or sf (in log x on a positive support) inside a verified bracket.

**Exact reference.** Each row stores the mpmath value as a double-double
`hi + lo`; the error of a result `y` is `|y - (hi + lo)| / ulp(hi)`, so a
correctly rounded result scores at most 0.5.

**SciPy on the same points.** The fixture also holds the value SciPy 1.18.1
returns for the same float64 arguments, computed with the x86-64 build (the
reference build: NumPy and SciPy's macOS arm64 wheels round some libm calls
differently), and the Go test measures it exactly as it measures this
package. SciPy is a point of comparison, never the reference.

**Regions.** Each distribution function is measured in standard form (loc 0,
scale 1) over regions that cross the shape regimes (small, moderate, large and
huge shapes) with the sides of the distribution: the points `x` are SciPy's own
quantiles at log-uniform probabilities from 1e-300 to 1e-3 in both tails, and
uniform probabilities in the bulk; the quantile functions take the
probabilities themselves, down to 1e-300 and up to 1 - 1.2e-16.

**Budgets.** `TestAccuracy` fails when a region's maximum error exceeds its
budget (`testdata/budgets/<fixture>.txt`: 1.5 times the larger of the errors
measured natively on arm64 and under amd64, at least 1 ulp). CI runs the
fixtures on amd64, arm64, 386, arm, riscv64, loong64, ppc64le and s390x:
architectures where Go fuses multiply-adds and ones where it does not meet
the same budgets.

Regenerate:

```
uv run --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
    python testdata/gen/gen.py                      # mpmath pass, native
uv run --python cpython-3.12-macos-x86_64-none --with mpmath==1.4.1 \
    --with scipy==1.18.1 --with numpy==2.5.4 \
    python testdata/gen/gen.py --scipy-column       # SciPy column, x86-64
go test -run TestAccuracyBudgets -accuracy.budgets
GOARCH=amd64 go test -run TestAccuracyBudgets -accuracy.budgets -accuracy.merge
go test -run TestAccuracyTable -accuracy.table=docs/accuracy-table.md
```
