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

## Headline (distributions)

Over 256 fixtures (33 distributions × 8 functions, 1479 regions, about
212 000 points), the maximum error of this package is at or below SciPy's
in **1473 of 1479 regions**. The six exceptions are all in the beta
distribution and all below 4.5 ulp (next section). In hundreds of regions
SciPy's error is in the thousands to 10^16 ulp, or its result is "wrong"
(NaN, 0 or ±Inf where the true value is finite): deep tails, large shapes,
log forms beyond underflow.

The per-function summary and every region are in
[accuracy-table.md](accuracy-table.md) (generated; measured on arm64, the
budgets also hold on amd64, 386, arm, riscv64, loong64, ppc64le and s390x,
where Go's math routines and fused multiply-adds move the last bits: the
logarithmic forms reach up to 2 ulp more there, see the budgets).

Typical worst cases per family, ours against SciPy's:

| family | ours (max over regions) | SciPy |
|---|---|---|
| normal: pdf, logpdf, cdf, sf, log forms, ppf, isf | ≤ 0.55 ulp | up to 455 ulp (pdf tail), 1e10–1e12 (cdf/sf deep tails) |
| gamma, chi-squared, chi, inverse gamma: pdf | ≤ 1.3 ulp | up to 6e6 ulp at large shapes |
| gamma family: cdf, sf | ≤ 2.1 ulp | up to 4e4 ulp |
| gamma family: ppf, isf | ≤ 1.9 ulp | up to 5e6 ulp |
| beta: pdf, logpdf | ≤ 1.1 / 3.9 ulp | 948 / 23375 ulp |
| beta: cdf, sf | ≤ 5.6 / 4.4 ulp | 7334 / 2.9 ulp (see below) |
| Student t: pdf, logpdf | ≤ 0.5 ulp | 5.6e4 / 3.0e4 ulp |
| Student t: cdf, sf, ppf, isf | ≤ 2.4 ulp, quantiles ≤ 9.4 ulp below 1e-20 | 1067 ulp, wrong (NaN/0) in far tails |
| F: pdf, cdf, sf, ppf, isf | ≤ 1.2 / 4.8 / 2.8 / 2.2 / 2.6 ulp | 7e4 / 1971 / 389 / wrong / wrong |
| noncentral chi-squared | pdf ≤ 9.8, cdf ≤ 116, sf ≤ 151 ulp (deep tails summed term by term) | up to 1e15–1e16 ulp, wrong |
| closed-form families (exponential … von Mises) | ≤ 0.5–4 ulp, every region ≤ SciPy | up to 1e15 ulp, wrong (see the table) |

## Where this package is not at least as accurate as SciPy

| fixture | region | ours max | SciPy max |
|---|---|--:|--:|
| beta_sf | small_mid | 1.93 | 0.50 |
| beta_sf | mid_mid | 0.81 | 0.49 |
| beta_sf | mid_hi | 0.70 | 0.50 |
| beta_sf | large_mid | 2.11 | 2.11 (equal to 3 digits) |
| beta_sf | large_hi | 4.42 | 2.45 |
| beta_ppf | large_core | 1.03 | 0.79 |

The beta survival function and quantile are go-numerics/special's
`BetaIncC` and `BetaIncInv` called directly; Boost's (SciPy's) are a little
more accurate on these points. Reformulating (`I_{1-x}(b, a)` for x ≥ 1/2,
where 1-x is exact) gave the same errors, so the fix belongs in the
incomplete beta function itself. The same function is why the binomial and
negative binomial cdf/sf reach 6–11 ulp (SciPy: 10^6–10^7 ulp there).

Other limits stated plainly:

- The ULP fixtures sample standard forms; with loc and scale, (x-loc)/scale
  is rounded exactly as SciPy rounds it, and the result inherits that.
- The noncentral chi-squared sf in the far upper tail is a sum of many
  incomplete gamma terms: up to 151 ulp (SciPy up to 8.5e15).
- On s390x and ppc64le, where Go's log1p/exp differ in the last bits, the
  log forms of the gamma/beta families reach about 2 ulp more than on
  arm64/amd64 (budgets raised from measurements under qemu,
  `testdata/raise_budgets.py`); the comparison table is arm64's.
- Zipf's SciPy column is partial: SciPy's zipf cdf and quantile sum every
  mass up to k, so points with k beyond 1e4 are not compared with SciPy
  (they are with mpmath).

## Techniques

- The exponent of every density is carried in double-double and
  exponentiated once (internal/dd's ExpMant is accurate to about 2^-76);
  for large shapes the gamma and beta densities take Stirling's factors out
  analytically, and Γ(x+½)/Γ(x) has its own double-double series.
- Rounded arguments are corrected: the t, F, chi and inverse-gamma
  distribution functions evaluate the incomplete function at the float64
  part of a double-double argument (t²/(ν+t²), m·x/(m·x+n), x²/2, 1/x) and
  add the first-order term density × low part.
- Complements, never 1 - p: each tail is computed directly; log forms use
  log1p of the other tail near 1; beyond the incomplete functions' range
  (subnormal arguments, probabilities below 1e-280) the leading term is
  evaluated in logarithms.
- Entropies of the gamma and beta families cancel their x·log x terms
  analytically (SciPy's formulas lose up to 3e-9 relative, and
  scipy.stats.beta(3, 2e7).entropy() returns 0.0 where the value is -14.96).

## Tests, descriptive statistics, multivariate

Compared with SciPy 1.18.1 / NumPy 2.5.4 (x86-64) on generated datasets,
and with exact references (mpmath, rational arithmetic) where they exist:

- Means, variances, standard deviations, medians, all 13 quantile methods
  (also weighted), histograms and bin edges: **bit-identical to NumPy**
  (tens of thousands of values). skew, kurtosis, mode, sem, zscore,
  trimmed statistics, gmean/hmean/pmean, binned_statistic: bit-identical to
  SciPy; describe, moment, iqr, trim_mean, differential_entropy within
  2e-15. Relative entropy is computed in double-double: 3e-16 against
  mpmath where SciPy is off by 1e-6 on data offset by 1e8.
- Hypothesis tests (about 4100 SciPy cases plus 3100 exact references):
  statistics and p-values agree with SciPy to ≤ 1.4e-13 relative except
  where an exact reference shows SciPy is off: Fisher exact p (2e-16 vs
  3e-15), chi2_contingency near independence (statistic 1.8e-12 vs 2.2e-7),
  KS two-sample exact (2.4e-15 vs 3.3e-12), Kolmogorov distribution
  (Pomeranz in double-double), Mann-Whitney exact (1.5e-15 vs 1.2e-13),
  Anderson-Darling Gumbel fit (5.6e-15 vs 1.1e-8), Levene (7.7e-11 vs
  6.3e-10), Pearson with |r| = 1 (p = 0 exactly).
- Multivariate normal logpdf/pdf/entropy: exact against mpmath where SciPy
  is off by up to 4.6e-8 / 9.6e-7 / 3e-9 (covariances with condition number
  up to 3e9); bivariate cdf 3.9e-16 relative including the tails where SciPy
  returns 0; Dirichlet and multinomial pmf/pdf/entropy ≤ 3e-15 (SciPy up to
  1.6e-12). Gaussian KDE within 1e-13 of SciPy.

## Negative controls

A deliberately broken formula must fail the suite:

- `testdata/negative_control.sh accurate.go "1, -0.125, 0.0078125" "1,
  -0.1250001, 0.0078125"` (the Γ(x+½)/Γ(x) series): TestAccuracy fails in
  every t_logpdf / t_pdf region, by up to 3.6e7 ulp.
- Closed-form families: expm1D's 1/24 coefficient → 1/23: 19 regions of
  expon, gumbel_r, lomax, pareto, rayleigh and weibull_min fail (up to 5e7
  ulp).
- Descriptive: `_lerp`'s `t >= 0.5` → `t > 0.5` and the kurtosis bias
  coefficient 3 → 3.0000001: quantile/percentile bit mismatches and kurtosis/
  describe failures (71 messages).
- Tests: Shapiro's 0.221157 → 0.221158 (68 failures); Kendall's variance
  2n+5 → 2n+6 (162 failures).
- Discrete: Loader's Stirling coefficient 1/1260 → 1/1261: TestAccuracy
  fails in 21 fixtures (binomial, Poisson, negative binomial,
  hypergeometric masses and tails, Zipf quantiles) and TestParity in four.
- Multivariate: four perturbations (MVN log-density factor, KDE neff, a
  Gauss-Legendre weight, the BCa cube) each fail their parity tests.
