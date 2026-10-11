# Speed

The comparison is with SciPy 1.18.1 / NumPy 2.5.4 (native arm64 wheels, the
build a user of this machine runs), vectorised calls, on the same data.

## Method

- `go test -run TestPerfReport -perf.report=docs/perf-table.md` starts
  `bench/scipy_server.py` through `uv` and drives it over a pipe. For every
  case and every round it times, back to back, the SciPy call, a
  single-goroutine Go loop calling the scalar method (elementwise cases), and
  the call a Go user makes (`stats.Map(x, d.CDF)`, which splits the array
  over GOMAXPROCS goroutines; or the function itself).
- Points are generated identically on both sides
  (`lo + (hi-lo)·frac(0.618…·i)`), normal samples by the same seeded
  generator (go-ndarray reproduces NumPy's).
- Rounds are interleaved case by case so that the load of the shared host
  hits both sides alike; the table reports medians of 7 rounds. SciPy's time
  includes allocating its result, as Go's does.
- The host is shared (16 cores; load average in the table header, about 30
  during the run). Absolute times are inflated for both sides; the ratios are
  the robust numbers, and the single-goroutine column is the fair
  core-for-core comparison.

## Results

[perf-table.md](perf-table.md) (generated). Summary, Go (all cores) against
SciPy (one core, as SciPy runs):

| case | SciPy (ms) | Go (ms) | ratio |
|---|--:|--:|--:|
| norm.pdf, 10^6 | 15.8 | 12.6 | 1.26x faster |
| norm.cdf, 10^6 | 29.8 | 17.9 | 1.66x faster |
| norm.ppf, 10^6 | 27.1 | 43.9 | 1.6x slower |
| beta.pdf, 10^6 | 94.2 | 77.9 | 1.21x faster |
| gamma.cdf, 10^6 | 275 | 243 | 1.13x faster |
| beta.cdf, 10^6 | 298 | 595 | 2x slower |
| beta.ppf, 10^5 | 173 | 1149 | 6.6x slower |
| t.cdf, 10^6 | 275 | 842 | 3x slower |
| t.ppf, 10^5 | 50 | 1096 | 22x slower |
| norm.rvs / gamma.rvs, 10^6 | 9.6 / 46.6 | 11.2 / 34.2 | 0.86x / 1.36x |
| ttest_ind, 10^6 + 10^6 | 6.8 | 7.7 | 0.88x |
| ttest_1samp, 10^6 | 3.1 | 3.2 | 0.99x |
| pearsonr, 10^6 | 5.9 | 97 | 16x slower |
| spearmanr, 10^6 | 861 | 368 | 2.3x faster |
| describe, 10^6 | 184 | 60 | 3.1x faster |
| percentile (5 q), 10^6 | 40.7 | 27.2 | 1.5x faster |
| gaussian_kde evaluate, 10^4 × 10^4 | 916 | 113 | 8.1x faster |

## Where it is slower, and why

- **Single core, the scalar distribution functions are 2–15x slower than
  SciPy's C/C++ ufuncs** (the one-goroutine column). The accuracy costs: the
  densities carry their exponent in double-double and exponentiate with a
  nearly correctly rounded exponential; the parallel `Map` recovers the
  difference for the cheap functions on a multi-core machine.
- **t and beta cdf/ppf** go through go-numerics/special's incomplete beta
  function and its inverse, which are 10–15x slower than Boost's (their own
  measurement, go-numerics/special docs/perf.md); t.ppf additionally pays two
  inverse evaluations to get both z and 1-z without cancellation. A scalar
  `StudentT.PPF` call is about 75 µs: irrelevant for a confidence interval,
  slow for 10^5 quantiles.
- **norm.ppf** is special.Ndtri (correctly rounded but for rare cases),
  slower than Cephes' rational approximation (which is off by up to 2.8 ulp).
- **Pearson** computes its sums of products in double-double (exact
  correlation for nearly collinear data), where SciPy uses a BLAS dot
  product.
- **t-tests** are at parity: both sides are a couple of passes of pairwise
  sums over the data.
