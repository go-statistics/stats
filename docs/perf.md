# Speed

The comparison is with SciPy 1.18.1 / NumPy 2.5.4 (native arm64 wheels, the
build a user of this machine runs), on the same data, **core for core**.

## Method

- `go test -run TestPerfReport -perf.report=docs/perf-table.md -timeout 180m`
  starts SciPy through `uv` (one `python -c` process, fed one JSON request
  per line; every case's SciPy arguments and call are Python expressions
  defined in Go, `perf_scipy_test.go`) and, case by case and round by round,
  times back to back the SciPy call, the Go call on one core and the Go call
  a user makes (all cores).
- **One core on both sides.** SciPy runs with one BLAS/OpenMP thread. The
  Go one-core column is a single-goroutine loop for the elementwise
  functions, and the function itself under `GOMAXPROCS=1` (set outside the
  timed interval) for everything else. The ratios quoted below are that
  column; the all-cores column (`stats.Map`, the parallel KDE) is shown for
  information and never compared with single-threaded SciPy as parity.
- **Every public operation**: 355 cases — every distribution's pdf, logpdf
  (pmf, logpmf), cdf, sf, ppf, isf, draws, entropy and fit (32
  distributions), every descriptive statistic, every test, KDE, bootstrap,
  the multivariate distributions. Points are generated identically on both
  sides (`lo + (hi-lo)·frac(0.618…·i)`), normal samples by the same seeded
  generator (go-ndarray reproduces NumPy's); `go test -run
  TestPerfSciPyData -perf.datahash=FILE` hashes the arguments SciPy
  receives (byte-identical to those of the former Python server for all 355
  cases).
- Rounds are interleaved case by case so that the load of the shared host
  hits both sides alike; medians of 7 rounds (3 for calls over 2 s, 1 over
  20 s). The host is shared (16 cores): its load is in each table's header.
- `go test -run TestPerfCompare -perf.compare=BEFORE,AFTER,OUT` writes
  [perf-compare.md](perf-compare.md) from two tables.

## Results

[perf-table.md](perf-table.md) is this version, [perf-table-before.md](perf-table-before.md)
v0.1.0 measured with the same harness, [perf-compare.md](perf-compare.md)
the two side by side with the cause of every row still slower than SciPy.

Of 355 operations, **267 were slower than SciPy on one core in v0.1.0, 236
are now**. 160 of those remain slower for reasons in this package, all of
them the price of its accuracy (below); 51 for reasons in
go-numerics/special and 25 in go-ndarray's generator.

Selected rows (SciPy time over Go one-core time; above 1 is faster):

| case | n | v0.1.0 | now | gain |
|---|--:|--:|--:|--:|
| pearsonr | 10^6 | 0.106x | 0.599x | 5.6x |
| linregress | 10^6 | 0.056x | 0.351x | 6.3x |
| spearmanr | 10^6 | 0.879x | 2.22x | 2.5x |
| var / std | 10^6 | 0.35x / 0.47x | 1.52x / 1.53x | 4.3x / 3.3x |
| mean | 10^6 | 0.348x | 0.966x | 2.8x |
| kurtosis | 10^6 | 0.597x | 1.29x | 2.2x |
| rankdata | 10^6 | 0.871x | 2.15x | 2.5x |
| cumfreq | 10^6 | 0.626x | 4.86x | 7.8x |
| ks_2samp | 10^5 | 1.16x | 4.02x | 3.5x |
| differential_entropy | 10^5 | 0.468x | 1.32x | 2.8x |
| entropy / gmean / pmean | 10^6 | 0.040x / 0.061x / 0.064x | 0.208x / 0.395x / 0.233x | 5.1x / 6.5x / 3.6x |
| power_divergence / chisquare | 10^6 | 0.062x / 0.183x | 0.243x / 0.478x | 3.9x / 2.6x |
| f_oneway | 3·10^6 | 0.164x | 0.509x | 3.1x |
| dirichlet.logpdf | 10^5 | 0.079x | 0.321x | 4.1x |
| t.pdf / t.logpdf | 10^5 | 0.082x / 0.071x | 0.511x / 0.591x | 6.3x / 8.4x |
| beta.pdf / gamma.pdf | 10^6 | 0.201x / 0.242x | 0.747x / 0.425x | 3.7x / 1.8x |
| zipf.pmf / zipf.cdf | 10^5 | 0.027x / 0.877x | 1.21x / 8x | 45x / 9.1x |
| gumbel_r.rvs / weibull_min.rvs / pareto.rvs | 10^6 | 0.051x / 0.074x / 0.094x | 0.251x / 0.269x / 0.315x | 4.9x / 3.7x / 3.4x |
| t.fit / beta.fit / f.fit | 10^4 | 0.020x / 0.0083x / 0.00037x | 0.197x / 0.040x / 0.0010x | 10x / 4.9x / 2.7x |

The tables were measured on 2026-10-17 with go-numerics/special v0.1.0 on a
host whose load ranged from 13 to 135 during the runs (other sessions'
builds): the ratios are robust to load only to the extent the interleaving
makes them, so single rows can move by tens of percent between runs. The
smallest calls (expon.fit and uniform.fit at 10^4, about 10 µs) were
dominated in those runs by the switch to `GOMAXPROCS=1` inside the timed
interval; the harness now sets it outside, and on one core they take
0.015 ms and 0.009 ms against SciPy's 0.05 and 0.014 ms.

## What changed, and why the results are the same bits

Every change returns, bit for bit, what v0.1.0 returned with the same
go-numerics/special: a golden comparison of 2.4 million values — every
distribution function at 3000 points and its tails, every draw, entropy
and fit, every descriptive statistic and test at n = 7, 130, 5000 and
100001, KDE, the multivariate distributions — is byte-identical between
v0.1.0 and this branch (with special v0.1.0 on both sides, and again with
special's main branch on both sides), and the accuracy and parity tests
pass with their budgets unchanged. The techniques:

- **Ziv's strategy where a value is rounded once at the end.** A cheaper
  evaluation with a rigorous running error bound decides the rounding; the
  accurate evaluation runs only when the bound cannot (near a midpoint
  between two floats, or exactly on one):
  - the inner products of Pearson, Spearman, linregress, Shapiro-Wilk and
    one-way ANOVA (`dotDev`): a compensated inner product (Ogita, Rump and
    Oishi's Dot2: TwoProd and TwoSum per term) on two accumulators, the
    deviations from the mean formed on the fly, decided against both its own
    bound and the double-double sum's (3u²·n·Σ|p|); undecided for about 3
    data sets in 10^4 of normal data;
  - the chi-squared sums (`sumFast`, the same on plain terms);
  - the correctly rounded logarithm (`crLog`: `dd.Log`, 2^-72, decides
    unless within 2^-66 of a midpoint, then `dd.LogAccurate`) and power
    (`crPow`: the exponent from `dd.Log`, the mantissa of `dd.ExpMant`
    decided against both paths' bounds);
  - the Dirichlet log-density.

  Checked on 10^8 random and edge arguments (crLog), 10^7 (crPow) and
  300 000 data sets including ill-conditioned and cancelling ones (inner
  products): `go test -run 'Fast|Identity' -fastpath.n=...`.
- **A memo of shape-only constants** (1/Γ(a), 1/B(a,b), the t constant, their
  logarithms, ζ(a)): a lock-free seqlock of atomics holding the values the
  functions compute (`shapecache.go`).
- **No temporary arrays for sums**: NumPy's pairwise summation walked
  without bounds checks (3.3x faster), and `sumTerms`, which fills each leaf
  of the same recursion into one reused buffer (variances, moments,
  weighted means, trimmed statistics, ANOVA).
- **Radix sort** for the statistics that sort, on data without NaN and
  without both signed zeros, where the sorted sequence is unique as bit
  patterns (otherwise the comparison sort runs).
- Builtin `min`/`max` (math.Min/Max's semantics, inline) instead of calls,
  branch-free masks for the trimmed statistics, the KDE maximum without
  calls, the MVN solve on the stack.

## Where it is still slower, and why

- **go-numerics/special** (51 rows): the cdf, sf and quantiles of the beta,
  t, F, gamma, chi-squared, chi, inverse gamma, noncentral chi-squared,
  binomial, negative binomial, Poisson, normal and log-normal distributions
  are its incomplete gamma and beta functions, their inverses, Ndtr and
  Ndtri; t.ppf (0.005x) and the beta/F quantiles (0.02–0.04x) are the worst.
  Its main branch (faster incomplete functions, not yet tagged) is in this
  branch's go.mod; the tables above predate it.
- **go-ndarray** (25 rows): draws made by its generator (geometric,
  logistic, Zipf, multinomial, Dirichlet, …), reproduced bit for bit from
  NumPy's algorithms.
- **This package, by design** (160 rows):
  - densities, closed-form distribution functions and discrete masses
    evaluated in double-double for 0.5–2 ulp results (SciPy's are off by up
    to 10^16 ulp in the tails): 0.2–0.9x;
  - the numerical fits (beta, chi, F, gamma, t, …): Nelder–Mead polished to
    about 1e-10 relative with restarts, where SciPy's fmin stops at four
    digits — 10 to 100 times more likelihood evaluations, each a sum of
    double-double log-densities: 0.001–0.2x;
  - correctly rounded logarithms and powers per element (entropy, gmean,
    pmean, power_divergence) and SciPy's draw transforms with correctly
    rounded elementary functions (gumbel, weibull, pareto, lomax): 0.2–0.4x;
  - exact inner products (Pearson 0.6x, linregress 0.35x) and the plain
    pairwise loops where NumPy has SIMD kernels (mean 0.97x, average 0.85x,
    tmean 0.63x): candidates for go-simd kernels;
  - multivariate normal log-density and cdf (double-double triangular solve
    per point, randomised lattice rule in double-double): 0.07x and 0.24x;
  - mode on data holding both -0 and +0 (rounded data): the comparison sort,
    since the radix order of the two zeros would change which one is
    reported.

The scalar entropies, moments and the small tests (Fisher exact, chi-squared
contingency) are faster than SciPy by its per-call overhead.
