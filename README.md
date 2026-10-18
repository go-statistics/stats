# stats

Probability distributions, descriptive statistics and statistical tests in
pure Go (no cgo), for the go-ndarray notebook ecosystem: the results of
`scipy.stats` (and NumPy's statistics), with every distribution function
measured in ULPs against [mpmath](https://mpmath.org), at least as accurate as
SciPy in 1473 of 1479 measured regions, and random draws that reproduce
SciPy's for the same seed.

```
go get github.com/go-statistics/stats
```

Built on [go-ndarray](https://github.com/go-ndarray/ndarray) (arrays, linear
algebra, and `random`: NumPy's `Generator` bit for bit),
[go-numerics/special](https://github.com/go-numerics/special) and
[go-numerics/optimize](https://github.com/go-numerics/optimize).

## Examples, with the SciPy call each one is

```go
n := stats.Normal{Loc: 0, Scale: 1}
n.PDF(1.5)    // scipy.stats.norm.pdf(1.5)       0.12951759566589172
n.LogCDF(-40) // scipy.stats.norm.logcdf(-40)   -804.6084420137538
n.PPF(0.975)  // scipy.stats.norm.ppf(0.975)     1.9599639845400538

// scipy.stats.beta(2.5, 7).ppf(p) for an array p (parallel over the cores)
q := stats.Map(p, stats.Beta{A: 2.5, B: 7, Scale: 1}.PPF)

// scipy.stats.binom.pmf(4, 10, 0.3)
stats.Binomial{N: 10, P: 0.3}.PMF(4)

// scipy.stats.norm.rvs(10, 2, size=1000, random_state=np.random.default_rng(42)), to the bit
x, _ := stats.Normal{Loc: 10, Scale: 2}.Rvs(random.DefaultRNG(42), 1000)

// scipy.stats.gamma.fit(x, floc=0)
g, _ := stats.FitGamma(x, &stats.FitOptions{Fixed: map[string]float64{"Loc": 0}})

// scipy.stats.ttest_ind(a, b, equal_var=False) and .confidence_interval(0.95)
r, _ := stats.TTestInd(a, b, &stats.TTestIndOptions{Welch: true})
ci, _ := r.ConfidenceInterval(0.95)

// numpy.percentile(x, [25, 50, 75]), scipy.stats.describe(x)
qs, _ := stats.Percentiles(x, []float64{25, 50, 75}, nil)
d, _ := stats.Describe(x, nil)

// scipy.stats.gaussian_kde(data)(points)
k, _ := stats.NewGaussianKDE(data, nil)
dens, _ := k.EvaluateSlice(points)
```

Functions take `[]float64` or a go-ndarray `*ndarray.Array` wherever SciPy
takes an array; option structs have SciPy's defaults as their zero value (nil
is fine); results are structs; errors wrap `ErrParam`, `ErrData` or
`ErrFit`. Runnable versions with their output: [example_test.go](example_test.go).

## What is here

| | |
|---|---|
| Continuous (24) | Normal, Beta, Gamma, ChiSquared, Chi, InverseGamma, StudentT, F, NoncentralChiSquared, Exponential, Uniform, LogNormal, Cauchy, Laplace, Logistic, Weibull (weibull_min), Pareto, Reciprocal (= LogUniform), Gumbel (gumbel_r), Triangular, HalfNormal, Rayleigh, Lomax, VonMises — pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf, mean, var, std, skew, kurtosis, median, entropy, moment, interval, support, rvs, and FitXxx (closed forms bit-identical to SciPy's, otherwise maximum likelihood at least as good as SciPy's) |
| Discrete (8) | Bernoulli, Binomial, Poisson, Geometric, NegativeBinomial, Hypergeometric, DiscreteUniform (randint), Zipf — pmf, logpmf, cdf, logcdf, sf, logsf, ppf, isf, moments, entropy, rvs |
| Multivariate | MultivariateNormal (pdf, logpdf, cdf/logcdf by SciPy's lattice rule, exact bivariate, rvs, entropy, fit), Dirichlet, Multinomial |
| Descriptive | Describe, Mean/Var/Std/Average (weights), Median, Quantile(s)/Percentile(s) (NumPy's 13 methods, weights), Mode, Skew, Kurtosis, Moment, ZScore, ZMap, IQR, SEM, Variation, GMean, HMean, PMean, MedianAbsDeviation, TMean/TVar/TStd/TSem/TMin/TMax, TrimMean, TrimBoth, Trim1, Entropy (and relative entropy), DifferentialEntropy, Histogram, HistogramBinEdges (all bin rules), CumFreq, RelFreq, BinnedStatistic |
| Tests | TTest1Samp, TTestInd (Student, Welch, trimmed), TTestRel, ChiSquare, PowerDivergence, Chi2Contingency, FisherExact, Pearsonr, Spearmanr, Kendalltau, Linregress, RankData, KSTest1Samp, KSTest2Samp, MannWhitneyU, Wilcoxon, KruskalWallis, ShapiroWilk, AndersonDarling, Normaltest, SkewTest, KurtosisTest, JarqueBera, FOneway, Levene, Bartlett |
| Estimation | GaussianKDE (Scott/Silverman/scalar/function bandwidth, weights, evaluate, logpdf, integrals, resample), Bootstrap (percentile, basic, BCa) |

## Accuracy

Every distribution function is measured in standard form against mpmath (two
precisions, the exact value stored as a double-double) over regions crossing
small to huge shapes with both tails down to probabilities of 1e-300, next to
SciPy 1.18.1's value at the same arguments (x86-64 build). Per-region budgets
are enforced by `go test` on amd64, arm64, 386, arm, riscv64, loong64,
ppc64le and s390x.

- 256 fixtures, 1479 regions, ~212 000 points: ours ≤ SciPy in **1473**; the
  six exceptions are the beta sf/ppf, by at most 2 ulp more than SciPy (both
  under 4.5 ulp), from the incomplete beta function of go-numerics/special.
- Normal: every function ≤ 0.55 ulp. Gamma, chi, chi-squared, inverse gamma,
  beta and t densities ≤ 1.3 ulp where SciPy loses up to 10^7 ulp at large
  shapes; binomial, Poisson, negative binomial masses ≤ 0.5 ulp (SciPy up to
  10^15–10^16 ulp); many tails where SciPy returns 0, NaN or ±Inf.
- Tests and descriptive statistics: NumPy's means, variances, quantiles and
  histograms bit for bit; SciPy's test statistics and p-values to ≤ 1.4e-13,
  and more accurate where an exact reference says SciPy is not (Fisher exact,
  chi2_contingency near independence, exact KS and Mann-Whitney, Anderson-
  Darling Gumbel).

Details, the exceptions and the negative controls: [docs/accuracy.md](docs/accuracy.md);
every region: [docs/accuracy-table.md](docs/accuracy-table.md).

## Draws

`Rvs` makes the generator calls SciPy's `rvs` makes. Bit-identical to SciPy
(x86-64) for the same seed: normal, t, exponential, uniform, half-normal,
logistic, Rayleigh, triangular, noncentral chi-squared, all eight discrete
distributions, Dirichlet, multinomial and the bootstrap resamples. Gamma,
beta, chi, chi-squared, F, log-normal, Pareto, Weibull, Lomax, Gumbel,
Laplace and Reciprocal differ in a few draws per thousand by 1–4 ulp (the C
library's pow/exp/log in NumPy's samplers); Cauchy, inverse gamma and von
Mises differ more (SciPy's ppf-based draws use its own tan / inverse gamma;
von Mises wraps). Counts: [docs/rvs-table.md](docs/rvs-table.md).

## Speed

Every public operation against SciPy on the same data, core for core
([docs/perf.md](docs/perf.md), 355 cases): faster on one core for the
variances, moments, ranks, rank tests, KS, histograms and frequencies,
Spearman (2.2x), the 1-D Gaussian KDE, the closed-form fits and the small
tests; slower where the accuracy costs (double-double densities and
distribution functions, correctly rounded logarithms and powers per element,
exact inner products: Pearson 0.6x), where the time is in
go-numerics/special (the incomplete beta and gamma functions and their
inverses: t, beta and F quantiles at 0.005–0.04x), and in the numerical fits,
polished to about 1e-10 where SciPy stops at four digits. `Map` spreads the
elementwise functions over every core.

## Robustness

No argument a caller can pass crashes or stalls a distribution method:
every PDF/PMF, CDF, SF, their logarithms, PPF, ISF, the moments, Median,
Entropy and Interval of every distribution are swept over huge, tiny,
subnormal, signed-zero, NaN and infinite arguments at every fixture
instance (`TestExtremeInputsNoPanic`, in CI), with each parameter set to
extremes as well (`go test -run TestExtremeParamsNoPanic
-extreme.params`), and fuzzed (`FuzzDistributionMethods`). The cases the
sweeps found are pinned in `TestHangRegressions`. The one known slow call
is NoncentralChiSquared's quantile in the far tail at very large Nc
(ISF(1e-308) at Nc = 1e8 takes minutes); beyond Nc = 2·10^10 its
distribution functions return NaN.

## Design

[docs/design.md](docs/design.md).

## License

BSD-3-Clause.
