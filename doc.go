// Package stats is probability distributions, descriptive statistics and
// statistical tests in pure Go (no cgo), with the results of scipy.stats
// and NumPy's statistics, measured against mpmath to be at least as accurate
// as SciPy in 1473 of 1479 regions (the exceptions, all beta sf and ppf
// within 4.5 ulp, are listed in docs/accuracy.md).
//
// # Distributions
//
// Every distribution is a struct of its parameters with SciPy's names and
// meaning, plus Loc and Scale (continuous) or Loc (discrete):
//
//	stats.Normal{Loc: 0, Scale: 1}.PDF(1.5)       // scipy.stats.norm.pdf(1.5)
//	stats.Beta{A: 2, B: 5, Scale: 1}.PPF(0.975)   // scipy.stats.beta.ppf(0.975, 2, 5)
//	stats.Binomial{N: 10, P: 0.3}.PMF(4)          // scipy.stats.binom.pmf(4, 10, 0.3)
//
// with the methods of a frozen SciPy distribution (Continuous, Discrete):
// PDF/PMF, LogPDF, CDF, LogCDF, SF, LogSF, PPF, ISF, Mean, Var, Std, Skew,
// Kurtosis, Median, Entropy, Moment, Interval, Support, Rvs and Validate.
// The zero value has Scale 0 and is invalid: methods then return NaN as
// SciPy does, and the NewXxx constructors return an error. Fits are
// FitXxx functions (scipy's xxx.fit, with FitOptions.Fixed for floc,
// fscale, fa ...). Map applies a method to a go-ndarray array in parallel.
//
// Continuous: Normal, Beta, Gamma, ChiSquared, Chi, InverseGamma, StudentT,
// F, NoncentralChiSquared, Exponential, Uniform, LogNormal, Cauchy, Laplace,
// Logistic, Weibull, Pareto, Reciprocal (LogUniform), Gumbel, Triangular,
// HalfNormal, Rayleigh, Lomax, VonMises. Discrete: Bernoulli, Binomial,
// Poisson, Geometric, NegativeBinomial, Hypergeometric, DiscreteUniform,
// Zipf. Multivariate: MultivariateNormal, Dirichlet, Multinomial.
//
// # Draws
//
// Rvs takes a go-ndarray *random.Generator, which reproduces NumPy's
// numpy.random.Generator bit for bit for the same seed, and makes the
// generator calls SciPy's rvs makes, so that
//
//	stats.Normal{Loc: 10, Scale: 2}.Rvs(random.DefaultRNG(42), 1000)
//
// is scipy.stats.norm.rvs(10, 2, size=1000,
// random_state=np.random.default_rng(42)) to the bit. docs/rvs-table.md
// lists, per distribution, how many draws are bit-identical to SciPy's
// (x86-64 build) and why the others differ.
//
// # Descriptive statistics and tests
//
// The functions take Data: a []float64 or a go-ndarray *ndarray.Array.
// Options are structs whose zero value (or nil) is SciPy's default; results
// are structs (Statistic, PValue, DF, confidence intervals where SciPy
// gives them). Quantiles and histograms are NumPy's to the bit, for all
// thirteen quantile methods.
//
// # Accuracy
//
// Every distribution function is measured in ULPs against mpmath over
// regions covering the bulk and both tails, next to SciPy's value at the
// same arguments; per-region budgets are enforced by go test on every CI
// architecture. See docs/accuracy.md.
package stats
