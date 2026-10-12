package stats

import (
	"flag"
	"fmt"
	"math"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
)

// The rest of the public surface in the SciPy speed comparison: every
// distribution's elementwise functions and draws (generic cases, driven by
// "@dist" / "@rvs" / "@entropy" commands to bench/scipy_server.py), and every
// descriptive statistic, test, fit and multivariate operation (named cases,
// mirrored in the server's CASES).

// perfCause says, for a row still slower than SciPy on one core, where the
// time goes (filled from profiles; see docs/perf.md).
var perfCause = map[string]string{}

func perfNote(name string, ratio float64) string {
	if ratio >= 1 {
		return " |"
	}
	if c, ok := perfCause[name]; ok {
		return " " + c + " |"
	}
	if i := strings.IndexByte(name, '.'); i > 0 {
		if c, ok := perfCause[name[i:]]; ok {
			return " " + c + " |"
		}
	}
	return " |"
}

// perfDist is one distribution of the generic cases: the Go value, SciPy's
// name and shape arguments, the x range of the elementwise cases, and
// whether it is discrete (x is then floor(points)).
type perfDist struct {
	d        any
	name     string
	args     []float64
	lo, hi   float64
	discrete bool
}

var perfDists = []perfDist{
	{Beta{A: 2.5, B: 7, Scale: 1}, "beta", []float64{2.5, 7}, 1e-6, 1 - 1e-6, false},
	{Cauchy{Scale: 1}, "cauchy", nil, -50, 50, false},
	{Chi{Df: 4.5, Scale: 1}, "chi", []float64{4.5}, 1e-3, 6, false},
	{ChiSquared{Df: 12, Scale: 1}, "chi2", []float64{12}, 1e-3, 100, false},
	{Exponential{Scale: 1}, "expon", nil, 0, 20, false},
	{F{Dfn: 5, Dfd: 12, Scale: 1}, "f", []float64{5, 12}, 1e-3, 10, false},
	{Gamma{A: 4.5, Scale: 1}, "gamma", []float64{4.5}, 1e-3, 30, false},
	{Gumbel{Scale: 1}, "gumbel_r", nil, -3, 10, false},
	{HalfNormal{Scale: 1}, "halfnorm", nil, 0, 6, false},
	{InverseGamma{A: 3.5, Scale: 1}, "invgamma", []float64{3.5}, 1e-3, 10, false},
	{Laplace{Scale: 1}, "laplace", nil, -15, 15, false},
	{Logistic{Scale: 1}, "logistic", nil, -20, 20, false},
	{LogNormal{S: 0.8, Scale: 1}, "lognorm", []float64{0.8}, 1e-3, 20, false},
	{Lomax{C: 2.5, Scale: 1}, "lomax", []float64{2.5}, 0, 30, false},
	{NoncentralChiSquared{Df: 4, Nc: 3.5, Scale: 1}, "ncx2", []float64{4, 3.5}, 1e-3, 40, false},
	{Normal{Scale: 1}, "norm", nil, -10, 10, false},
	{Pareto{B: 2.5, Scale: 1}, "pareto", []float64{2.5}, 1, 50, false},
	{Rayleigh{Scale: 1}, "rayleigh", nil, 0, 6, false},
	{Reciprocal{A: 0.5, B: 20, Scale: 1}, "reciprocal", []float64{0.5, 20}, 0.5, 20, false},
	{StudentT{Df: 7.5, Scale: 1}, "t", []float64{7.5}, -20, 20, false},
	{Triangular{C: 0.3, Scale: 1}, "triang", []float64{0.3}, 0, 1, false},
	{Uniform{Scale: 1}, "uniform", nil, 0, 1, false},
	{VonMises{Kappa: 2.5, Scale: 1}, "vonmises", []float64{2.5}, -math.Pi, math.Pi, false},
	{Weibull{C: 1.7, Scale: 1}, "weibull_min", []float64{1.7}, 0, 6, false},
	{Bernoulli{P: 0.3}, "bernoulli", []float64{0.3}, 0, 2, true},
	{Binomial{N: 30, P: 0.3}, "binom", []float64{30, 0.3}, 0, 31, true},
	{DiscreteUniform{Low: 0, High: 50}, "randint", []float64{0, 50}, -1, 51, true},
	{Geometric{P: 0.2}, "geom", []float64{0.2}, 1, 60, true},
	{Hypergeometric{Total: 100, Good: 30, Draws: 20}, "hypergeom", []float64{100, 30, 20}, 0, 21, true},
	{NegativeBinomial{N: 5.5, P: 0.4}, "nbinom", []float64{5.5, 0.4}, 0, 60, true},
	{Poisson{Mu: 7.5}, "poisson", []float64{7.5}, 0, 30, true},
	{Zipf{A: 2.5}, "zipf", []float64{2.5}, 1, 200, true},
}

func pyArgs(a []float64) string {
	if len(a) == 0 {
		return "-"
	}
	s := make([]string, len(a))
	for i, v := range a {
		s[i] = fmt.Sprint(v)
	}
	return strings.Join(s, ",")
}

type rvser interface {
	Rvs(g *random.Generator, size ...int) (*ndarray.Array, error)
}

func init() {
	const nx, nq = 100_000, 20_000
	for _, pd := range perfDists {
		var methods map[string]func(float64) float64
		switch d := pd.d.(type) {
		case Continuous:
			methods = map[string]func(float64) float64{
				"pdf": d.PDF, "logpdf": d.LogPDF, "cdf": d.CDF, "sf": d.SF,
				"ppf": d.PPF, "isf": d.ISF,
			}
		case Discrete:
			methods = map[string]func(float64) float64{
				"pmf": d.PMF, "logpmf": d.LogPMF, "cdf": d.CDF, "sf": d.SF,
				"ppf": d.PPF, "isf": d.ISF,
			}
		}
		for m, f := range methods {
			name := pd.name + "." + m
			if _, ok := perfCases[name]; ok {
				continue // a hand-written case (larger n) already
			}
			n, lo, hi, disc := nx, pd.lo, pd.hi, pd.discrete
			if m == "ppf" || m == "isf" {
				n, lo, hi, disc = nq, 1e-9, 1-1e-9, false
			}
			c := elementwise(n, lo, hi, f)
			if disc {
				c = elementwiseFloor(n, lo, hi, f)
			}
			c.py = fmt.Sprintf("@dist %s %s %v %v %d %s", pd.name, m, lo, hi, b2i(disc), pyArgs(pd.args))
			perfCases[name] = c
		}
		name := pd.name + ".rvs"
		if _, ok := perfCases[name]; !ok {
			r := pd.d.(rvser)
			perfCases[name] = perfCase{n: 1_000_000, py: fmt.Sprintf("@rvs %s %s", pd.name, pyArgs(pd.args)),
				setup: func(n int) (func(), func()) {
					return nil, func() { r.Rvs(random.DefaultRNG(1), n) }
				}}
		}
		ent := pd.d.(interface{ Entropy() float64 })
		perfCases[pd.name+".entropy"] = perfCase{n: 1, py: fmt.Sprintf("@entropy %s %s", pd.name, pyArgs(pd.args)),
			setup: func(int) (func(), func()) {
				return nil, func() {
					for range 100 {
						ent.Entropy()
					}
				}
			}}
	}
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

func elementwiseFloor(n int, lo, hi float64, f func(float64) float64) perfCase {
	c := elementwise(n, lo, hi, f)
	c.setup = func(n int) (func(), func()) {
		x := points(n, lo, hi)
		for i := range x {
			x[i] = math.Floor(x[i])
		}
		xa, _ := ndarray.FromData(x, n)
		out := make([]float64, n)
		return func() {
				for i, v := range x {
					out[i] = f(v)
				}
			}, func() {
				Map(xa, f)
			}
	}
	return c
}

// fcase is a non-elementwise case: setup returns the call a user makes.
func fcase(n int, setup func(n int) func()) perfCase {
	return perfCase{n: n, setup: func(n int) (func(), func()) { return nil, setup(n) }}
}

func positive(n int, seed uint64) []float64 {
	x := normalDraws(n, seed)
	for i := range x {
		x[i] = math.Abs(x[i]) + 0.5
	}
	return x
}

func counts(n int) []float64 {
	x := points(n, 50, 150)
	for i := range x {
		x[i] = math.Floor(x[i])
	}
	return x
}

func init() {
	const m, k = 1_000_000, 100_000
	cs := map[string]perfCase{
		"mean":     fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Mean(a, nil) } }),
		"var":      fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Var(a, nil) } }),
		"std":      fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Std(a, nil) } }),
		"median":   fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Median(a, nil) } }),
		"quantile": fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Quantile(a, 0.3, nil) } }),
		"average": fcase(m, func(n int) func() {
			a, w := normalDraws(n, 1), positive(n, 2)
			return func() { Average(a, &MeanOptions{Weights: w}) }
		}),
		"moment":   fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Moment(a, 3, nil) } }),
		"skew":     fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Skew(a, nil) } }),
		"kurtosis": fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Kurtosis(a, nil) } }),
		"sem":      fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { SEM(a, nil) } }),
		"variation": fcase(m, func(n int) func() {
			a := positive(n, 1)
			return func() { Variation(a, nil) }
		}),
		"zscore": fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { ZScore(a, nil) } }),
		"zmap": fcase(m, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { ZMap(a, b, nil) }
		}),
		"gmean": fcase(m, func(n int) func() { a := positive(n, 1); return func() { GMean(a, nil) } }),
		"hmean": fcase(m, func(n int) func() { a := positive(n, 1); return func() { HMean(a, nil) } }),
		"pmean": fcase(m, func(n int) func() { a := positive(n, 1); return func() { PMean(a, 2.5, nil) } }),
		"iqr":   fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { IQR(a, nil) } }),
		"mad":   fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { MedianAbsDeviation(a, nil) } }),
		"mode": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			for i := range a {
				a[i] = math.RoundToEven(a[i] * 10)
			}
			return func() { Mode(a, nil) }
		}),
		"tmean": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { TMean(a, &TrimmedOptions{Limits: &Limits{Lower: -1, Upper: 1}}) }
		}),
		"tvar": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { TVar(a, &TrimmedOptions{Limits: &Limits{Lower: -1, Upper: 1}}) }
		}),
		"tstd": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { TStd(a, &TrimmedOptions{Limits: &Limits{Lower: -1, Upper: 1}}) }
		}),
		"tsem": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { TSem(a, &TrimmedOptions{Limits: &Limits{Lower: -1, Upper: 1}}) }
		}),
		"tmin":      fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { TMin(a, -1, nil) } }),
		"tmax":      fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { TMax(a, 1, nil) } }),
		"trim_mean": fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { TrimMean(a, 0.1) } }),
		"trimboth":  fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { TrimBoth(a, 0.1) } }),
		"trim1":     fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Trim1(a, 0.1, TailRight) } }),
		"rankdata":  fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { RankData(a, RankAverage) } }),
		"histogram": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { Histogram(a, &HistogramOptions{Bins: 100}) }
		}),
		"histogram_bin_edges": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { HistogramBinEdges(a, &HistogramOptions{Rule: BinsAuto}) }
		}),
		"cumfreq": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { CumFreq(a, &FreqOptions{NumBins: 100}) }
		}),
		"relfreq": fcase(m, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { RelFreq(a, &FreqOptions{NumBins: 100}) }
		}),
		"binned_statistic": fcase(m, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { BinnedStatistic(a, b, &BinnedOptions{Statistic: BinMean, Bins: 50}) }
		}),
		"entropy": fcase(m, func(n int) func() { a := positive(n, 1); return func() { Entropy(a, nil) } }),
		"differential_entropy": fcase(k, func(n int) func() {
			a := normalDraws(n, 1)
			return func() { DifferentialEntropy(a, nil) }
		}),
		"ttest_rel": fcase(m, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { TTestRel(a, b, nil) }
		}),
		"linregress": fcase(m, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { Linregress(a, b, nil) }
		}),
		"kendalltau": fcase(k, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { Kendalltau(a, b, nil) }
		}),
		"mannwhitneyu": fcase(k, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { MannWhitneyU(a, b, nil) }
		}),
		"wilcoxon": fcase(k, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { Wilcoxon(a, b, nil) }
		}),
		"kruskal": fcase(k, func(n int) func() {
			a, b, c := normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)
			return func() { KruskalWallis(a, b, c) }
		}),
		"f_oneway": fcase(m, func(n int) func() {
			a, b, c := normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)
			return func() { FOneway(nil, a, b, c) }
		}),
		"bartlett": fcase(m, func(n int) func() {
			a, b, c := normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)
			return func() { Bartlett(a, b, c) }
		}),
		"levene": fcase(m, func(n int) func() {
			a, b, c := normalDraws(n, 1), normalDraws(n, 2), normalDraws(n, 3)
			return func() { Levene(nil, a, b, c) }
		}),
		"ks_1samp": fcase(k, func(n int) func() {
			a := normalDraws(n, 1)
			cdf := Normal{Scale: 1}.CDF
			return func() { KSTest1Samp(a, cdf, nil) }
		}),
		"ks_2samp": fcase(k, func(n int) func() {
			a, b := normalDraws(n, 1), normalDraws(n, 2)
			return func() { KSTest2Samp(a, b, nil) }
		}),
		"shapiro":      fcase(5000, func(n int) func() { a := normalDraws(n, 1); return func() { ShapiroWilk(a) } }),
		"anderson":     fcase(k, func(n int) func() { a := normalDraws(n, 1); return func() { AndersonDarling(a, ADNorm) } }),
		"skewtest":     fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { SkewTest(a, nil) } }),
		"kurtosistest": fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { KurtosisTest(a, nil) } }),
		"normaltest":   fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { Normaltest(a) } }),
		"jarque_bera":  fcase(m, func(n int) func() { a := normalDraws(n, 1); return func() { JarqueBera(a) } }),
		"chisquare":    fcase(m, func(n int) func() { a := counts(n); return func() { ChiSquare(a, nil) } }),
		"power_divergence": fcase(m, func(n int) func() {
			a := counts(n)
			return func() { PowerDivergence(a, 2.0/3, nil) }
		}),
		"chi2_contingency": fcase(10_000, func(n int) func() {
			t, _ := ndarray.FromData(counts(n), 100, n/100)
			return func() { Chi2Contingency(t, nil) }
		}),
		"fisher_exact": fcase(1, func(int) func() {
			return func() { FisherExact([2][2]int64{{1200, 800}, {900, 1100}}, nil) }
		}),
		"bootstrap": fcase(1000, func(n int) func() {
			a := normalDraws(n, 1)
			mean := func(s ...[]float64) float64 { v, _ := Mean(s[0], nil); return v }
			return func() { Bootstrap([][]float64{a}, mean, random.DefaultRNG(3), nil) }
		}),
		"kde.logpdf": fcase(10_000, func(n int) func() {
			d, x := normalDraws(n, 1), points(n, -4, 4)
			return func() {
				k, _ := NewGaussianKDE(d, nil)
				k.LogPDFSlice(x)
			}
		}),
		"kde.evaluate2d": fcase(10_000, func(n int) func() {
			d := append(normalDraws(n, 1), normalDraws(n, 2)...)
			da, _ := ndarray.FromData(d, 2, n)
			x := append(points(n, -4, 4), points(n, -3, 3)...)
			xa, _ := ndarray.FromData(x, 2, n)
			return func() {
				k, _ := NewGaussianKDE(da, nil)
				k.Evaluate(xa)
			}
		}),
		"kde.resample": fcase(100_000, func(n int) func() {
			d := normalDraws(10_000, 1)
			k, _ := NewGaussianKDE(d, nil)
			return func() { k.Resample(random.DefaultRNG(4), n) }
		}),
		"kde.integrate_box_1d": fcase(10_000, func(n int) func() {
			d := normalDraws(n, 1)
			k, _ := NewGaussianKDE(d, nil)
			return func() { k.IntegrateBox1D(-1, 0.5) }
		}),
		"mvn.logpdf": fcase(k, func(n int) func() {
			mv, x := perfMVN(n)
			return func() { mv.LogPDFPoints(x) }
		}),
		"mvn.rvs": fcase(k, func(n int) func() {
			mv, _ := perfMVN(1)
			return func() { mv.Rvs(random.DefaultRNG(5), n) }
		}),
		"mvn.cdf": fcase(1, func(int) func() {
			mv, _ := perfMVN(1)
			x := []float64{0.5, 0.2, 1, -0.3, 0.8}
			return func() { mv.CDF(x, nil) }
		}),
		"dirichlet.logpdf": fcase(k, func(n int) func() {
			d, _ := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
			x, _ := d.Rvs(random.DefaultRNG(6), n)
			return func() { d.LogPDFPoints(x) }
		}),
		"dirichlet.rvs": fcase(k, func(n int) func() {
			d, _ := NewDirichlet([]float64{1.5, 2, 3.5, 0.8, 4})
			return func() { d.Rvs(random.DefaultRNG(6), n) }
		}),
		"multinomial.logpmf": fcase(k, func(n int) func() {
			d, _ := NewMultinomial(20, []float64{0.1, 0.2, 0.3, 0.15, 0.25})
			x, _ := d.Rvs(random.DefaultRNG(6), n)
			return func() { d.LogPMFPoints(x) }
		}),
		"multinomial.rvs": fcase(k, func(n int) func() {
			d, _ := NewMultinomial(20, []float64{0.1, 0.2, 0.3, 0.15, 0.25})
			return func() { d.Rvs(random.DefaultRNG(6), n) }
		}),
		"mvn.fit": fcase(k, func(n int) func() {
			_, x := perfMVN(n)
			return func() { FitMultivariateNormal(x, nil) }
		}),
	}
	for name, c := range cs {
		perfCases[name] = c
	}
	// Fits: data drawn by the distribution itself with seed 7 (bit-identical
	// to SciPy's rvs for these), n = 10^4.
	fits := map[string]func(x []float64){
		"beta":        func(x []float64) { FitBeta(x, nil) },
		"cauchy":      func(x []float64) { FitCauchy(x, nil) },
		"chi":         func(x []float64) { FitChi(x, nil) },
		"chi2":        func(x []float64) { FitChiSquared(x, nil) },
		"expon":       func(x []float64) { FitExponential(x, nil) },
		"f":           func(x []float64) { FitF(x, nil) },
		"gamma":       func(x []float64) { FitGamma(x, nil) },
		"gumbel_r":    func(x []float64) { FitGumbel(x, nil) },
		"halfnorm":    func(x []float64) { FitHalfNormal(x, nil) },
		"invgamma":    func(x []float64) { FitInverseGamma(x, nil) },
		"laplace":     func(x []float64) { FitLaplace(x, nil) },
		"logistic":    func(x []float64) { FitLogistic(x, nil) },
		"lognorm":     func(x []float64) { FitLogNormal(x, nil) },
		"lomax":       func(x []float64) { FitLomax(x, nil) },
		"norm":        func(x []float64) { FitNormal(x, nil) },
		"pareto":      func(x []float64) { FitPareto(x, nil) },
		"rayleigh":    func(x []float64) { FitRayleigh(x, nil) },
		"reciprocal":  func(x []float64) { FitReciprocal(x, nil) },
		"t":           func(x []float64) { FitStudentT(x, nil) },
		"triang":      func(x []float64) { FitTriangular(x, nil) },
		"uniform":     func(x []float64) { FitUniform(x, nil) },
		"vonmises":    func(x []float64) { FitVonMises(x, nil) },
		"weibull_min": func(x []float64) { FitWeibull(x, nil) },
	}
	for _, pd := range perfDists {
		fit, ok := fits[pd.name]
		if !ok {
			continue
		}
		r := pd.d.(rvser)
		perfCases[pd.name+".fit"] = perfCase{n: 10_000, py: fmt.Sprintf("@fit %s %s", pd.name, pyArgs(pd.args)),
			setup: func(n int) (func(), func()) {
				a, _ := r.Rvs(random.DefaultRNG(7), n)
				x, _ := ndarray.Data[float64](a)
				return nil, func() { fit(x) }
			}}
	}
}

// perfMVN is the 5-dimensional normal of the multivariate cases and n
// points drawn from it (seed 8).
func perfMVN(n int) (*MultivariateNormal, *ndarray.Array) {
	cov, _ := ndarray.FromData([]float64{
		2, 0.5, 0.3, 0.1, 0,
		0.5, 1.5, 0.2, 0, 0.1,
		0.3, 0.2, 1, 0.4, 0.2,
		0.1, 0, 0.4, 2.5, 0.3,
		0, 0.1, 0.2, 0.3, 1.2,
	}, 5, 5)
	mv, _ := NewMultivariateNormal([]float64{0, 1, -1, 0.5, 2}, cov, nil)
	x, _ := mv.Rvs(random.DefaultRNG(8), n)
	return mv, x
}

var perfProfile = flag.String("perf.profile", "", "run this case on one core for -perf.seconds (with -cpuprofile)")
var perfSeconds = flag.Float64("perf.seconds", 3, "duration of -perf.profile")

// TestPerfProfile runs one case alone, on one core, for profiling:
//
//	go test -run TestPerfProfile -perf.profile=pearsonr -cpuprofile=cpu.out
func TestPerfProfile(t *testing.T) {
	c, ok := perfCases[*perfProfile]
	if !ok {
		t.Skip("set -perf.profile=CASE")
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	_, par := c.setup(c.n)
	t0 := time.Now()
	k := 0
	for ; time.Since(t0).Seconds() < *perfSeconds || k == 0; k++ {
		par()
	}
	t.Logf("%s: %d calls, %.4g ms each", *perfProfile, k, time.Since(t0).Seconds()*1e3/float64(k))
}
