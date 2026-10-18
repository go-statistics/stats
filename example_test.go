package stats_test

import (
	"fmt"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats"
)

// scipy.stats.norm.pdf(1.5), norm.logcdf(-40), norm.ppf(0.975)
func Example_normal() {
	n := stats.Normal{Loc: 0, Scale: 1}
	fmt.Printf("%.17g\n%.17g\n%.17g\n", n.PDF(1.5), n.LogCDF(-40), n.PPF(0.975))
	// Output:
	// 0.12951759566589172
	// -804.6084420137538
	// 1.9599639845400538
}

// scipy.stats.beta(2.5, 7).ppf([0.05, 0.5, 0.95]) through an array
func Example_array() {
	p, _ := ndarray.FromData([]float64{0.05, 0.5, 0.95}, 3)
	q := stats.Map(p, stats.Beta{A: 2.5, B: 7, Scale: 1}.PPF)
	fmt.Println(q)
	// Output:
	// Array(shape=[3], data=[0.07152999268253016 0.24603732885541799 0.51374106221438])
}

// scipy.stats.norm.rvs(loc=10, scale=2, size=3, random_state=np.random.default_rng(42))
func Example_rvs() {
	x, _ := stats.Normal{Loc: 10, Scale: 2}.Rvs(random.DefaultRNG(42), 3)
	fmt.Println(x)
	// Output:
	// Array(shape=[3], data=[10.609434159508863 7.920031787519009 11.500902391612914])
}

// scipy.stats.ttest_ind(a, b, equal_var=False)
func Example_tTest() {
	a := []float64{5.1, 4.9, 5.6, 5.8, 6.0, 5.3}
	b := []float64{4.2, 4.8, 4.6, 5.0, 4.4}
	r, _ := stats.TTestInd(a, b, &stats.TTestIndOptions{Welch: true})
	ci, _ := r.ConfidenceInterval(0.95)
	fmt.Printf("t=%.6f p=%.6f df=%.4f ci=[%.4f, %.4f]\n", r.Statistic, r.PValue, r.DF, ci.Low, ci.High)
	// Output:
	// t=3.807667 p=0.004226 df=8.9328 ci=[0.3444, 1.3556]
}

// scipy.stats.gaussian_kde(data)(points)
func Example_gaussianKDE() {
	data := []float64{-1.2, -0.4, 0.1, 0.3, 0.9, 1.7}
	k, _ := stats.NewGaussianKDE(data, nil)
	d, _ := k.EvaluateSlice([]float64{0, 1})
	fmt.Printf("%.6f %.6f\n", d[0], d[1])
	// Output:
	// 0.329053 0.264227
}

// numpy.percentile(x, [25, 50, 75]) and scipy.stats.describe(x)
func Example_describe() {
	x := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	q, _ := stats.Percentiles(x, []float64{25, 50, 75}, nil)
	d, _ := stats.Describe(x, nil)
	fmt.Println(q, d.Mean, d.Variance)
	// Output:
	// [4 4.5 5.5] 5 4.571428571428571
}

// scipy.stats.gamma.fit(x, floc=0)
func Example_fit() {
	x, _ := stats.Gamma{A: 3, Scale: 2}.Rvs(random.DefaultRNG(1), 500)
	g, _ := stats.FitGamma(x, &stats.FitOptions{Fixed: map[string]float64{"Loc": 0}})
	fmt.Printf("a=%.4f scale=%.4f\n", g.A, g.Scale)
	// Output:
	// a=3.3544 scale=1.7401
}
