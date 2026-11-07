package stats

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/go-numerics/special/dd"
)

// Multiple-testing corrections: statsmodels.stats.multitest.multipletests
// (v0.15.0, BSD-3-Clause, Josef Perktold; methods bonferroni, sidak,
// holm-sidak, holm, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh,
// fdr_tsbky) and scipy.stats.false_discovery_control (SciPy 1.18.1).
//
// Every floating-point operation is performed in the order of the reference
// so that the adjusted p-values agree to the last bit: pow is evaluated
// correctly rounded (double-double internally; glibc's is within 0.52 ulp),
// log1p and expm1 are the msun routines glibc derives from (multitest_libm.go,
// without fused multiply-adds; sidak and holm-sidak can differ from glibc's
// FMA variants by one ulp on rare inputs), and the harmonic sum of fdr_by is
// NumPy's pairwise summation.
//
// Ties between equal p-values are ordered stably (the reference uses an
// unstable argsort, so the rejection of one member of a tie can differ from
// the reference's for n > 16 or on AVX-512 hosts; adjusted p-values do not
// depend on the order of ties).

// MultipleTestsResult is the outcome of MultipleTests, in the order of the
// input p-values.
type MultipleTestsResult struct {
	// Reject[i] is true when hypothesis i is rejected at the given alpha.
	Reject []bool
	// PAdj holds the adjusted p-values (clipped to 1).
	PAdj []float64
	// AlphaSidak = 1-(1-alpha)^(1/n) and AlphaBonf = alpha/n; NaN for n = 0.
	AlphaSidak, AlphaBonf float64
}

// multiMethods maps every statsmodels alias to its formal name.
var multiMethods = map[string]string{
	"b": "b", "bonf": "b", "bonferroni": "b",
	"s": "s", "sidak": "s",
	"h": "h", "holm": "h",
	"hs": "hs", "holm-sidak": "hs",
	"sh": "sh", "simes-hochberg": "sh",
	"ho": "ho", "hommel": "ho",
	"fdr_bh": "bh", "fdr_i": "bh", "fdr_p": "bh", "fdri": "bh", "fdrp": "bh",
	"fdr_by": "by", "fdr_n": "by", "fdr_c": "by", "fdrn": "by", "fdrcorr": "by",
	"fdr_tsbh": "tsbh", "fdr_2sbh": "tsbh",
	"fdr_tsbky": "tsbky", "fdr_2sbky": "tsbky", "fdr_twostage": "tsbky",
}

// checkPValues rejects NaN and values outside [0, 1].
func checkPValues(p []float64) error {
	for i, v := range p {
		if !(v >= 0 && v <= 1) {
			return fmt.Errorf("%w: p-value %d is %v, want a number in [0, 1]", ErrData, i, v)
		}
	}
	return nil
}

// sortedOrder returns the indices that sort p ascending (stable).
func sortedOrder(p []float64) []int {
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return p[idx[a]] < p[idx[b]] })
	return idx
}

// MultipleTests adjusts the p-values pvals for multiple testing at family
// level alpha, as statsmodels' multipletests(pvals, alpha, method, maxiter).
//
// method is any statsmodels name or alias, case-insensitive: "bonferroni"
// ("b", "bonf"), "sidak" ("s"), "holm-sidak" ("hs"), "holm" ("h"),
// "simes-hochberg" ("sh"), "hommel" ("ho"), "fdr_bh" ("fdr_i", "fdr_p",
// "fdri", "fdrp"), "fdr_by" ("fdr_n", "fdr_c", "fdrn", "fdrcorr"), "fdr_tsbh"
// ("fdr_2sbh") and "fdr_tsbky" ("fdr_2sbky", "fdr_twostage").
//
// maxiter bounds the iterations of the two-stage methods only: 1 is the
// statsmodels default (two stages), 0 a single stage, -1 full iteration
// (len(pvals)).
//
// It returns an error wrapping ErrParam for an unknown method or an alpha
// outside [0, 1] (or NaN), and ErrData for a p-value that is NaN or outside
// [0, 1] (statsmodels returns NaN or garbage there). Hommel costs O(n^2).
func MultipleTests(pvals []float64, alpha float64, method string, maxiter int) (MultipleTestsResult, error) {
	m, ok := multiMethods[strings.ToLower(method)]
	if !ok {
		return MultipleTestsResult{}, fmt.Errorf("%w: unknown multiple-testing method %q", ErrParam, method)
	}
	if !(alpha >= 0 && alpha <= 1) {
		return MultipleTestsResult{}, fmt.Errorf("%w: alpha is %v, want a number in [0, 1]", ErrParam, alpha)
	}
	if err := checkPValues(pvals); err != nil {
		return MultipleTestsResult{}, err
	}
	n := len(pvals)
	order := sortedOrder(pvals)
	p := make([]float64, n)
	for i, k := range order {
		p[i] = pvals[k]
	}
	res := MultipleTestsResult{AlphaSidak: math.NaN(), AlphaBonf: math.NaN()}
	if n > 0 {
		res.AlphaSidak = 1 - powCR(1-alpha, 1.0/float64(n))
		res.AlphaBonf = alpha / float64(n)
	}
	var rej []bool
	var adj []float64
	switch m {
	case "b":
		rej, adj = make([]bool, n), make([]float64, n)
		for i, v := range p {
			rej[i] = v <= res.AlphaBonf
			adj[i] = v * float64(n)
		}
	case "s":
		rej, adj = make([]bool, n), make([]float64, n)
		for i, v := range p {
			rej[i] = v <= res.AlphaSidak
			adj[i] = sidakAdj(float64(n), v)
		}
	case "hs":
		rej, adj = holmSidak(p, alpha)
	case "h":
		rej, adj = holm(p, alpha)
	case "sh":
		rej, adj = simesHochberg(p, alpha)
	case "ho":
		rej, adj = hommel(p, alpha)
	case "bh":
		rej, adj = fdrSorted(p, alpha, false)
	case "by":
		rej, adj = fdrSorted(p, alpha, true)
	case "tsbh":
		rej, adj, _ = fdrTwoStage(p, alpha, false, maxiter)
	default: // tsbky
		rej, adj, _ = fdrTwoStage(p, alpha, true, maxiter)
	}
	res.Reject, res.PAdj = make([]bool, n), make([]float64, n)
	for i, k := range order {
		res.Reject[k] = rej[i]
		if adj[i] > 1 {
			adj[i] = 1
		}
		res.PAdj[k] = adj[i]
	}
	return res, nil
}

// powCR returns x^y for 0 <= x <= 1 and y > 0, correctly rounded except in
// cases closer to a rounding boundary than about 2^-70.
func powCR(x, y float64) float64 {
	switch {
	case x == 0:
		return 0
	case y == 1 || x == 1:
		return x
	}
	return dd.Exp(dd.MulF(dd.Log(x), y))
}

// sidakAdj is the reference's -expm1(n * log1p(-p)) (libm routines in
// multitest_libm.go).
func sidakAdj(n, p float64) float64 { return -expm1FD(float64(n * log1pFD(-p))) }

func holmSidak(p []float64, alpha float64) (rej []bool, adj []float64) {
	n := len(p)
	rej, adj = make([]bool, n), make([]float64, n)
	notReject := false
	for i, v := range p {
		k := float64(n - i)
		if !notReject && v > 1-powCR(1-alpha, 1.0/k) {
			notReject = true
		}
		rej[i] = !notReject
		a := sidakAdj(k, v)
		if i > 0 && a < adj[i-1] {
			a = adj[i-1]
		}
		adj[i] = a
	}
	return rej, adj
}

func holm(p []float64, alpha float64) (rej []bool, adj []float64) {
	n := len(p)
	rej, adj = make([]bool, n), make([]float64, n)
	notReject := false
	for i, v := range p {
		k := float64(n - i)
		if !notReject && v > alpha/k {
			notReject = true
		}
		rej[i] = !notReject
		a := v * k
		if i > 0 && a < adj[i-1] {
			a = adj[i-1]
		}
		adj[i] = a
	}
	return rej, adj
}

func simesHochberg(p []float64, alpha float64) (rej []bool, adj []float64) {
	n := len(p)
	rej, adj = make([]bool, n), make([]float64, n)
	last := -1
	for i, v := range p {
		if v <= alpha/float64(n-i) {
			rej[i] = true
			last = i
		}
	}
	for i := 0; i < last; i++ {
		rej[i] = true
	}
	for i := n - 1; i >= 0; i-- {
		a := float64(n-i) * p[i]
		if i < n-1 && adj[i+1] < a {
			a = adj[i+1]
		}
		adj[i] = a
	}
	return rej, adj
}

func hommel(p []float64, alpha float64) (rej []bool, adj []float64) {
	n := len(p)
	a := append([]float64(nil), p...)
	for m := n; m > 1; m-- {
		fm := float64(m)
		cim := math.Inf(1)
		for k := 0; k < m; k++ {
			if v := fm * p[n-m+k] / float64(k+1); v < cim {
				cim = v
			}
		}
		for k := n - m; k < n; k++ {
			if a[k] < cim {
				a[k] = cim
			}
		}
		for k := 0; k < n-m; k++ {
			t := fm * p[k]
			if cim < t {
				t = cim
			}
			if a[k] < t {
				a[k] = t
			}
		}
	}
	rej = make([]bool, n)
	for i, v := range a {
		rej[i] = v <= alpha
	}
	return rej, a
}

// npSum is numpy.sum of a contiguous float64 array: pairwise summation with
// 8 accumulators between 8 and 128 items, halves (rounded to a multiple of 8)
// above. Not taken from NumPy's source but reconstructed from, and checked
// bit-for-bit against, 307 sums recorded from NumPy 2.5.4 (n up to 100000).
func npSum(a []float64) float64 { return npPairwise(a) }

func npPairwise(a []float64) float64 {
	n := len(a)
	switch {
	case n < 8:
		res := 0.0
		for _, v := range a {
			res += v
		}
		return res
	case n <= 128:
		var r [8]float64
		copy(r[:], a[:8])
		i := 8
		for ; i < n-n%8; i += 8 {
			for j := 0; j < 8; j++ {
				r[j] += a[i+j]
			}
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return npPairwise(a[:n2]) + npPairwise(a[n2:])
}

// harmonic returns numpy.sum(1.0 / numpy.arange(1, n+1)).
func harmonic(n int) float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1.0 / float64(i+1)
	}
	return npSum(v)
}

// fdrSorted is statsmodels fdrcorrection(is_sorted=True): Benjamini-Hochberg
// (by false) or Benjamini-Yekutieli.
func fdrSorted(p []float64, alpha float64, by bool) (rej []bool, adj []float64) {
	n := len(p)
	ecdf := make([]float64, n)
	for i := range ecdf {
		ecdf[i] = float64(i+1) / float64(n)
	}
	if by {
		cm := harmonic(n)
		for i := range ecdf {
			ecdf[i] /= cm
		}
	}
	rej, adj = make([]bool, n), make([]float64, n)
	last := -1
	for i, v := range p {
		if v <= ecdf[i]*alpha {
			rej[i] = true
			last = i
		}
	}
	for i := 0; i < last; i++ {
		rej[i] = true
	}
	for i := n - 1; i >= 0; i-- {
		a := p[i] / ecdf[i]
		if i < n-1 && adj[i+1] < a {
			a = adj[i+1]
		}
		adj[i] = a
		if adj[i] > 1 {
			adj[i] = 1
		}
	}
	return rej, adj
}

func countTrue(b []bool) int {
	c := 0
	for _, v := range b {
		if v {
			c++
		}
	}
	return c
}

// fdrTwoStage is statsmodels fdrcorrection_twostage(is_sorted=True): the
// two-stage Benjamini-Krieger-Yekutieli (bky) or Benjamini-Hochberg
// procedure. It also returns the number of estimated true hypotheses.
func fdrTwoStage(p []float64, alpha float64, bky bool, maxiter int) (rej []bool, adj []float64, m0 int) {
	n := len(p)
	if maxiter == -1 {
		maxiter = n
	}
	fact, alphaPrime := 1.0, alpha
	if bky {
		fact = 1.0 + alpha
		alphaPrime = alpha / fact
	}
	rej, adj = fdrSorted(p, alphaPrime, false)
	r1 := countTrue(rej)
	ri := r1
	if r1 == 0 || r1 == n {
		for i := range adj {
			adj[i] *= fact
		}
	} else {
		riOld := r1
		ntests0 := float64(n)
		for it := 0; it < maxiter; it++ {
			ntests0 = 1.0*float64(n) - float64(riOld)
			alphaStar := alphaPrime * float64(n) / ntests0
			rej, adj = fdrSorted(p, alphaStar, false)
			ri = countTrue(rej)
			if it >= maxiter-1 || ri == riOld {
				break
			}
			// ri < riOld cannot happen: alphaStar only grows, and the
			// rejection set of the step-up procedure grows with alpha.
			riOld = ri
		}
		for i := range adj {
			adj[i] *= ntests0 * 1.0 / float64(n)
			if bky {
				adj[i] *= 1.0 + alpha
			}
		}
	}
	for i := range adj {
		if adj[i] > 1 {
			adj[i] = 1
		}
	}
	return rej, adj, n - ri
}

// FalseDiscoveryControl adjusts p-values to control the false discovery rate,
// as scipy.stats.false_discovery_control(ps, method=method) on a 1-D array
// (SciPy 1.18.1): method "bh" (Benjamini-Hochberg) or "by"
// (Benjamini-Yekutieli), case-insensitive. The result is in the order of ps.
//
// The factors are formed as SciPy does, p * (m/i) (not p / (i/m) as in
// MultipleTests), so the two functions can differ in the last bit.
// A p-value that is NaN or outside [0, 1] gives an error wrapping ErrData
// (SciPy raises ValueError), an unknown method one wrapping ErrParam.
func FalseDiscoveryControl(ps []float64, method string) ([]float64, error) {
	meth := strings.ToLower(method)
	if meth != "bh" && meth != "by" {
		return nil, fmt.Errorf("%w: unknown false-discovery-control method %q, want \"bh\" or \"by\"", ErrParam, method)
	}
	if err := checkPValues(ps); err != nil {
		return nil, err
	}
	m := len(ps)
	out := append([]float64(nil), ps...)
	if m <= 1 {
		return out, nil
	}
	order := sortedOrder(ps)
	p := make([]float64, m)
	inv := make([]float64, m)
	for i, k := range order {
		p[i] = ps[k]
		inv[i] = 1 / float64(i+1)
	}
	var h float64
	if meth == "by" {
		h = npSum(inv)
	}
	for i := range p {
		p[i] *= float64(m) / float64(i+1)
		if meth == "by" {
			p[i] *= h
		}
	}
	for i := m - 2; i >= 0; i-- {
		if p[i+1] < p[i] {
			p[i] = p[i+1]
		}
	}
	for i, k := range order {
		v := p[i]
		if v > 1 {
			v = 1
		}
		out[k] = v
	}
	return out, nil
}
