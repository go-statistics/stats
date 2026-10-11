package stats

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-numerics/special"
)

// KSOptions are the options of KSTest1Samp and KSTest2Samp. The zero
// value (or nil) is SciPy's default: two-sided, method "auto".
type KSOptions struct {
	// Alternative: TwoSided compares the distributions for equality; Less
	// is the alternative that the CDF of the (first) sample lies below the
	// hypothesised (second) one somewhere (statistic D-); Greater that it
	// lies above (statistic D+).
	Alternative Alternative
	// Method selects the p-value: MethodAuto, MethodExact,
	// MethodAsymptotic or (one-sample, two-sided only) MethodApprox.
	Method Method
}

// KSResult is the outcome of a Kolmogorov-Smirnov test (SciPy's
// KstestResult).
type KSResult struct {
	Statistic, PValue float64
	// StatisticLocation is the observation at which the statistic is
	// attained (statistic_location).
	StatisticLocation float64
	// StatisticSign is +1 when the statistic is the maximum of the
	// (first) empirical CDF minus the other CDF, -1 when it is the
	// maximum of the opposite difference (statistic_sign); 0 when the
	// result is NaN.
	StatisticSign int
}

var nanKS = KSResult{qnan, qnan, qnan, 0}

// KSTest1Samp is the one-sample Kolmogorov-Smirnov test of the null
// hypothesis that x is a sample of the continuous distribution whose CDF
// is cdf (a distribution's CDF method value, e.g. Normal{Scale: 1}.CDF,
// or any function): scipy.stats.ks_1samp(x, cdf, alternative, method).
//
// The p-value of the two-sided test comes from the exact distribution of
// D_n (scipy.stats.kstwo) with MethodAuto and MethodExact, from its limit
// (scipy.stats.kstwobign at D·sqrt(n)) with MethodAsymptotic, and from
// twice the one-sided p-value with MethodApprox; one-sided tests always
// use the exact distribution of D+_n (scipy.stats.ksone).
func KSTest1Samp[D Data](x D, cdf func(float64) float64, o *KSOptions) (KSResult, error) {
	if o == nil {
		o = &KSOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanKS, err
	}
	if o.Method < MethodAuto || o.Method > MethodApprox {
		return nanKS, fmt.Errorf("%w: ks_1samp method %v", ErrParam, o.Method)
	}
	v := floats(x)
	if len(v) == 0 || hasNaN(v) {
		return nanKS, nil
	}
	s := slices.Clone(v)
	slices.Sort(s)
	n := float64(len(s))
	dplus, lplus := math.Inf(-1), qnan
	dminus, lminus := math.Inf(-1), qnan
	for i, xi := range s {
		c := cdf(xi)
		if d := float64(i+1)/n - c; d > dplus {
			dplus, lplus = d, xi
		}
		if d := c - float64(i)/n; d > dminus {
			dminus, lminus = d, xi
		}
	}
	N := len(s)
	switch o.Alternative {
	case Greater:
		return KSResult{dplus, smirnovSF(N, dplus), lplus, 1}, nil
	case Less:
		return KSResult{dminus, smirnovSF(N, dminus), lminus, -1}, nil
	}
	d, loc, sign := dminus, lminus, -1
	if dplus > dminus {
		d, loc, sign = dplus, lplus, 1
	}
	var p float64
	switch o.Method {
	case MethodAsymptotic:
		p = kolmogorovSF(d * math.Sqrt(n))
	case MethodApprox:
		p = 2 * smirnovSF(N, d)
	default:
		p = kstwoSF(N, d)
	}
	return KSResult{d, clip01(p), loc, sign}, nil
}

// KSTest2Samp is the two-sample Kolmogorov-Smirnov test of the null
// hypothesis that a and b are samples of the same continuous
// distribution: scipy.stats.ks_2samp(a, b, alternative, method).
//
// With MethodAuto the p-value is exact when both samples have at most
// 10000 observations, asymptotic otherwise. The exact p-values count the
// lattice paths staying inside (or crossing) the boundary (Hodges 1958;
// Viehmann 2021), as SciPy does; when the exact computation is not
// possible (lcm(n1, n2) too large, or overflow) the asymptotic one is used
// instead, as SciPy does with a warning. The asymptotic two-sided p-value
// is kstwo's survival function at round(n1·n2/(n1+n2)), the one-sided one
// Hodges' approximation (his eq. 5.3).
func KSTest2Samp[D Data](a, b D, o *KSOptions) (KSResult, error) {
	if o == nil {
		o = &KSOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanKS, err
	}
	if o.Method < MethodAuto || o.Method > MethodAsymptotic {
		return nanKS, fmt.Errorf("%w: ks_2samp method %v", ErrParam, o.Method)
	}
	x, y := floats(a), floats(b)
	if len(x) == 0 || len(y) == 0 || hasNaN(x, y) {
		return nanKS, nil
	}
	s1, s2 := slices.Clone(x), slices.Clone(y)
	slices.Sort(s1)
	slices.Sort(s2)
	n1, n2 := len(s1), len(s2)
	// The ECDF values as np.linspace(0, 1, n+1) forms them: k·(1/n), the
	// last one exactly 1.
	step1, step2 := 1/float64(n1), 1/float64(n2)
	ecdf := func(k, n int, step float64) float64 {
		if k == n {
			return 1
		}
		return float64(k) * step
	}
	maxS, minD := math.Inf(-1), math.Inf(1)
	var locMax, locMin float64
	i, j := 0, 0
	for i < n1 || j < n2 {
		var v float64
		switch {
		case j == n2 || (i < n1 && s1[i] <= s2[j]):
			v = s1[i]
		default:
			v = s2[j]
		}
		for i < n1 && s1[i] == v {
			i++
		}
		for j < n2 && s2[j] == v {
			j++
		}
		diff := ecdf(i, n1, step1) - ecdf(j, n2, step2)
		if diff > maxS {
			maxS, locMax = diff, v
		}
		if diff < minD {
			minD, locMin = diff, v
		}
	}
	minS := clipTo(-minD, 0, 1)
	var d, loc float64
	sign := 1
	switch {
	case o.Alternative == Less || (o.Alternative == TwoSided && minS > maxS):
		d, loc, sign = minS, locMin, -1
	default:
		d, loc = maxS, locMax
	}
	return KSResult{d, ks2Prob(d, n1, n2, o.Method, o.Alternative), loc, sign}, nil
}

// ks2Prob is SciPy's _ks_2samp_prob.
func ks2Prob(d float64, n1, n2 int, method Method, alt Alternative) float64 {
	g := gcdInt(n1, n2)
	n1g, n2g := n1/g, n2/g
	switch method {
	case MethodAuto:
		method = MethodAsymptotic
		if max(n1, n2) <= 10000 {
			method = MethodExact
		}
	case MethodExact:
		if float64(n1g) >= float64(math.MaxInt32)/float64(n2g) {
			method = MethodAsymptotic
		}
	}
	if method == MethodExact {
		if p, ok := ks2Exact(n1, n2, g, d, alt); ok {
			return clip01(p)
		}
	}
	m, n := float64(max(n1, n2)), float64(min(n1, n2))
	en := m * n / (m + n)
	if alt == TwoSided {
		return clip01(kstwoSF(int(math.RoundToEven(en)), d))
	}
	z := math.Sqrt(en) * d
	expt := -2*z*z - 2*z*(m+2*n)/math.Sqrt(m*n*(m+n))/3.0
	return clip01(math.Exp(expt))
}

func gcdInt(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// ks2Exact is SciPy's _attempt_exact_2kssamp: the exact p-value, and
// whether it could be computed.
func ks2Exact(n1, n2, g int, d float64, alt Alternative) (float64, bool) {
	lcm := (n1 / g) * n2
	h := int(math.RoundToEven(d * float64(lcm)))
	if h == 0 {
		return 1, true
	}
	var prob float64
	switch {
	case alt == TwoSided && n1 == n2:
		prob = probOutsideSquare(n1, h)
	case alt == TwoSided:
		prob = outerProbInside(n1, n2, g, h)
	case n1 == n2:
		prob = 1
		for j := 0; j < h; j++ {
			prob *= float64(n1-j) / (float64(n1) + float64(j) + 1)
		}
	default:
		paths := countPathsOutside(n1, n2, g, h)
		bin := special.Binom(float64(n1+n2), float64(n1))
		if paths > bin || math.IsInf(bin, 0) {
			return qnan, false
		}
		prob = paths / bin
	}
	if !(prob >= 0 && prob <= 1) {
		return prob, false
	}
	return prob, true
}

// probOutsideSquare is SciPy's _compute_prob_outside_square(n, h):
// Pr(D_{n,n} >= h/n) by a Horner-like alternating sum.
func probOutsideSquare(n, h int) float64 {
	P := 0.0
	for k := n / h; k >= 0; k-- {
		p1 := 1.0
		for j := 0; j < h; j++ {
			p1 = float64(n-k*h-j) * p1 / float64(n+k*h+j+1)
		}
		P = p1 * (1 - P)
	}
	return 2 * P
}

// outerProbInside is SciPy's _compute_outer_prob_inside_method(m, n, g,
// h): the proportion of lattice paths from (0, 0) to (m, n) leaving the
// band |x/m - y/n| < h/lcm(m, n), by Viehmann's recursion for 1 - p.
func outerProbInside(m, n, g, h int) float64 {
	if m < n {
		m, n = n, m
	}
	mg, ng := m/g, n/g
	minj, maxj := 0, min(int(math.Ceil(float64(h)/float64(mg))), n+1)
	curlen := maxj - minj
	lenA := min(2*maxj+2, n+1)
	A := make([]float64, lenA)
	for i := range A {
		A[i] = 1
	}
	for i := minj; i < maxj; i++ {
		A[i] = 0
	}
	for i := 1; i <= m; i++ {
		lastminj, lastlen := minj, curlen
		minj = max(int(math.Floor(float64(ng*i-h)/float64(mg)))+1, 0)
		minj = min(minj, n)
		maxj = min(int(math.Ceil(float64(ng*i+h)/float64(mg))), n+1)
		if maxj <= minj {
			return 1
		}
		val := 1.0
		if minj == 0 {
			val = 0
		}
		fi := float64(i)
		for jj := 0; jj < maxj-minj; jj++ {
			j := float64(jj + minj)
			val = (A[jj+minj-lastminj]*fi + val*j) / (fi + j)
			A[jj] = val
		}
		curlen = maxj - minj
		for k := curlen; k < lastlen; k++ {
			A[k] = 1
		}
	}
	return A[maxj-minj-1]
}

// countPathsOutside is SciPy's _count_paths_outside_method(m, n, g, h):
// the number of lattice paths from (0, 0) to (m, n) reaching
// m·y <= n·x - h·g (Hodges' treatment of Drion/Gnedenko/Korolyuk).
func countPathsOutside(m, n, g, h int) float64 {
	if m < n {
		m, n = n, m
	}
	mg, ng := m/g, n/g
	// h <= lcm(m, n) = mg·n, so there is at least one x to check.
	lxj := n + floorDiv(mg-h, mg)
	xj := make([]int, lxj)
	for j := range xj {
		xj[j] = floorDiv(h+mg*j+ng-1, ng)
	}
	B := make([]float64, lxj)
	B[0] = 1
	for j := 1; j < lxj; j++ {
		Bj := special.Binom(float64(xj[j]+j), float64(j))
		for i := 0; i < j; i++ {
			Bj -= special.Binom(float64(xj[j]-xj[i]+j-i), float64(j-i)) * B[i]
		}
		B[j] = Bj
	}
	paths := 0.0
	for j := 0; j < lxj; j++ {
		paths += B[j] * special.Binom(float64(m-xj[j]+n-j), float64(n-j))
	}
	return paths
}

func floorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}
