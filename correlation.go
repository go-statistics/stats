package stats

import (
	"fmt"
	"math"

	"github.com/go-numerics/special"
)

// CorrelationOptions are the options of Pearsonr, Spearmanr and
// Linregress. The zero value (or nil) is a two-sided test.
type CorrelationOptions struct {
	Alternative Alternative
}

// PearsonResult is the outcome of Pearsonr (SciPy's PearsonRResult).
type PearsonResult struct {
	// Statistic is Pearson's product-moment correlation coefficient r.
	Statistic float64
	// PValue is the p-value of the test of zero correlation, from the
	// exact distribution of r for bivariate normal data.
	PValue float64

	n   float64
	alt Alternative
}

// ConfidenceInterval returns the confidence interval for the population
// correlation coefficient at the given level (in [0, 1]), by Fisher's
// z-transformation, on the side of the test's alternative:
// PearsonRResult.confidence_interval(confidence_level). For n <= 3 the
// interval is [-1, 1].
func (r PearsonResult) ConfidenceInterval(level float64) (ConfidenceInterval, error) {
	if !(level >= 0 && level <= 1) {
		return ConfidenceInterval{qnan, qnan}, fmt.Errorf("%w: confidence level %v outside [0, 1]", ErrParam, level)
	}
	if r.n <= 3 {
		return ConfidenceInterval{-1, 1}, nil
	}
	zr := math.Atanh(r.Statistic)
	se := math.Sqrt(1 / (r.n - 3))
	switch r.alt {
	case Less:
		h := special.Ndtri(level)
		return ConfidenceInterval{-1, math.Tanh(zr + h*se)}, nil
	case Greater:
		h := special.Ndtri(level)
		return ConfidenceInterval{math.Tanh(zr - h*se), 1}, nil
	}
	h := special.Ndtri(0.5 + level/2)
	return ConfidenceInterval{math.Tanh(zr - h*se), math.Tanh(zr + h*se)}, nil
}

// pairs returns the two samples as float64 slices of equal length.
func pairs[D Data](x, y D, name string) ([]float64, []float64, error) {
	a, b := floats(x), floats(y)
	if len(a) != len(b) {
		return a, b, fmt.Errorf("%w: %s needs samples of equal length, got %d and %d", ErrData, name, len(a), len(b))
	}
	return a, b, nil
}

// constant reports whether every value equals the first.
func isConstant(a []float64) bool {
	for _, v := range a {
		if v != a[0] {
			return false
		}
	}
	return true
}

// Pearsonr is Pearson's correlation coefficient and the p-value of the
// test of zero correlation: scipy.stats.pearsonr(x, y, alternative=...).
// Under the null hypothesis r has the beta distribution on [-1, 1] with
// both shapes n/2 - 1, whose survival function gives the p-value. The
// samples must have equal lengths of at least 2. A constant sample or a
// NaN gives a NaN statistic and p-value, as in SciPy.
func Pearsonr[D Data](x, y D, o *CorrelationOptions) (PearsonResult, error) {
	if o == nil {
		o = &CorrelationOptions{}
	}
	bad := PearsonResult{Statistic: qnan, PValue: qnan, n: qnan}
	if err := checkAlternative(o.Alternative); err != nil {
		return bad, err
	}
	a, b, err := pairs(x, y, "pearsonr")
	if err != nil {
		return bad, err
	}
	if len(a) < 2 {
		return bad, fmt.Errorf("%w: pearsonr needs at least 2 observations, got %d", ErrData, len(a))
	}
	n := float64(len(a))
	res := PearsonResult{n: n, alt: o.Alternative}
	if anyNaN(a, b) || isConstant(a) || isConstant(b) {
		res.Statistic, res.PValue = qnan, qnan
		return res, nil
	}
	r := clipTo(correlate(a, b), -1, 1)
	if len(a) == 2 {
		res.Statistic, res.PValue = math.RoundToEven(r), 1
		return res, nil
	}
	ab := n/2 - 1
	beta := Beta{A: ab, B: ab, Loc: -1, Scale: 2}
	res.Statistic = r
	res.PValue = pvalueSym(r, func(t float64) float64 {
		// The null distribution is symmetric about 0: sf(-t) = cdf(t).
		if t < 0 {
			return beta.CDF(-t)
		}
		return beta.SF(t)
	}, o.Alternative)
	return res, nil
}

// correlate is the sample correlation of a and b, the deviations from
// the means scaled by their norms before the inner product, as SciPy's
// pearsonr forms it (with a double-double inner product).
func correlate(a, b []float64) float64 {
	ma, mb := mean(a), mean(b)
	return devDot(a, b, ma, mb) / math.Sqrt(devDot(a, a, ma, ma)) / math.Sqrt(devDot(b, b, mb, mb))
}

// Spearmanr is Spearman's rank correlation coefficient of two samples
// and the p-value of the test of zero correlation by the t distribution
// with n - 2 degrees of freedom: scipy.stats.spearmanr(x, y,
// alternative=...). Ties get average ranks. With fewer than 2
// observations, a constant sample or a NaN the result is NaN, as in
// SciPy.
func Spearmanr[D Data](x, y D, o *CorrelationOptions) (TestResult, error) {
	if o == nil {
		o = &CorrelationOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return TestResult{qnan, qnan}, err
	}
	a, b, err := pairs(x, y, "spearmanr")
	if err != nil {
		return TestResult{qnan, qnan}, err
	}
	if len(a) <= 1 || anyNaN(a, b) {
		return TestResult{qnan, qnan}, nil
	}
	// The two rankings (each a sort) run concurrently.
	var ra, rb []float64
	done := make(chan struct{})
	go func() {
		ra, _ = rankWithTies(a, RankAverage, false)
		close(done)
	}()
	rb, _ = rankWithTies(b, RankAverage, false)
	<-done
	// np.corrcoef: the covariance matrix scaled by the standard deviations.
	n := float64(len(a))
	ma, mb := mean(ra), mean(rb)
	cxy := devDot(ra, rb, ma, mb) / (n - 1)
	sx := math.Sqrt(devDot(ra, ra, ma, ma) / (n - 1))
	sy := math.Sqrt(devDot(rb, rb, mb, mb) / (n - 1))
	rs := clipTo(cxy/sx/sy, -1, 1)
	dof := n - 2
	t := rs * math.Sqrt(dof/((rs+1)*(1-rs)))
	return TestResult{rs, pvalueSym(t, studentSF(dof), o.Alternative)}, nil
}

// KendallVariant selects Kendall's tau-b or tau-c (SciPy's variant=). The
// zero value is TauB, SciPy's default.
type KendallVariant int

const (
	// TauB is tau-b, adjusted for ties in either variable.
	TauB KendallVariant = iota
	// TauC is Stuart's tau-c, for rectangular contingency tables.
	TauC
)

// KendallOptions are the options of Kendalltau. The zero value (or nil)
// is SciPy's default: tau-b, method "auto", two-sided.
type KendallOptions struct {
	Variant     KendallVariant
	Method      Method // MethodAuto, MethodExact or MethodAsymptotic
	Alternative Alternative
}

// Kendalltau is Kendall's rank correlation tau and the p-value of the
// test of independence: scipy.stats.kendalltau(x, y, method, variant,
// alternative). The p-value depends only on the difference of concordant
// and discordant pairs: with MethodAuto it is the exact one (Kendall's
// recursion) when there are no ties and n <= 33 or the discordant (or
// concordant) pairs number at most 1, the normal approximation with tie
// corrections otherwise. MethodExact with ties is an error, as in SciPy.
// With fewer than 2 observations, a constant sample or a NaN the result
// is NaN.
func Kendalltau[D Data](x, y D, o *KendallOptions) (TestResult, error) {
	if o == nil {
		o = &KendallOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return TestResult{qnan, qnan}, err
	}
	if o.Variant != TauB && o.Variant != TauC {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: Kendall variant %d", ErrParam, int(o.Variant))
	}
	if o.Method != MethodAuto && o.Method != MethodExact && o.Method != MethodAsymptotic {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: kendalltau method %v", ErrParam, o.Method)
	}
	a, b, err := pairs(x, y, "kendalltau")
	if err != nil {
		return TestResult{qnan, qnan}, err
	}
	if len(a) <= 1 || anyNaN(a, b) {
		return TestResult{qnan, qnan}, nil
	}
	return kendall(a, b, o)
}

func kendall(a, b []float64, o *KendallOptions) (TestResult, error) {
	size := len(a)
	// Dense ranks of y and x; the pairs sorted by (x, y).
	ry := denseRanks(b)
	rx := denseRanks(a)
	p := make([]kendallPair, size)
	for i := range p {
		p[i] = kendallPair{rx[i], ry[i]}
	}
	sortKendall(p)
	ys := make([]int, size)
	for i, q := range p {
		ys[i] = q.y
	}
	dis := countInversions(ys)

	var ntie int64
	for i := 0; i < size; {
		j := i + 1
		for j < size && p[j] == p[i] {
			j++
		}
		c := int64(j - i)
		ntie += c * (c - 1) / 2
		i = j
	}
	xtie, x0, x1, nx := tieCounts(rx)
	ytie, y0, y1, ny := tieCounts(ry)
	tot := int64(size) * int64(size-1) / 2
	if xtie == tot || ytie == tot {
		return TestResult{qnan, qnan}, nil
	}
	cmd := tot - xtie - ytie + ntie - 2*dis
	var tau float64
	if o.Variant == TauB {
		tau = float64(cmd) / math.Sqrt(float64(tot-xtie)) / math.Sqrt(float64(tot-ytie))
	} else {
		minclasses := min(nx, ny)
		tau = 2 * float64(cmd) / (float64(int64(size)*int64(size)*int64(minclasses-1)) / float64(minclasses))
	}
	tau = min(1, max(-1, tau))
	method := o.Method
	if method == MethodExact && (xtie != 0 || ytie != 0) {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: ties found, the exact method cannot be used", ErrData)
	}
	if method == MethodAuto {
		method = MethodAsymptotic
		if xtie == 0 && ytie == 0 && (size <= 33 || min(dis, tot-dis) <= 1) {
			method = MethodExact
		}
	}
	if method == MethodExact {
		return TestResult{tau, kendallExactP(size, tot-dis, o.Alternative)}, nil
	}
	n := float64(size)
	m := n * (n - 1)
	v := (m*(2*n+5)-x1-y1)/18 + 2*float64(xtie)*float64(ytie)/m + x0*y0/(9*m*(n-2))
	z := float64(cmd) / math.Sqrt(v)
	return TestResult{tau, pvalueSym(z, normSF, o.Alternative)}, nil
}

type kendallPair struct{ x, y int }

// denseRanks returns the dense ranks (1, 2, ...) of the values.
func denseRanks(a []float64) []int {
	p := sortedPairs(a, false)
	r := make([]int, len(a))
	k := 0
	for i, q := range p {
		if i == 0 || q.v != p[i-1].v {
			k++
		}
		r[q.i] = k
	}
	return r
}

// sortKendall sorts the pairs by x, then y, by counting sort on the
// dense ranks (two stable passes).
func sortKendall(p []kendallPair) {
	n := len(p)
	tmp := make([]kendallPair, n)
	pass := func(src, dst []kendallPair, key func(kendallPair) int) {
		cnt := make([]int, n+2)
		for _, q := range src {
			cnt[key(q)+1]++
		}
		for i := 1; i < len(cnt); i++ {
			cnt[i] += cnt[i-1]
		}
		for _, q := range src {
			k := key(q)
			dst[cnt[k]] = q
			cnt[k]++
		}
	}
	pass(p, tmp, func(q kendallPair) int { return q.y })
	pass(tmp, p, func(q kendallPair) int { return q.x })
}

// countInversions counts the pairs i < j with y[i] > y[j], by merge sort.
func countInversions(y []int) int64 {
	buf := make([]int, len(y))
	var rec func(a, b []int) int64
	rec = func(a, b []int) int64 {
		n := len(a)
		if n < 2 {
			return 0
		}
		h := n / 2
		c := rec(a[:h], b[:h]) + rec(a[h:], b[h:])
		i, j, k := 0, h, 0
		for i < h && j < n {
			if a[j] < a[i] {
				b[k] = a[j]
				c += int64(h - i)
				j++
			} else {
				b[k] = a[i]
				i++
			}
			k++
		}
		k += copy(b[k:], a[i:h])
		copy(b[k:], a[j:n])
		copy(a, b[:n])
		return c
	}
	return rec(y, buf)
}

// tieCounts returns, for dense ranks, Σ c(c-1)/2, Σ c(c-1)(c-2),
// Σ c(c-1)(2c+5) over the tie group sizes c, and the number of groups.
func tieCounts(r []int) (t int64, v0, v1 float64, groups int) {
	cnt := make([]int64, len(r)+1)
	for _, k := range r {
		cnt[k]++
	}
	for _, c := range cnt {
		if c > 0 {
			groups++
		}
		if c > 1 {
			t += c * (c - 1) / 2
			f := float64(c)
			v0 += f * (f - 1) * (f - 2)
			v1 += f * (f - 1) * (2*f + 5)
		}
	}
	return t, v0, v1, groups
}

// kendallExactP is SciPy's _kendall_p_exact(n, c, alternative): the
// exact p-value for c concordant pairs among n untied observations, by
// Kendall's recursion for the number of permutations with a given
// number of inversions.
func kendallExactP(n int, c int64, alt Alternative) float64 {
	tot := int64(n) * int64(n-1) / 2
	inRightTail := c >= tot-c
	c = min(c, tot-c)
	var prob, pmass float64
	switch {
	case n == 2:
		prob, pmass = 1, 0.5
	case c == 0:
		prob = 0
		if n < 171 {
			prob = 2 / special.Factorial(float64(n))
		}
		pmass = prob / 2
	case c == 1:
		prob = 0
		if n < 172 {
			prob = 2 / special.Factorial(float64(n-1))
		}
		pmass = float64(n-1) / special.Factorial(float64(n))
	case 4*c == tot*2 && alt == TwoSided:
		prob = 1
	default:
		cc := int(c)
		nw := make([]float64, cc+1)
		nw[0], nw[1] = 1, 1
		for j := 3; j <= n; j++ {
			// cumsum, divided by j for n >= 171 to stay finite
			for i := 1; i <= cc; i++ {
				nw[i] += nw[i-1]
			}
			if n >= 171 {
				for i := range nw {
					nw[i] /= float64(j)
				}
			}
			if j <= cc {
				for i := cc; i >= j; i-- {
					nw[i] -= nw[i-j]
				}
			}
		}
		if n < 171 {
			f := special.Factorial(float64(n))
			prob = 2 * sum(nw) / f
			pmass = nw[cc] / f
		} else {
			prob = sum(nw)
			pmass = nw[cc] / 2
		}
	}
	if alt != TwoSided {
		if inRightTail == (alt == Greater) {
			prob /= 2
		} else {
			prob = 1 - prob/2 + pmass
		}
	}
	return clip01(prob)
}

// LinregressResult is the outcome of Linregress (SciPy's
// LinregressResult).
type LinregressResult struct {
	Slope, Intercept float64
	// RValue is Pearson's correlation coefficient.
	RValue float64
	// PValue is the p-value of the test of zero slope, by the t
	// distribution with n - 2 degrees of freedom.
	PValue float64
	// StdErr is the standard error of the slope, InterceptStdErr that of
	// the intercept.
	StdErr, InterceptStdErr float64
}

// Linregress is the least-squares regression line of y on x:
// scipy.stats.linregress(x, y, alternative=...). With fewer than 2
// observations or a NaN every field is NaN; a constant x gives NaN slope
// and intercept, as in SciPy.
func Linregress[D Data](x, y D, o *CorrelationOptions) (LinregressResult, error) {
	if o == nil {
		o = &CorrelationOptions{}
	}
	bad := LinregressResult{qnan, qnan, qnan, qnan, qnan, qnan}
	if err := checkAlternative(o.Alternative); err != nil {
		return bad, err
	}
	a, b, err := pairs(x, y, "linregress")
	if err != nil {
		return bad, err
	}
	if len(a) <= 1 || anyNaN(a, b) {
		return bad, nil
	}
	const tiny = 1.0e-20
	n := float64(len(a))
	xmean, ymean := mean(a), mean(b)
	ssxm := devDot(a, a, xmean, xmean) / n
	ssym := devDot(b, b, ymean, ymean) / n
	ssxym := devDot(a, b, xmean, ymean) / n
	var r float64
	switch {
	case ssxm == 0 || ssym == 0:
		r = 0
		if ssxym == 0 {
			r = qnan
		}
	default:
		r = clipTo(ssxym/math.Sqrt(ssxm*ssym), -1, 1)
	}
	slope := ssxym / ssxm
	intercept := ymean - slope*xmean
	df := n - 2
	t := r * math.Sqrt(df/((1-r+tiny)*(1+r+tiny)))
	p := pvalueSym(t, studentSF(df), o.Alternative)
	se := math.Sqrt((1 - r*r) * ssym / ssxm / df)
	ise := se * math.Sqrt(ssxm+xmean*xmean)
	return LinregressResult{slope, intercept, r, p, se, ise}, nil
}
