package stats

import (
	"fmt"
	"math"

	"github.com/go-numerics/special"
)

// MannWhitneyUOptions are the options of MannWhitneyU. The zero value
// (or nil) is SciPy's default: two-sided, method "auto", with the
// continuity correction.
type MannWhitneyUOptions struct {
	Alternative Alternative
	// Method is MethodAuto (see MannWhitneyU), MethodExact or
	// MethodAsymptotic.
	Method Method
	// NoContinuity disables the continuity correction of the asymptotic
	// p-value (SciPy's use_continuity=False).
	NoContinuity bool
}

// RankTestResult is the outcome of MannWhitneyU and Wilcoxon.
type RankTestResult struct {
	Statistic, PValue float64
	// ZStatistic is the standardized statistic of the normal
	// approximation when the method requested is MethodAsymptotic
	// (SciPy's zstatistic attribute), NaN otherwise.
	ZStatistic float64
}

var nanRank = RankTestResult{qnan, qnan, qnan}

// MannWhitneyU is the Mann-Whitney U rank test of the null hypothesis
// that the distributions underlying two independent samples are equal:
// scipy.stats.mannwhitneyu(x, y, use_continuity, alternative, method).
// The statistic is U1, that of x. With MethodAuto the p-value is exact
// (the null distribution by Löffler's recurrence, no correction for ties)
// when one of the samples has at most 8 observations and there are no
// ties, asymptotic (normal, tie-corrected) otherwise.
func MannWhitneyU[D Data](x, y D, o *MannWhitneyUOptions) (RankTestResult, error) {
	if o == nil {
		o = &MannWhitneyUOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanRank, err
	}
	if o.Method != MethodAuto && o.Method != MethodExact && o.Method != MethodAsymptotic {
		return nanRank, fmt.Errorf("%w: mannwhitneyu method %v", ErrParam, o.Method)
	}
	a, b := floats(x), floats(y)
	if len(a) == 0 || len(b) == 0 || hasNaN(a, b) {
		return nanRank, nil
	}
	xy := make([]float64, 0, len(a)+len(b))
	xy = append(append(xy, a...), b...)
	ranks, t := rankWithTies(xy, RankAverage, true)
	n1, n2 := float64(len(a)), float64(len(b))
	r1 := sum(ranks[:len(a)])
	u1 := r1 - n1*(n1+1)/2
	u2 := n1*n2 - u1
	u, f := math.Max(u1, u2), 2.0
	switch o.Alternative {
	case Greater:
		u, f = u1, 1
	case Less:
		u, f = u2, 1
	}
	method := o.Method
	if method == MethodAuto {
		ties := false
		for _, c := range t {
			if c > 1 {
				ties = true
			}
		}
		method = MethodExact
		if (n1 > 8 && n2 > 8) || ties {
			method = MethodAsymptotic
		}
	}
	res := RankTestResult{Statistic: u1, ZStatistic: qnan}
	if method == MethodExact {
		d := newMWU(len(a), len(b))
		res.PValue = clip01(d.sf(int64(u)) * f)
		return res, nil
	}
	n := n1 + n2
	s := math.Sqrt(n1 * n2 / 12 * ((n + 1) - tieTerm(t)/(n*(n-1))))
	num := u1 - n1*n2/2
	if !o.NoContinuity {
		num -= 0.5 * correctionSign(num, o.Alternative)
	}
	z := num / s
	res.PValue = clip01(pvalueSym(z, normSF, o.Alternative))
	if o.Method == MethodAsymptotic {
		res.ZStatistic = z
	}
	return res, nil
}

// correctionSign is SciPy's _correction_sign.
func correctionSign(z float64, alt Alternative) float64 {
	switch alt {
	case Greater:
		return 1
	case Less:
		return -1
	}
	switch {
	case z > 0:
		return 1
	case z < 0:
		return -1
	}
	return z
}

// mwu is the null distribution of the Mann-Whitney U statistic for
// samples of sizes n1 <= n2, SciPy's _MWU: the counts of the
// configurations by Löffler's recurrence
// c(u) = Σ_{i<u} c(i)·σ(u-i) / u, σ(a) the sum of the divisors d of a with
// d <= n1, minus those with n2 < d <= n1+n2.
type mwu struct {
	n1, n2 int
}

func newMWU(a, b int) mwu { return mwu{min(a, b), max(a, b)} }

// pmfs returns P(U = u) for u = 0..maxu.
func (m mwu) pmfs(maxu int) []float64 {
	n1, n2 := m.n1, m.n2
	sigma := make([]float64, maxu+1)
	for d := 1; d <= n1; d++ {
		for i := d; i <= maxu; i += d {
			sigma[i] += float64(d)
		}
	}
	for d := n2 + 1; d <= n2+n1; d++ {
		for i := d; i <= maxu; i += d {
			sigma[i] -= float64(d)
		}
	}
	conf := make([]float64, maxu+1)
	conf[0] = 1
	for u := 1; u <= maxu; u++ {
		s := 0.0
		for i := 0; i < u; i++ {
			s += conf[i] * sigma[u-i]
		}
		conf[u] = s / float64(u)
	}
	total := special.Binom(float64(n1+n2), float64(n1))
	for i := range conf {
		conf[i] /= total
	}
	return conf
}

// sf is P(U >= k), SciPy's _MWU.sf: the complement's CDF where it is
// the smaller tail.
func (m mwu) sf(k int64) float64 {
	kc := int64(m.n1)*int64(m.n2) - k
	if k < kc {
		p := m.pmfs(int(k))
		c := 0.0
		for _, v := range p {
			c += v
		}
		return 1 - c + p[k]
	}
	p := m.pmfs(int(kc))
	c := 0.0
	for _, v := range p {
		c += v
	}
	return c
}

// ZeroMethod is how Wilcoxon treats zero differences (SciPy's
// zero_method=). The zero value is ZeroWilcox, SciPy's default.
type ZeroMethod int

const (
	// ZeroWilcox drops the zero differences (zero_method="wilcox").
	ZeroWilcox ZeroMethod = iota
	// ZeroPratt ranks the zeros, then drops them from the statistic
	// (zero_method="pratt").
	ZeroPratt
	// ZeroZsplit ranks the zeros and splits their ranks evenly between
	// the positive and negative sums (zero_method="zsplit").
	ZeroZsplit
)

// WilcoxonOptions are the options of Wilcoxon. The zero value (or nil)
// is SciPy's default: zero_method "wilcox", no continuity correction,
// two-sided, method "auto".
type WilcoxonOptions struct {
	ZeroMethod ZeroMethod
	// Correction applies the continuity correction to the normal
	// approximation (SciPy's correction=True).
	Correction  bool
	Alternative Alternative
	// Method: MethodAuto, MethodExact or MethodAsymptotic. MethodAuto
	// follows SciPy 1.18: the exact distribution when there are at most
	// 50 differences and neither ties nor zeros; with ties or zeros and
	// at most 13 differences, the exact permutation distribution over all
	// 2^n sign changes (SciPy's PermutationMethod); the normal
	// approximation otherwise.
	Method Method
}

// Wilcoxon is the Wilcoxon signed-rank test of the null hypothesis that
// the differences x - y (or x itself when y is empty) are symmetric about
// zero: scipy.stats.wilcoxon(x, y, zero_method, correction, alternative,
// method). The statistic is the smaller of the positive- and
// negative-rank sums for the two-sided test, the positive-rank sum
// otherwise. Pass a nil y (or an empty one) for the one-sample test.
func Wilcoxon[D Data](x, y D, o *WilcoxonOptions) (RankTestResult, error) {
	if o == nil {
		o = &WilcoxonOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return nanRank, err
	}
	if o.ZeroMethod < ZeroWilcox || o.ZeroMethod > ZeroZsplit {
		return nanRank, fmt.Errorf("%w: zero method %d", ErrParam, int(o.ZeroMethod))
	}
	if o.Method != MethodAuto && o.Method != MethodExact && o.Method != MethodAsymptotic {
		return nanRank, fmt.Errorf("%w: wilcoxon method %v", ErrParam, o.Method)
	}
	a, b := floats(x), floats(y)
	d := a
	if len(b) > 0 {
		if len(a) != len(b) {
			return nanRank, fmt.Errorf("%w: wilcoxon needs samples of equal length, got %d and %d", ErrData, len(a), len(b))
		}
		d = make([]float64, len(a))
		for i := range a {
			d[i] = a[i] - b[i]
		}
	}
	if len(d) == 0 || hasNaN(d) {
		return nanRank, nil
	}
	return wilcoxon(d, o), nil
}

func wilcoxon(d []float64, o *WilcoxonOptions) RankTestResult {
	n := len(d)
	method := o.Method
	if method == MethodAuto && n > 50 {
		method = MethodAsymptotic
	}
	nZero := 0
	abs := make([]float64, n)
	for i, v := range d {
		abs[i] = math.Abs(v)
		if v == 0 {
			nZero++
			if o.ZeroMethod == ZeroWilcox {
				abs[i] = qnan // dropped: ranked last, never counted
			}
		}
	}
	r, t := rankWithTies(abs, RankAverage, true)
	count := float64(n)
	if o.ZeroMethod == ZeroWilcox {
		count -= float64(nZero)
	}
	pos := make([]float64, n)
	neg := make([]float64, n)
	zer := make([]float64, n)
	for i, v := range d {
		switch {
		case v > 0:
			pos[i] = r[i]
		case v < 0:
			neg[i] = r[i]
		default:
			zer[i] = r[i]
		}
	}
	rPlus, rMinus := sum(pos), sum(neg)
	hasTies := false
	for _, c := range t {
		if c == 0 {
			hasTies = true
		}
	}
	zsplit := 0.0
	if o.ZeroMethod == ZeroZsplit {
		zsplit = sum(zer) / 2
		rPlus += zsplit
		rMinus += zsplit
	}
	mn := count * (count + 1) * 0.25
	se := count * (count + 1) * (2*count + 1)
	if o.ZeroMethod == ZeroPratt {
		nz := float64(nZero)
		mn -= nz * (nz + 1) * 0.25
		se -= nz * (nz + 1) * (2*nz + 1)
		if nZero > 0 {
			t[0] = 0
		}
	}
	se = math.Sqrt((se - tieTerm(t)/2) / 24)
	z := qnan
	if method == MethodAuto || method == MethodAsymptotic {
		z = (rPlus - mn) / se
	}
	perm := false
	if method == MethodAuto {
		switch {
		case !hasTies && nZero == 0:
			method = MethodExact
		case n <= 13:
			perm = true
		default:
			method = MethodAsymptotic
		}
	}
	var p float64
	switch {
	case perm:
		p = wilcoxonPermutation(d, r, zsplit, rPlus, o.Alternative)
	case method == MethodAsymptotic:
		if o.Correction {
			z -= correctionSign(z, o.Alternative) * 0.5 / se
		}
		p = pvalueSym(z, normSF, o.Alternative)
	default:
		dist := wilcoxonPMF(int(count))
		mean := count * (count + 1) / 4
		lo, hi := int(math.Ceil(rPlus)), int(math.Floor(rPlus))
		switch o.Alternative {
		case Less:
			p = wilcoxonCDF(dist, lo, mean)
		case Greater:
			p = wilcoxonSF(dist, hi, mean)
		default:
			p = clip01(2 * math.Min(wilcoxonSF(dist, hi, mean), wilcoxonCDF(dist, lo, mean)))
		}
	}
	stat := rPlus
	if o.Alternative == TwoSided {
		stat = math.Min(rPlus, rMinus)
	}
	zs := qnan
	if o.Method == MethodAsymptotic {
		zs = z
		if o.Alternative == TwoSided {
			zs = -math.Abs(z)
		}
	}
	return RankTestResult{stat, p, zs}
}

// wilcoxonPMF is SciPy's _get_wilcoxon_distr(n): P(R+ = r), r = 0..n(n+1)/2.
func wilcoxonPMF(n int) []float64 {
	c := []float64{1}
	for k := 1; k <= n; k++ {
		next := make([]float64, k*(k+1)/2+1)
		m := len(c)
		for i, v := range c {
			next[i] = v * 0.5
		}
		off := len(next) - m
		for i, v := range c {
			next[off+i] += v * 0.5
		}
		c = next
	}
	return c
}

// wilcoxonCDF and wilcoxonSF are SciPy's WilcoxonDistribution.cdf and
// .sf: the sum of the probabilities of the near tail, one minus that of
// the far tail beyond the mean.
func wilcoxonCDF(p []float64, k int, mean float64) float64 {
	if float64(k) <= mean {
		return sum(p[:min(max(k+1, 0), len(p))])
	}
	return 1 - sum(p[min(k+1, len(p)):])
}

func wilcoxonSF(p []float64, k int, mean float64) float64 {
	if float64(k) <= mean {
		return sum(p[min(max(k, 0), len(p)):])
	}
	return 1 - sum(p[:min(k, len(p))])
}

// wilcoxonPermutation is the p-value of SciPy's permutation_test over all
// 2^n sign changes of the differences (exact for n <= 13), the statistic
// being the positive-rank sum; values within a relative 100·eps of the
// observed one count as equal, as in SciPy.
func wilcoxonPermutation(d, r []float64, zsplit, observed float64, alt Alternative) float64 {
	n := len(d)
	w := make([]float64, 0, n)
	for i, v := range d {
		if v != 0 {
			w = append(w, r[i])
		}
	}
	// Each zero difference doubles every count, which cancels in the
	// ratio; enumerate the sign changes of the non-zero ones.
	total := 1 << len(w)
	null := make([]float64, total)
	for s := 0; s < total; s++ {
		v := 0.0
		for i := range w {
			if s&(1<<i) != 0 {
				v += w[i]
			}
		}
		null[s] = v + zsplit
	}
	gamma := math.Abs(0x1p-52 * 100 * observed)
	less, greater := 0, 0
	for _, v := range null {
		if v <= observed+gamma {
			less++
		}
		if v >= observed-gamma {
			greater++
		}
	}
	pl := float64(less) / float64(total)
	pg := float64(greater) / float64(total)
	switch alt {
	case Less:
		return clip01(pl)
	case Greater:
		return clip01(pg)
	}
	return clip01(math.Min(pl, pg) * 2)
}

// KruskalWallis is the Kruskal-Wallis H test of the null hypothesis that
// the medians of all groups are equal: scipy.stats.kruskal(*samples). The
// statistic is corrected for ties; the p-value is from the chi-squared
// distribution with k - 1 degrees of freedom. At least two groups are
// required; an empty group or a NaN gives a NaN result, as in SciPy.
func KruskalWallis[D Data](samples ...D) (TestResult, error) {
	if len(samples) < 2 {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: kruskal needs at least two groups, got %d", ErrData, len(samples))
	}
	groups := make([][]float64, len(samples))
	total := 0
	for i, s := range samples {
		groups[i] = floats(s)
		if len(groups[i]) == 0 || hasNaN(groups[i]) {
			return TestResult{qnan, qnan}, nil
		}
		total += len(groups[i])
	}
	all := make([]float64, 0, total)
	for _, g := range groups {
		all = append(all, g...)
	}
	ranks, t := rankWithTies(all, RankAverage, true)
	n := float64(total)
	ties := 1 - tieTerm(t)/(n*n*n-n)
	ssbn := 0.0
	off := 0
	for _, g := range groups {
		s := sum(ranks[off : off+len(g)])
		ssbn += s * s / float64(len(g))
		off += len(g)
	}
	h := 12.0/(n*(n+1))*ssbn - 3*(n+1)
	h /= ties
	return TestResult{h, chi2SF(float64(len(groups)-1), h)}, nil
}
