package stats

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-numerics/optimize"
	"github.com/go-numerics/special"
)

// ShapiroWilk is the Shapiro-Wilk test of the null hypothesis that the
// data come from a normal distribution: scipy.stats.shapiro(x), with
// SciPy 1.18's algorithm (Royston 1992: the approximate coefficients and
// the normalising transformation of W; for n = 3 the exact distribution).
// The data are read whole (an array in row-major order). With fewer than
// 3 observations, constant data or a NaN, the result is NaN.
func ShapiroWilk[D Data](x D) (TestResult, error) {
	v := floats(x)
	n := len(v)
	if n < 3 || anyNaN(v) {
		return TestResult{qnan, qnan}, nil
	}
	y := slices.Clone(v)
	slices.Sort(y)
	c := v[n/2] // SciPy subtracts x[n//2] of the unsorted data (gh-15777)
	for i := range y {
		y[i] -= c
	}
	if n == 3 {
		h := math.Sqrt(2) / 2
		w := clipTo(swilkW(y, []float64{-h, 0, h}), 0.75, 1)
		return TestResult{w, clip01(1 - 6/math.Pi*math.Acos(math.Sqrt(w)))}, nil
	}
	fn := float64(n)
	m := make([]float64, n)
	for i := range m {
		m[i] = special.Ndtri((float64(i+1) - 3.0/8) / (fn + 1.0/4))
	}
	u := math.Pow(fn, -0.5)
	mTm := dotDD(m, m)
	r := math.Pow(mTm, -0.5)
	cn, cnm1 := r*m[n-1], r*m[n-2]
	mn, mnm1 := m[n-1], m[n-2]
	an := cn + 0.221157*u - 0.147981*u*u - 2.071190*u*u*u + 4.434685*u*u*u*u - 2.706056*u*u*u*u*u
	anm1 := cnm1 + 0.042981*u - 0.293762*u*u - 1.752461*u*u*u + 5.682633*u*u*u*u - 3.582633*u*u*u*u*u
	var phi float64
	if n <= 5 {
		phi = (mTm - 2*mn*mn) / (1 - 2*an*an)
	} else {
		phi = (mTm - 2*mn*mn - 2*mnm1*mnm1) / (1 - 2*an*an - 2*anm1*anm1)
	}
	s := math.Pow(phi, -0.5)
	a := make([]float64, n)
	for i := range a {
		a[i] = s * m[i]
	}
	if n > 5 {
		a[n-2], a[1] = anm1, -anm1
	}
	a[n-1], a[0] = an, -an
	w := swilkW(y, a)
	var mu, sigma, g float64
	if n <= 11 {
		gamma := -2.273 + 0.459*fn
		mu = 0.5440 - 0.39978*fn + 0.025054*fn*fn - 0.0006714*fn*fn*fn
		sigma = math.Exp(1.3822 - 0.77857*fn + 0.062767*fn*fn - 0.0020322*fn*fn*fn)
		g = -math.Log(gamma - math.Log(1-w))
	} else {
		lu := math.Log(fn)
		mu = -1.5861 - 0.31082*lu - 0.083751*lu*lu + 0.0038915*lu*lu*lu
		sigma = math.Exp(-0.4803 - 0.082676*lu + 0.0030302*lu*lu)
		g = math.Log(1 - w)
	}
	z := (g - mu) / sigma
	return TestResult{w, special.Ndtr(-z)}, nil
}

// swilkW is the Shapiro-Wilk statistic (a·y)² / Σ(y - ȳ)² for sorted y.
func swilkW(y, a []float64) float64 {
	num := dotDD(a, y)
	d := deviations(y)
	return num * num / dotDD(d, d)
}

// ADDist is the distribution family of AndersonDarling (SciPy's dist=).
// The zero value is ADNorm, SciPy's default.
type ADDist int

const (
	ADNorm     ADDist = iota // "norm": normal, mean and standard deviation estimated
	ADExpon                  // "expon": exponential, scale estimated (loc 0)
	ADLogistic               // "logistic": logistic, by maximum likelihood
	ADGumbelL                // "gumbel_l" ("gumbel", "extreme1"): left-skewed Gumbel, by maximum likelihood
	ADGumbelR                // "gumbel_r": right-skewed Gumbel, by maximum likelihood
)

// AndersonResult is the outcome of AndersonDarling (SciPy's AndersonResult
// and, for the p-value, the SignificanceResult of method="interpolate").
type AndersonResult struct {
	// Statistic is the Anderson-Darling statistic A².
	Statistic float64
	// PValue is the p-value interpolated linearly in the table of
	// critical values (SciPy's method="interpolate"), clamped to the
	// table's significance levels at its ends.
	PValue float64
	// CriticalValues are the critical values for SignificanceLevels (in
	// percent), as SciPy returns them (rounded to 3 decimals).
	CriticalValues, SignificanceLevels []float64
	// Loc and Scale are the fitted parameters of the distribution.
	Loc, Scale float64
}

var (
	adAvalsNorm     = []float64{0.561, 0.631, 0.752, 0.873, 1.035}
	adAvalsExpon    = []float64{0.916, 1.062, 1.321, 1.591, 1.959}
	adAvalsGumbel   = []float64{0.474, 0.637, 0.757, 0.877, 1.038}
	adAvalsLogistic = []float64{0.426, 0.563, 0.660, 0.769, 0.906, 1.010}
)

// AndersonDarling is the Anderson-Darling test of the null hypothesis
// that the data come from the given family, its parameters estimated from
// the data: scipy.stats.anderson(x, dist) with the critical values and
// significance levels, and the p-value of method="interpolate".
func AndersonDarling[D Data](x D, dist ADDist) (AndersonResult, error) {
	bad := AndersonResult{Statistic: qnan, PValue: qnan, Loc: qnan, Scale: qnan}
	if dist < ADNorm || dist > ADGumbelR {
		return bad, fmt.Errorf("%w: Anderson-Darling distribution %d", ErrParam, int(dist))
	}
	v := floats(x)
	if len(v) == 0 {
		return bad, fmt.Errorf("%w: anderson needs at least one observation", ErrData)
	}
	y := slices.Clone(v)
	slices.Sort(y)
	n := float64(len(y))
	xbar := mean(v)
	w := make([]float64, len(y))
	logcdf := make([]float64, len(y))
	logsf := make([]float64, len(y))
	var loc, scale float64
	var sig, crit []float64
	round3 := func(a []float64, den float64) []float64 {
		out := make([]float64, len(a))
		for i, c := range a {
			out[i] = math.RoundToEven(c/den*1000) / 1000
		}
		return out
	}
	switch dist {
	case ADNorm:
		loc, scale = xbar, math.Sqrt(variance(v, 1))
		for i, t := range y {
			w[i] = (t - loc) / scale
			logcdf[i] = special.LogNdtr(w[i])
			logsf[i] = special.LogNdtr(-w[i])
		}
		sig = []float64{15, 10, 5, 2.5, 1}
		crit = round3(adAvalsNorm, 1.0+0.75/n+2.25/n/n)
	case ADExpon:
		loc, scale = 0, xbar
		for i, t := range y {
			w[i] = t / xbar
			logcdf[i] = math.Inf(-1)
			logsf[i] = 0
			if w[i] > 0 {
				logcdf[i] = math.Log(-math.Expm1(-w[i]))
				logsf[i] = -w[i]
			}
		}
		sig = []float64{15, 10, 5, 2.5, 1}
		crit = round3(adAvalsExpon, 1.0+0.6/n)
	case ADLogistic:
		loc, scale = fitLogisticAD(v, xbar)
		for i, t := range y {
			w[i] = (t - loc) / scale
			logcdf[i] = special.LogExpit(w[i])
			logsf[i] = special.LogExpit(-w[i])
		}
		sig = []float64{25, 10, 5, 2.5, 1, 0.5}
		crit = round3(adAvalsLogistic, 1.0+0.25/n)
	default:
		if anyNaN(v) || slices.ContainsFunc(v, func(t float64) bool { return math.IsInf(t, 0) }) {
			return bad, fmt.Errorf("%w: the data contain non-finite values", ErrData)
		}
		data := v
		if dist == ADGumbelL {
			data = make([]float64, len(v))
			for i, t := range v {
				data[i] = -t
			}
		}
		l, s, xmin, c, err := fitGumbelR(data)
		if err != nil {
			return bad, err
		}
		loc, scale = l, s
		if dist == ADGumbelL {
			loc = -l
		}
		for i, t := range y {
			// (t - loc)/scale with loc = xmin - scale·c unrounded: data far
			// from 0 would otherwise lose the digits of the rounded loc.
			z := (t-xmin)/scale + c
			if dist == ADGumbelL {
				z = (t+xmin)/scale - c
			}
			if dist == ADGumbelR {
				logcdf[i] = -math.Exp(-z)
				logsf[i] = math.Log(-math.Expm1(-math.Exp(-z)))
			} else {
				logcdf[i] = math.Log(-math.Expm1(-math.Exp(z)))
				logsf[i] = -math.Exp(z)
			}
		}
		sig = []float64{25, 10, 5, 2.5, 1}
		crit = round3(adAvalsGumbel, 1.0+0.2/math.Sqrt(n))
	}
	terms := make([]float64, len(y))
	for i := range y {
		terms[i] = (2*float64(i+1) - 1.0) / n * (logcdf[i] + logsf[len(y)-1-i])
	}
	a2 := -n - sum(terms)
	levels := make([]float64, len(sig))
	for i, s := range sig {
		levels[i] = s / 100
	}
	return AndersonResult{
		Statistic:          a2,
		PValue:             interpLinear(a2, crit, levels),
		CriticalValues:     crit,
		SignificanceLevels: sig,
		Loc:                loc,
		Scale:              scale,
	}, nil
}

// interp is np.interpLinear(x, xp, fp) for increasing xp: linear interpolation,
// clamped to fp's end values outside [xp[0], xp[-1]], NaN for a NaN x.
func interpLinear(x float64, xp, fp []float64) float64 {
	k := len(xp) - 1
	switch {
	case x != x:
		return x
	case x <= xp[0]:
		return fp[0]
	case x >= xp[k]:
		return fp[k]
	}
	j := 0
	for xp[j+1] <= x {
		j++
	}
	slope := (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
	return slope*(x-xp[j]) + fp[j]
}

// fitLogisticAD is the logistic fit of anderson(dist="logistic"): the
// maximum-likelihood equations solved by fsolve from (mean, std) with
// xtol 1e-5, as SciPy does.
func fitLogisticAD(x []float64, xbar float64) (loc, scale float64) {
	n := float64(len(x))
	t1 := make([]float64, len(x))
	t2 := make([]float64, len(x))
	f := func(ab, out []float64) {
		a, b := ab[0], ab[1]
		for i, xi := range x {
			tmp := (xi - a) / b
			e := math.Exp(tmp)
			t1[i] = 1.0 / (1 + e)
			t2[i] = tmp * (1.0 - e) / (1 + e)
		}
		out[0] = sum(t1) - 0.5*n
		out[1] = sum(t2) + n
	}
	// Fsolve fails only on a malformed problem; non-convergence is in
	// r.Status, which SciPy's anderson ignores too.
	r, _ := optimize.Fsolve(optimize.SystemProblem{Func: f}, []float64{xbar, math.Sqrt(variance(x, 1))}, &optimize.RootOptions{Xtol: 1e-5})
	return r.X[0], r.X[1]
}

// fitGumbelR is scipy.stats.gumbel_r.fit(data): the scale solving the
// maximum-likelihood equation mean(x) - Σ x·e^{-x/s} / Σ e^{-x/s} = s by
// Brent's method (bracket grown from (0.5, 2), xtol = rtol = 1e-14), then
// loc = -s·(logsumexp(-x/s) - log n). The equation is written in the
// deviations d = x - min(x), which leaves it unchanged mathematically but
// removes the cancellation SciPy suffers between mean(x) and the weighted
// mean when the data sit far from 0 (data near 1e8 leave SciPy's scale
// with an error of about 1e-9). It also returns min(x) and
// c = logsumexp(-d/s) - log n, loc being min(x) - s·c.
func fitGumbelR(data []float64) (loc, scale, xmin, c float64, err error) {
	n := float64(len(data))
	xmin = slices.Min(data)
	d := make([]float64, len(data))
	for i, x := range data {
		d[i] = x - xmin
	}
	dm := mean(d)
	sd := make([]float64, len(data))
	wts := make([]float64, len(data))
	prod := make([]float64, len(data))
	fn := func(s float64) float64 {
		for i, v := range d {
			wts[i] = math.Exp(-v / s)
			prod[i] = v * wts[i]
		}
		return dm - sum(prod)/sum(wts) - s
	}
	sign := func(v float64) float64 {
		switch {
		case v > 0:
			return 1
		case v < 0:
			return -1
		}
		return v
	}
	lo, hi := 0.5, 2.0
	for sign(fn(lo)) == sign(fn(hi)) && (lo > 0 || hi < math.Inf(1)) {
		lo /= 2
		hi *= 2
	}
	r, err := optimize.Brentq(fn, lo, hi, &optimize.BracketOptions{Xtol: 1e-14, Rtol: 1e-14})
	if err != nil {
		return qnan, qnan, qnan, qnan, fmt.Errorf("%w: gumbel fit: %v", ErrFit, err)
	}
	scale = r.Root
	for i, v := range d {
		sd[i] = -v / scale
	}
	c = special.LogSumExp(sd) - math.Log(n)
	return xmin - scale*c, scale, xmin, c, nil
}

// SkewTestOptions are the options of SkewTest and KurtosisTest. The zero
// value (or nil) is a two-sided test.
type SkewTestOptions struct {
	Alternative Alternative
}

// SkewTest is D'Agostino's test of the null hypothesis that the skewness
// of the population is that of a normal distribution:
// scipy.stats.skewtest(a, alternative). It needs at least 8 observations
// (fewer, constant data or a NaN give NaN, as in SciPy).
func SkewTest[D Data](x D, o *SkewTestOptions) (TestResult, error) {
	if o == nil {
		o = &SkewTestOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return TestResult{qnan, qnan}, err
	}
	z := skewZ(floats(x))
	return TestResult{z, pvalueSym(z, normSF, o.Alternative)}, nil
}

func skewZ(a []float64) float64 {
	if len(a) < 8 || anyNaN(a) {
		return qnan
	}
	b2 := skewness(a, false)
	n := float64(len(a))
	y := b2 * math.Sqrt(((n+1)*(n+3))/(6.0*(n-2)))
	beta2 := 3.0 * (n*n + 27*n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
	w2 := -1 + math.Sqrt(2*(beta2-1))
	delta := 1 / math.Sqrt(0.5*math.Log(w2))
	alpha := math.Sqrt(2.0 / (w2 - 1))
	if y == 0 {
		y = 1
	}
	ya := y / alpha
	return delta * math.Log(ya+math.Sqrt(ya*ya+1))
}

// KurtosisTest is Anscombe and Glynn's test of the null hypothesis that
// the kurtosis of the population is that of a normal distribution:
// scipy.stats.kurtosistest(a, alternative). It needs at least 5
// observations (fewer, constant data or a NaN give NaN, as in SciPy).
func KurtosisTest[D Data](x D, o *SkewTestOptions) (TestResult, error) {
	if o == nil {
		o = &SkewTestOptions{}
	}
	if err := checkAlternative(o.Alternative); err != nil {
		return TestResult{qnan, qnan}, err
	}
	z := kurtosisZ(floats(x))
	return TestResult{z, pvalueSym(z, normSF, o.Alternative)}, nil
}

func kurtosisZ(a []float64) float64 {
	if len(a) < 5 || anyNaN(a) {
		return qnan
	}
	b2 := kurt(a, false, true)
	n := float64(len(a))
	e := 3.0 * (n - 1) / (n + 1)
	varb2 := 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5))
	x := (b2 - e) / math.Sqrt(varb2)
	sqrtbeta1 := 6.0 * (n*n - 5*n + 2) / ((n + 7) * (n + 9)) * math.Sqrt((6.0*(n+3)*(n+5))/(n*(n-2)*(n-3)))
	A := 6.0 + 8.0/sqrtbeta1*(2.0/sqrtbeta1+math.Sqrt(1+4.0/(sqrtbeta1*sqrtbeta1)))
	term1 := 1 - 2/(9.0*A)
	denom := 1 + x*math.Sqrt(2/(A-4.0))
	// sign(denom)·((1-2/A)/|denom|)^(1/3); denom/|denom| is NaN for
	// denom = 0, where SciPy returns NaN too.
	term2 := crPow((1-2.0/A)/math.Abs(denom), 1.0/3) * (denom / math.Abs(denom))
	return (term1 - term2) / math.Sqrt(2/(9.0*A))
}

// Normaltest is D'Agostino and Pearson's omnibus test of normality, the
// sum of the squared statistics of SkewTest and KurtosisTest against the
// chi-squared distribution with 2 degrees of freedom:
// scipy.stats.normaltest(a). It needs at least 8 observations.
func Normaltest[D Data](x D) (TestResult, error) {
	a := floats(x)
	s, k := skewZ(a), kurtosisZ(a)
	stat := s*s + k*k
	return TestResult{stat, chi2SF(2, stat)}, nil
}

// JarqueBera is the Jarque-Bera goodness-of-fit test of normality,
// n/6·(S² + (K-3)²/4) against the chi-squared distribution with 2 degrees
// of freedom: scipy.stats.jarque_bera(x). Constant data or a NaN give NaN;
// an empty sample gives NaN, as in SciPy.
func JarqueBera[D Data](x D) (TestResult, error) {
	a := floats(x)
	if len(a) == 0 || anyNaN(a) {
		return TestResult{qnan, qnan}, nil
	}
	m := mean(a)
	d := make([]float64, len(a))
	for i, v := range a {
		d[i] = v - m
	}
	s := skewness(d, false)
	k := kurt(d, false, false)
	n := float64(len(a))
	stat := n / 6 * (s*s + k*k/4)
	return TestResult{stat, chi2SF(2, stat)}, nil
}
