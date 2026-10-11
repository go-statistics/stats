package stats

import (
	"fmt"
	"math"
	"slices"

	"github.com/go-statistics/stats/internal/dd"
)

// EntropyOptions are the options of Entropy.
type EntropyOptions struct {
	// QK, when not nil, is the second distribution: the result is then the
	// relative entropy (Kullback-Leibler divergence) of pk from qk.
	QK []float64
	// Base is the base of the logarithm; 0 means e (nats).
	Base float64
}

// Entropy returns the Shannon entropy -Σ p log p of the distribution pk
// (normalised to sum 1), or the relative entropy Σ p log(p/q) when QK is
// given: scipy.stats.entropy(pk, qk, base). A NaN gives NaN.
func Entropy[D Data](pk D, o *EntropyOptions) (float64, error) {
	if o == nil {
		o = &EntropyOptions{}
	}
	if o.Base < 0 || math.IsNaN(o.Base) {
		return math.NaN(), fmt.Errorf("%w: Entropy: base %v is not positive", ErrParam, o.Base)
	}
	p := floats(pk)
	if o.QK != nil && len(o.QK) != len(p) {
		return math.NaN(), fmt.Errorf("%w: Entropy: %d probabilities against %d", ErrData, len(p), len(o.QK))
	}
	sp := sum(p)
	vec := make([]float64, len(p))
	if o.QK == nil {
		for i, t := range p {
			vec[i] = entr(t / sp)
		}
	} else if r, ok := relEntropyDD(p, o.QK); ok {
		if o.Base != 0 {
			r /= crLog(o.Base)
		}
		return r, nil
	} else {
		sq := sum(o.QK)
		for i, t := range p {
			vec[i] = relEntr(t/sp, o.QK[i]/sq)
		}
	}
	s := sum(vec)
	if o.Base != 0 {
		s /= crLog(o.Base)
	}
	return s, nil
}

// entr is scipy.special.entr: -x log x, 0 at 0, -Inf below.
func entr(x float64) float64 {
	switch {
	case x != x:
		return x
	case x > 0:
		return -x * crLog(x)
	case x == 0:
		return 0
	}
	return math.Inf(-1)
}

// relEntr is scipy.special.rel_entr: x log(x/y), 0 at x = 0, +Inf where
// undefined.
func relEntr(x, y float64) float64 {
	switch {
	case x != x || y != y:
		return math.NaN()
	case x > 0 && y > 0:
		// log x - log y in double-double: log of the exact ratio, which
		// matters when x and y are close (the ratio rounded to float64
		// would lose the digits of its logarithm)
		return x * dd.Sub(dd.LogAccurate(x), dd.LogAccurate(y)).Float()
	case x == 0 && y >= 0:
		return 0
	}
	return math.Inf(1)
}

// DiffEntropyMethod is the estimator of DifferentialEntropy.
type DiffEntropyMethod int

const (
	// DiffEntropyAuto picks van Es for n <= 10, Ebrahimi for n <= 1000 and
	// Vasicek above (SciPy's method="auto").
	DiffEntropyAuto DiffEntropyMethod = iota
	// DiffEntropyVasicek is method="vasicek".
	DiffEntropyVasicek
	// DiffEntropyVanEs is method="van es".
	DiffEntropyVanEs
	// DiffEntropyEbrahimi is method="ebrahimi".
	DiffEntropyEbrahimi
	// DiffEntropyCorrea is method="correa".
	DiffEntropyCorrea
)

// DiffEntropyOptions are the options of DifferentialEntropy.
type DiffEntropyOptions struct {
	// WindowLength is the window m of the spacings; 0 means
	// floor(sqrt(n) + 0.5), SciPy's default.
	WindowLength int
	// Base is the base of the logarithm; 0 means e (nats).
	Base float64
	// Method is the estimator (default DiffEntropyAuto).
	Method DiffEntropyMethod
	// OmitNaN ignores the NaNs; by default a NaN gives NaN.
	OmitNaN bool
}

// DifferentialEntropy estimates the differential entropy of the continuous
// distribution the sample comes from, by the m-spacing estimators:
// scipy.stats.differential_entropy(x, window_length, base, method). It
// needs 2 <= 2m < n.
func DifferentialEntropy[D Data](x D, o *DiffEntropyOptions) (float64, error) {
	if o == nil {
		o = &DiffEntropyOptions{}
	}
	if o.Base < 0 || math.IsNaN(o.Base) {
		return math.NaN(), fmt.Errorf("%w: DifferentialEntropy: base %v is not positive", ErrParam, o.Base)
	}
	if o.Method < DiffEntropyAuto || o.Method > DiffEntropyCorrea {
		return math.NaN(), fmt.Errorf("%w: DifferentialEntropy: unknown method %d", ErrParam, o.Method)
	}
	v := floats(x)
	if o.OmitNaN {
		v = dropNaN(v)
	}
	n := len(v)
	m := o.WindowLength
	if m == 0 {
		m = int(math.Floor(math.Sqrt(float64(n)) + 0.5))
	}
	if !(2 <= 2*m && 2*m < n) {
		return math.NaN(), fmt.Errorf("%w: DifferentialEntropy: window length %d needs 2 <= 2m < n = %d", ErrData, m, n)
	}
	if hasNaN(v) {
		return math.NaN(), nil
	}
	s := slices.Clone(v)
	slices.Sort(s)
	method := o.Method
	if method == DiffEntropyAuto {
		switch {
		case n <= 10:
			method = DiffEntropyVanEs
		case n <= 1000:
			method = DiffEntropyEbrahimi
		default:
			method = DiffEntropyVasicek
		}
	}
	var r float64
	switch method {
	case DiffEntropyVanEs:
		r = vanEsEntropy(s, m)
	case DiffEntropyEbrahimi:
		r = ebrahimiEntropy(s, m)
	case DiffEntropyCorrea:
		r = correaEntropy(s, m)
	default:
		r = vasicekEntropy(s, m)
	}
	if o.Base != 0 {
		r /= crLog(o.Base)
	}
	return r, nil
}

// padded returns X with m copies of its first and last values at each end.
func padded(x []float64, m int) []float64 {
	n := len(x)
	p := make([]float64, n+2*m)
	for i := range m {
		p[i] = x[0]
		p[n+m+i] = x[n-1]
	}
	copy(p[m:], x)
	return p
}

func vasicekEntropy(x []float64, m int) float64 {
	n := len(x)
	p := padded(x, m)
	c := float64(n) / float64(2*m)
	logs := make([]float64, n)
	for i := range logs {
		logs[i] = crLog(c * (p[i+2*m] - p[i]))
	}
	return mean(logs)
}

func vanEsEntropy(x []float64, m int) float64 {
	n := len(x)
	c := float64(n+1) / float64(m)
	logs := make([]float64, n-m)
	for i := range logs {
		logs[i] = crLog(c * (x[i+m] - x[i]))
	}
	term1 := 1 / float64(n-m) * sum(logs)
	inv := make([]float64, n+1-m)
	for i := range inv {
		inv[i] = 1 / float64(m+i)
	}
	return term1 + sum(inv) + crLog(float64(m)) - crLog(float64(n+1))
}

func ebrahimiEntropy(x []float64, m int) float64 {
	n := len(x)
	p := padded(x, m)
	fn, fm := float64(n), float64(m)
	logs := make([]float64, n)
	for k := range logs {
		i := float64(k + 1)
		ci := 2.0
		if i <= fm {
			ci = 1 + (i-1)/fm
		}
		if i >= fn-fm+1 {
			ci = 1 + (fn-i)/fm
		}
		logs[k] = crLog(float64(fn*(p[k+2*m]-p[k])) / float64(ci*fm))
	}
	return mean(logs)
}

func correaEntropy(x []float64, m int) float64 {
	n := len(x)
	p := padded(x, m)
	logs := make([]float64, n)
	diff := make([]float64, 2*m+1)
	for k := range n {
		// window j = i-m .. i+m (1-based i = k+1), padded index j+m-1
		bar := 0.0
		for d := range 2*m + 1 {
			bar += p[k+d]
		}
		bar /= float64(2*m + 1)
		num, den := 0.0, 0.0
		for d := range 2*m + 1 {
			diff[d] = p[k+d] - bar
			num += float64(diff[d] * float64(d-m))
			den += float64(diff[d] * diff[d])
		}
		logs[k] = crLog(num / float64(float64(n)*den))
	}
	return -mean(logs)
}

// relEntropyDD is Σ p log(p/q) of the normalised distributions carried in
// double-double throughout: the normalising sums, the logarithms of the
// exact ratios and the final sum. Two close distributions make the terms
// cancel, and SciPy's float64 result (its normalised probabilities are
// rounded first) then keeps only a few digits; this one stays accurate. It
// reports false when the data need the special cases (a zero, a negative
// value, a NaN or an infinity), left to the float64 path.
func relEntropyDD(p, q []float64) (float64, bool) {
	var sp, sq dd.DD
	for i, t := range p {
		u := q[i]
		if !(t > 0 && u > 0) || math.IsInf(t, 0) || math.IsInf(u, 0) {
			return 0, false
		}
		sp, sq = dd.AddF(sp, t), dd.AddF(sq, u)
	}
	if len(p) == 0 || !(sp.Hi < math.Inf(1) && sq.Hi < math.Inf(1)) {
		return 0, false
	}
	logOf := func(a dd.DD) dd.DD { return dd.AddF(dd.LogAccurate(a.Hi), a.Lo/a.Hi) }
	c := dd.Sub(logOf(sq), logOf(sp))
	var acc dd.DD
	for i, t := range p {
		l := dd.Add(dd.Sub(dd.LogAccurate(t), dd.LogAccurate(q[i])), c)
		acc = dd.Add(acc, dd.Mul(dd.Div(dd.F(t), sp), l))
	}
	return acc.Float(), true
}
