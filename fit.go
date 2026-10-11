package stats

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/go-numerics/optimize"
)

// FitOptions configures the FitXxx functions.
type FitOptions struct {
	// Fixed holds parameters that are not estimated but held at the given
	// value, by struct field name: {"Loc": 0} is SciPy's floc=0,
	// {"Scale": 2} fscale=2, {"A": 1.5} fa=1.5 (or f0=1.5).
	Fixed map[string]float64
}

var errAllFixed = fmt.Errorf("%w: every parameter is fixed; there is nothing to fit", ErrParam)

// fixed checks that o fixes only parameters the family has, and returns
// the fixed values (never nil).
func (o *FitOptions) fixed(family string, names ...string) (map[string]float64, error) {
	out := map[string]float64{}
	if o == nil {
		return out, nil
	}
	for k, v := range o.Fixed {
		found := false
		for _, n := range names {
			if n == k {
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("%w: %s has no parameter %q (it has %s)", ErrParam, family, k, strings.Join(names, ", "))
		}
		out[k] = v
	}
	if len(out) == len(names) {
		return nil, errAllFixed
	}
	return out, nil
}

// checkFitData rejects what SciPy's fit rejects: no data, or a non-finite
// value.
func checkFitData(v []float64) error {
	if len(v) == 0 {
		return fmt.Errorf("%w: no data to fit", ErrData)
	}
	for _, x := range v {
		if !finite(x) {
			return fmt.Errorf("%w: the data contains non-finite values", ErrData)
		}
	}
	return nil
}

// mleProblem is a numerical maximum-likelihood fit: the parameters by name,
// which of them must be positive (they are optimised on a log scale), a
// starting point, and the log-likelihood (NaN or -Inf where the parameters
// are invalid or a point falls outside the support).
type mleProblem struct {
	names    []string
	positive []bool
	start    []float64
	loglik   func(theta []float64) float64
}

// fitMLE maximises the log-likelihood over the parameters not fixed, with
// Nelder-Mead (SciPy's default fit optimiser, optimize.fmin) restarted
// from its own optimum until a restart no longer improves it, and with
// tolerances far tighter than fmin's defaults (xtol = ftol = 1e-4), so that
// the result is the optimum to about 1e-10 relative rather than SciPy's
// four digits. It returns the full parameter vector.
func fitMLE(p mleProblem, fixed map[string]float64) ([]float64, error) {
	n := len(p.names)
	theta := append([]float64(nil), p.start...)
	var free []int
	for i, name := range p.names {
		if v, ok := fixed[name]; ok {
			theta[i] = v
			continue
		}
		free = append(free, i)
	}
	full := make([]float64, n)
	unpack := func(u []float64) []float64 {
		copy(full, theta)
		for j, i := range free {
			if p.positive[i] {
				full[i] = math.Exp(u[j])
			} else {
				full[i] = u[j]
			}
		}
		return full
	}
	obj := func(u []float64) float64 {
		ll := p.loglik(unpack(u))
		if ll != ll || math.IsInf(ll, 0) {
			return math.Inf(1)
		}
		return -ll
	}
	u := make([]float64, len(free))
	for j, i := range free {
		if p.positive[i] {
			u[j] = math.Log(theta[i])
		} else {
			u[j] = theta[i]
		}
	}
	best := obj(u)
	if math.IsInf(best, 1) {
		return nil, fmt.Errorf("%w: the starting point has zero likelihood", ErrFit)
	}
	for restart := 0; restart < 6; restart++ {
		r, _ := optimize.Minimize(optimize.Problem{Func: obj}, u, optimize.NelderMead, &optimize.Options{
			Xatol: 1e-12, Fatol: 1e-13 * math.Max(1, math.Abs(best)), MaxIter: 1000 * len(u), MaxFev: 2000 * len(u), Adaptive: len(u) > 2,
		})
		improved := best - r.Fun
		if !(r.Fun <= best) {
			break
		}
		u, best = r.X, r.Fun
		if improved <= 1e-14*math.Max(1, math.Abs(best)) {
			break
		}
	}
	return append([]float64(nil), unpack(u)...), nil
}

// fitLocScaleSupport is SciPy's starting point for loc and scale given the
// shape (rv_continuous._fit_loc_scale_support): the method of moments, moved
// so that the support covers the data when it does not.
func fitLocScaleSupport(v []float64, s contStd) (loc, scale float64) {
	mu, mu2 := s.mean(), s.variance()
	muhat, mu2hat := mean(v), variance(v, 0)
	scale = math.Sqrt(mu2hat / mu2)
	loc = muhat - scale*mu
	if !finite(loc) {
		loc = 0
	}
	if !(finite(scale) && scale > 0) {
		scale = 1
	}
	a, b := s.support()
	width := b - a
	lo, hi := minMax(v)
	if !(width > 0) {
		return loc, scale
	}
	ahat, bhat := loc+a*scale, loc+b*scale
	if ahat < lo && hi < bhat {
		return loc, scale
	}
	margin := (hi - lo) * 0.1
	if !math.IsInf(width, 1) {
		return (lo - a) - margin, (hi - lo + 2*margin) / width
	}
	if !math.IsInf(a, -1) {
		return (lo - a) - margin, 1
	}
	return (hi - b) + margin, 1
}

func minMax(v []float64) (lo, hi float64) {
	lo, hi = v[0], v[0]
	for _, x := range v[1:] {
		lo = math.Min(lo, x)
		hi = math.Max(hi, x)
	}
	return lo, hi
}

// contLogLik is the log-likelihood of v under the standard form s moved by
// loc and scale, with the boundary rules of LogPDF; NaN for invalid
// parameters.
func contLogLik(v []float64, s contStd, loc, scale float64) float64 {
	if !(scale > 0) || !s.valid() {
		return math.NaN()
	}
	a, b := s.support()
	ll := 0.0
	for _, x := range v {
		z := (x - loc) / scale
		if z < a || z > b || math.IsInf(z, 0) {
			return math.Inf(-1)
		}
		ll += s.logpdf(z)
	}
	return ll - float64(len(v))*math.Log(scale)
}

// fitContinuous is the numerical maximum-likelihood fit of a continuous
// family with the given shape parameters (names and which are positive),
// started as SciPy starts it: shapes from start, loc and scale from
// fitLocScaleSupport. It returns the shapes, loc and scale.
func fitContinuous(v []float64, fixed map[string]float64, names []string, positive []bool, start []float64, mk func(shapes []float64) contStd) ([]float64, float64, float64, error) {
	k := len(names)
	sh := append([]float64(nil), start...)
	for i, n := range names {
		if f, ok := fixed[n]; ok {
			sh[i] = f
		}
	}
	loc, scale := fitLocScaleSupport(v, mk(sh))
	if f, ok := fixed["Loc"]; ok {
		loc = f
	}
	if f, ok := fixed["Scale"]; ok {
		scale = f
	}
	p := mleProblem{
		names:    append(append([]string(nil), names...), "Loc", "Scale"),
		positive: append(append([]bool(nil), positive...), false, true),
		start:    append(append([]float64(nil), sh...), loc, scale),
		loglik: func(th []float64) float64 {
			return contLogLik(v, mk(th[:k]), th[k], th[k+1])
		},
	}
	th, err := fitMLE(p, fixed)
	if err != nil {
		return nil, 0, 0, err
	}
	return th[:k], th[k], th[k+1], nil
}

// sortedCopy returns a sorted copy of v.
func sortedCopy(v []float64) []float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s
}
