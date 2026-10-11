package stats

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/go-ndarray/ndarray/random"
)

// TestDistributionContract holds every registered distribution, at every
// instance of its parity fixture, to the conventions of Continuous and
// Discrete: boundary values, NaN propagation, quantile round trips,
// consistency of the log forms, and the behaviour of invalid parameters.
func TestDistributionContract(t *testing.T) {
	nan := math.NaN()
	inf := math.Inf(1)
	for _, pf := range loadParity(t) {
		e := distTypes[pf.Go]
		t.Run(pf.SciPy, func(t *testing.T) {
			for _, in := range pf.Instances {
				v, err := e.construct(in.Params.floats())
				if err != nil {
					t.Fatal(err)
				}
				fns := distFns(v)
				d := v.(momenter)
				name := pf.Go
				for k, f := range fns {
					if r := f(nan); !math.IsNaN(r) {
						t.Errorf("%s %s(NaN) = %v", name, k, r)
					}
				}
				lo, hi := d.Support()
				check := func(what string, got, want float64) {
					t.Helper()
					if !(got == want || math.IsNaN(got) && math.IsNaN(want)) {
						t.Errorf("%s %v %s = %v, want %v", name, in.Params, what, got, want)
					}
				}
				check("cdf(-inf)", fns["cdf"](-inf), 0)
				check("cdf(+inf)", fns["cdf"](inf), 1)
				check("sf(-inf)", fns["sf"](-inf), 1)
				check("sf(+inf)", fns["sf"](inf), 0)
				check("logcdf(-inf)", fns["logcdf"](-inf), -inf)
				check("logcdf(+inf)", fns["logcdf"](inf), 0)
				check("logsf(-inf)", fns["logsf"](-inf), 0)
				check("logsf(+inf)", fns["logsf"](inf), -inf)
				dens, ldens := fns["pdf"], fns["logpdf"]
				if pf.Kind == "discrete" {
					dens, ldens = fns["pmf"], fns["logpmf"]
				}
				check("density(-inf)", dens(-inf), 0)
				check("density(+inf)", dens(inf), 0)
				check("logdensity(-inf)", ldens(-inf), -inf)
				check("logdensity(+inf)", ldens(inf), -inf)
				if !math.IsInf(lo, 0) {
					below := lo - 1 - math.Abs(lo)*1e-6
					check("density(below)", dens(below), 0)
					check("logdensity(below)", ldens(below), -inf)
					check("cdf(below)", fns["cdf"](below), 0)
					check("sf(below)", fns["sf"](below), 1)
					check("logcdf(below)", fns["logcdf"](below), -inf)
					check("logsf(below)", fns["logsf"](below), 0)
				}
				if !math.IsInf(hi, 0) {
					above := hi + 1 + math.Abs(hi)*1e-6
					check("density(above)", dens(above), 0)
					check("logdensity(above)", ldens(above), -inf)
					check("cdf(above)", fns["cdf"](above), 1)
					check("sf(above)", fns["sf"](above), 0)
					check("logcdf(above)", fns["logcdf"](above), 0)
					check("logsf(above)", fns["logsf"](above), -inf)
				}
				if pf.Kind == "continuous" {
					check("ppf(0)", fns["ppf"](0), lo)
					check("ppf(1)", fns["ppf"](1), hi)
					check("isf(0)", fns["isf"](0), hi)
					check("isf(1)", fns["isf"](1), lo)
				} else {
					check("ppf(0)", fns["ppf"](0), lo-1)
					check("ppf(1)", fns["ppf"](1), hi)
					check("isf(0)", fns["isf"](0), hi)
					check("isf(1)", fns["isf"](1), lo-1)
				}
				for _, p := range []float64{-0.5, 1.5, nan} {
					check("ppf(out)", fns["ppf"](p), nan)
					check("isf(out)", fns["isf"](p), nan)
				}
				il, ih := d.Interval(1)
				check("interval(1).lo", il, fns["ppf"](0))
				check("interval(1).hi", ih, fns["ppf"](1))
				il, ih = d.Interval(1.5)
				check("interval(1.5).lo", il, nan)
				check("interval(1.5).hi", ih, nan)
				check("moment(-1)", d.Moment(-1), nan)
				check("moment(0)", d.Moment(0), 1)
				for _, p := range []float64{1e-12, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-6} {
					x := fns["ppf"](p)
					xs := fns["isf"](p)
					if pf.Kind == "continuous" {
						if c := fns["cdf"](x); math.Abs(c-p) > 1e-9*p+1e-15 {
							t.Errorf("%s %v cdf(ppf(%g)) = %v", name, in.Params, p, c)
						}
						if s := fns["sf"](xs); math.Abs(s-p) > 1e-9*p+1e-15 {
							t.Errorf("%s %v sf(isf(%g)) = %v", name, in.Params, p, s)
						}
					} else {
						if fns["cdf"](x) < p || (x > lo && fns["cdf"](x-1) >= p) {
							t.Errorf("%s %v ppf(%g) = %v is not the smallest k with cdf(k) >= p", name, in.Params, p, x)
						}
						if fns["sf"](xs) > p || (xs > lo && fns["sf"](xs-1) <= p) {
							t.Errorf("%s %v isf(%g) = %v is not the smallest k with sf(k) <= q", name, in.Params, p, xs)
						}
					}
					for _, y := range []float64{x, xs} {
						if math.IsInf(y, 0) {
							continue
						}
						for _, pair := range [][2]string{{"cdf", "logcdf"}, {"sf", "logsf"}} {
							a, b := fns[pair[0]](y), math.Exp(fns[pair[1]](y))
							if math.Abs(a-b) > 1e-12*a+1e-300 {
								t.Errorf("%s %v exp(%s(%v)) = %v, %s = %v", name, in.Params, pair[1], y, b, pair[0], a)
							}
						}
						a, b := dens(y), math.Exp(ldens(y))
						if math.Abs(a-b) > 1e-12*a+1e-300 {
							t.Errorf("%s %v exp(logdensity(%v)) = %v, density = %v", name, in.Params, y, b, a)
						}
					}
				}
				if d.Validate() != nil {
					t.Errorf("%s %v: Validate on a valid instance: %v", name, in.Params, d.Validate())
				}
				if _, err := d.Rvs(random.DefaultRNG(1), -1); err == nil {
					t.Errorf("%s: Rvs with a negative size did not fail", name)
				}
			}
			// The zero value and the invalid instances: NaN everywhere, an
			// error from Validate and Rvs.
			zero := reflect.Zero(e.typ).Interface()
			bads := []any{zero}
			for _, p := range pf.Invalid {
				v, err := e.construct(p.floats())
				if !errors.Is(err, ErrParam) {
					t.Errorf("%s %v: constructor error %v does not wrap ErrParam", pf.Go, p, err)
				}
				bads = append(bads, v)
			}
			for _, v := range bads {
				d := v.(momenter)
				if pf.Kind == "discrete" && reflect.DeepEqual(v, zero) && d.Validate() == nil {
					continue // a discrete zero value may be valid (Bernoulli{P: 0})
				}
				if !errors.Is(d.Validate(), ErrParam) {
					t.Errorf("%s %+v: Validate = %v", pf.Go, v, d.Validate())
				}
				if _, err := d.Rvs(random.DefaultRNG(1), 3); err == nil {
					t.Errorf("%s %+v: Rvs accepted invalid parameters", pf.Go, v)
				}
				for k, f := range distFns(v) {
					if r := f(0.5); !math.IsNaN(r) {
						t.Errorf("%s %+v: %s = %v, want NaN", pf.Go, v, k, r)
					}
				}
				vals := []float64{d.Mean(), d.Var(), d.Std(), d.Skew(), d.Kurtosis(), d.Median(), d.Entropy(), d.Moment(2)}
				a, b := d.Support()
				vals = append(vals, a, b)
				for i, r := range vals {
					if !math.IsNaN(r) {
						t.Errorf("%s %+v: moment-like value #%d = %v, want NaN", pf.Go, v, i, r)
					}
				}
			}
		})
	}
}

func TestMomentLocScale(t *testing.T) {
	// E[(2 + 3Z)^3] for Z standard normal: 8 + 3·2·9·1 = 62 (odd terms vanish)
	if got := momentLocScale(3, 2, 3, normalStd{}.munp); math.Abs(got-62) > 1e-12 {
		t.Errorf("momentLocScale = %v", got)
	}
	if got := (Normal{Loc: 2, Scale: 3}).Moment(4); math.Abs(got-(16+6*4*9+81*3)) > 1e-12 {
		t.Errorf("Moment(4) = %v", got)
	}
}
