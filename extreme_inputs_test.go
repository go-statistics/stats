package stats

import (
	"flag"
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/go-numerics/special"
)

// An input that crashes or hangs the caller is a defect whatever the
// value would have been: these tests sweep every registered distribution,
// at every instance of its parity fixture and at extreme parameters, over
// arguments a caller can pass (huge, tiny, subnormal, signed zero, NaN,
// ±Inf, probabilities outside [0, 1]), and require that each method
// returns, without panicking, within a time bound.

// extremeArgs are the points every density, distribution function and
// quantile function is evaluated at.
var extremeArgs = []float64{
	math.NaN(), math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1),
	math.MaxFloat64, -math.MaxFloat64, 1e300, -1e300, 1e200, -1e200,
	1e100, -1e100, 1e40, -1e40, 1e20, -1e20, 0x1p63, -0x1p63, 0x1p53, -0x1p53,
	0x1p53 + 2, 1e15, -1e15, 0x1p31, -0x1p31, 1e8, -1e8,
	math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
	0x1p-1022, -0x1p-1022, 1e-300, -1e-300, 1e-100, 1e-20, 1e-8,
	0.5, 1, -1, 1 - 0x1p-53, 1 + 0x1p-52, 0x1p-53, 2, -2, 100, -100,
}

// extremeParams are the values each parameter is set to, one at a time,
// from a fixture instance; the instances whose constructor accepts them
// are swept as the fixture ones are.
var extremeParams = []float64{
	0, math.SmallestNonzeroFloat64, 1e-300, 1e-100, 1e-20, 1e-8, 0.5, 1, 2,
	1e8, 1e15, 0x1p53, 1e20, 1e100, 1e300, math.MaxFloat64, -1, -1e300,
}

// callTimeout bounds one evaluation; it is generous so that a slow runner
// does not fail the test, while a loop that never ends still does. A call
// that does not return keeps its goroutine running, so the sweep of an
// instance stops at its first one.
var runExtremeParams = flag.Bool("extreme.params", false, "run the extreme-parameter sweep (TestExtremeParamsNoPanic)")

var callTimeout = flag.Duration("extreme.timeout", 20*time.Second, "time bound of one call in the extreme-input sweeps")

// guarded calls f, returning a description of the panic it raised or of
// its not returning within callTimeout, "" if it returned.
func guarded(f func()) (failure string) {
	done := make(chan string, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- fmt.Sprintf("panic: %v", r)
			}
		}()
		f()
		done <- ""
	}()
	select {
	case s := <-done:
		return s
	case <-time.After(*callTimeout):
		return hangPrefix + callTimeout.String()
	}
}

// hangPrefix starts the failure guarded reports for a call that did not
// return.
const hangPrefix = "did not return within "

type namedCall struct {
	name string
	f    func()
}

// distCalls lists every method call of the sweep on the distribution v.
func distCalls(v any) []namedCall {
	var calls []namedCall
	fns := distFns(v)
	names := make([]string, 0, len(fns))
	for k := range fns {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		f := fns[k]
		for _, x := range extremeArgs {
			calls = append(calls, namedCall{fmt.Sprintf("%s(%v)", k, x), func() { f(x) }})
		}
	}
	d := v.(momenter)
	scalar := map[string]func() float64{
		"Mean": d.Mean, "Var": d.Var, "Std": d.Std, "Skew": d.Skew,
		"Kurtosis": d.Kurtosis, "Median": d.Median, "Entropy": d.Entropy,
	}
	for _, k := range []string{"Mean", "Var", "Std", "Skew", "Kurtosis", "Median", "Entropy"} {
		f := scalar[k]
		calls = append(calls, namedCall{k + "()", func() { f() }})
	}
	for _, n := range []int{-1, 0, 1, 2, 5, 40} {
		calls = append(calls, namedCall{fmt.Sprintf("Moment(%d)", n), func() { d.Moment(n) }})
	}
	for _, c := range []float64{math.NaN(), -1, 0, 0.5, 1 - 0x1p-53, 1, 2} {
		calls = append(calls, namedCall{fmt.Sprintf("Interval(%v)", c), func() { d.Interval(c) }})
	}
	return calls
}

// sweepDistribution makes every call of distCalls(v) and returns the
// failures, stopping at the first call that does not return.
func sweepDistribution(v any) []string {
	var fails []string
	for _, c := range distCalls(v) {
		if s := guarded(c.f); s != "" {
			fails = append(fails, c.name+": "+s)
			if strings.HasPrefix(s, hangPrefix) {
				break
			}
		}
	}
	return fails
}

// TestExtremeInputsNoPanic sweeps the fixture instances.
func TestExtremeInputsNoPanic(t *testing.T) {
	for _, pf := range loadParity(t) {
		e := distTypes[pf.Go]
		t.Run(pf.SciPy, func(t *testing.T) {
			t.Parallel()
			for _, in := range pf.Instances {
				v, err := e.construct(in.Params.floats())
				if err != nil {
					t.Fatal(err)
				}
				for _, f := range sweepDistribution(v) {
					t.Errorf("%s %v %s", pf.Go, in.Params, f)
				}
			}
		})
	}
}

// TestExtremeParamsNoPanic sweeps the instances obtained by setting one
// parameter of the first fixture instance to an extreme value. Invalid
// parameters must give an error (or NaN from every method), never a
// panic, in the constructor too.
//
// It takes minutes (some deep-tail quantiles at extreme parameters take
// seconds each), so it runs on request:
//
//	go test -run TestExtremeParamsNoPanic -extreme.params [-extreme.timeout=3s]
//
// TestHangRegressions keeps the cases it found.
func TestExtremeParamsNoPanic(t *testing.T) {
	if !*runExtremeParams {
		t.Skip("the extreme-parameter sweep runs with -extreme.params")
	}
	for _, pf := range loadParity(t) {
		e := distTypes[pf.Go]
		t.Run(pf.SciPy, func(t *testing.T) {
			t.Parallel()
			if len(pf.Instances) == 0 {
				return
			}
			base := pf.Instances[0].Params.floats()
			keys := make([]string, 0, len(base))
			for k := range base {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				for _, pv := range extremeParams {
					p := map[string]float64{}
					for kk, vv := range base {
						p[kk] = vv
					}
					p[k] = pv
					var v any
					var err error
					if s := guarded(func() { v, err = e.construct(p) }); s != "" {
						t.Errorf("%s %v: construct: %s", pf.Go, p, s)
						continue
					}
					if err != nil {
						continue
					}
					for _, f := range sweepDistribution(v) {
						t.Errorf("%s %v %s", pf.Go, p, f)
					}
				}
			}
		})
	}
}

// TestVonMisesHugeArgument pins the values at arguments far beyond the
// period, which crashed sincosSmall before the reduction was done to
// full precision: the density is that of the argument reduced modulo 2π
// exactly (as cos(x) is in SciPy's exp(kappa*cos(x))/(2π I0(kappa))).
func TestVonMisesHugeArgument(t *testing.T) {
	d := VonMises{Kappa: 2, Scale: 1}
	for _, x := range []float64{1e300, -1e300, math.MaxFloat64, -math.MaxFloat64, 1e22, 0x1p60, 1e15} {
		want := math.Exp(2*(math.Cos(x)-1)) / (2 * math.Pi * special.I0e(2))
		got := d.PDF(x)
		if math.Abs(got-want) > 1e-14*want {
			t.Errorf("PDF(%v) = %v, want %v", x, got, want)
		}
		if lg := d.LogPDF(x); math.Abs(lg-math.Log(want)) > 1e-14*math.Abs(math.Log(want))+1e-15 {
			t.Errorf("LogPDF(%v) = %v, want %v", x, lg, math.Log(want))
		}
		if c := d.CDF(x); math.IsNaN(c) {
			t.Errorf("CDF(%v) = NaN", x)
		}
	}
}
