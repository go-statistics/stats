package stats

import "math"

// invertCDF returns the x in (lo, hi) with cdf(x) = p, for a distribution
// with no closed-form quantile, to about an ulp: a bracket grown
// geometrically from the guess x0, then Newton steps on log cdf (or on log
// sf when p > 1/2, with q = 1-p given exactly by the caller), each kept
// inside the bracket and replaced by bisection when it is not, until the
// bracket is two adjacent floats or the step is below an ulp.
//
// f is the cdf when upper is false and the sf when it is true; target is
// p or q accordingly; pdf is the density.
func invertCDF(target float64, upper bool, f, pdf func(float64) float64, x0, lo, hi float64) float64 {
	// g(x) < 0 below the root, > 0 above
	g := func(x float64) float64 {
		v := f(x)
		if upper {
			return target - v
		}
		return v - target
	}
	x := x0
	if !(x > lo && x < hi) {
		x = midpoint(lo, hi)
	}
	a, b := lo, hi // g(a) <= 0 <= g(b) by the support
	gx := g(x)
	if gx < 0 {
		a = x
	} else {
		b = x
	}
	// grow towards the root from x until it is bracketed by finite values
	step := math.Max(math.Abs(x)*0.5, 1)
	for i := 0; i < 2100 && (math.IsInf(a, 0) || math.IsInf(b, 0)); i++ {
		if math.IsInf(b, 1) {
			y := a + step
			if g(y) >= 0 {
				b = y
			} else {
				a = y
			}
		} else {
			y := b - step
			if g(y) <= 0 {
				a = y
			} else {
				b = y
			}
		}
		step *= 2
	}
	x = midpoint(a, b) // x0 is now an end of the bracket
	for i := 0; i < 200; i++ {
		v := f(x)
		gx := v - target
		if upper {
			gx = target - v
		}
		if gx == 0 {
			return x
		}
		if gx < 0 {
			a = x
		} else {
			b = x
		}
		// Newton on log f: Δ = (log v - log target)/(d log f/dx)
		d := pdf(x) / v
		if upper {
			d = -d
		}
		nx := x - (math.Log(v)-math.Log(target))/d
		if !(nx > a && nx < b) {
			nx = midpoint(a, b)
		}
		if nx == x || math.Nextafter(a, b) >= b {
			return x
		}
		if math.Abs(nx-x) <= 2e-16*math.Abs(x) {
			return nx
		}
		x = nx
	}
	return x
}

// midpoint is (a+b)/2 without overflow, and in the middle of the
// exponents when the bracket spans many binades of positive numbers.
func midpoint(a, b float64) float64 {
	if math.IsInf(b, 1) {
		return math.Max(2*a, 1)
	}
	if math.IsInf(a, -1) {
		return math.Min(2*b, -1)
	}
	if a > 0 && b/a > 4 {
		return math.Sqrt(a) * math.Sqrt(b)
	}
	return a/2 + b/2
}
