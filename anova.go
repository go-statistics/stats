package stats

import (
	"fmt"
	"math"
	"slices"
)

// FOnewayOptions are the options of FOneway. The zero value (or nil) is
// SciPy's default: the classical one-way ANOVA assuming equal variances.
type FOnewayOptions struct {
	// Welch selects Welch's heteroscedastic ANOVA (SciPy's
	// equal_var=False).
	Welch bool
}

// groupsOf converts the samples, reporting whether any is empty or holds
// a NaN (SciPy then returns NaN).
func groupsOf[D Data](samples []D) (g [][]float64, degenerate bool) {
	g = make([][]float64, len(samples))
	for i, s := range samples {
		g[i] = floats(s)
		if len(g[i]) == 0 || hasNaN(g[i]) {
			degenerate = true
		}
	}
	return g, degenerate
}

// FOneway is the one-way analysis of variance of the null hypothesis that
// the groups have the same population mean: scipy.stats.f_oneway(*samples,
// equal_var=!Welch). At least two groups are required. An empty group, a
// NaN, or groups of one observation each give NaN; when every group is
// constant the statistic is +Inf (NaN if all the values are equal), as in
// SciPy.
func FOneway[D Data](o *FOnewayOptions, samples ...D) (TestResult, error) {
	if o == nil {
		o = &FOnewayOptions{}
	}
	if len(samples) < 2 {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: f_oneway needs at least two groups, got %d", ErrData, len(samples))
	}
	groups, bad := groupsOf(samples)
	allOne := true
	for _, g := range groups {
		if len(g) != 1 {
			allOne = false
		}
	}
	if bad || allOne {
		return TestResult{qnan, qnan}, nil
	}
	var all []float64
	allConst := true
	for _, g := range groups {
		all = append(all, g...)
		if !isConstant(g) {
			allConst = false
		}
	}
	k := float64(len(groups))
	bign := float64(len(all))
	var f, dfn, dfd float64
	if !o.Welch {
		offset := mean(all)
		ad := make([]float64, len(all))
		for i, v := range all {
			ad[i] = v - offset
		}
		s := sum(ad)
		normSS := s * s / bign
		sstot := dotDD(ad, ad) - normSS
		ssbn := 0.0
		for _, g := range groups {
			d := make([]float64, len(g))
			for i, v := range g {
				d[i] = v - offset
			}
			t := sum(d)
			ssbn += t * t / float64(len(g))
		}
		ssbn -= normSS
		sswn := sstot - ssbn
		dfn, dfd = k-1, bign-k
		f = (ssbn / dfn) / (sswn / dfd)
	} else {
		w := make([]float64, len(groups))
		y := make([]float64, len(groups))
		for i, g := range groups {
			y[i] = mean(g)
			w[i] = float64(len(g)) / variance(g, 1)
		}
		sw := sum(w)
		yhat := dotDD(w, y) / sw
		var num, q float64
		for i, g := range groups {
			d := y[i] - yhat
			num += w[i] * d * d
			e := 1 - w[i]/sw
			q += 1 / (float64(len(g)) - 1) * e * e
		}
		num /= k - 1
		den := 1 + 2*(k-2)/(k*k-1)*q
		f = num / den
		dfn, dfd = k-1, (k*k-1)/(3*q)
	}
	if allConst {
		f = math.Inf(1)
		if isConstant(all) {
			f = qnan
		}
	}
	return TestResult{f, F{Dfn: dfn, Dfd: dfd, Scale: 1}.SF(f)}, nil
}

// Bartlett is Bartlett's test of the null hypothesis that the groups have
// equal variances: scipy.stats.bartlett(*samples). At least two groups
// are required; an empty group or a NaN gives NaN.
func Bartlett[D Data](samples ...D) (TestResult, error) {
	if len(samples) < 2 {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: bartlett needs at least two groups, got %d", ErrData, len(samples))
	}
	groups, bad := groupsOf(samples)
	if bad {
		return TestResult{qnan, qnan}, nil
	}
	k := float64(len(groups))
	ni := make([]float64, len(groups))
	wss := make([]float64, len(groups))
	wls := make([]float64, len(groups))
	inv := make([]float64, len(groups))
	for i, g := range groups {
		n := float64(len(g))
		s := variance(g, 1)
		ni[i] = n
		wss[i] = (n - 1) * s
		wls[i] = (n - 1) * math.Log(s)
		inv[i] = 1 / (n - 1)
	}
	ntot := sum(ni)
	spsq := sum(wss) / (ntot - k)
	numer := (ntot-k)*math.Log(spsq) - sum(wls)
	denom := 1 + 1/(3*(k-1))*(sum(inv)-1/(ntot-k))
	t := numer / denom
	p := chi2SF(k-1, t)
	return TestResult{math.Max(t, 0), p}, nil
}

// LeveneCenter is the center of Levene's test (SciPy's center=). The
// zero value is CenterMedian, SciPy's default (the Brown-Forsythe test).
type LeveneCenter int

const (
	CenterMedian  LeveneCenter = iota // center="median"
	CenterMean                        // center="mean": Levene's original test
	CenterTrimmed                     // center="trimmed": the trimmed mean
)

// LeveneOptions are the options of Levene. The zero value (or nil) is
// SciPy's default: center "median"; ProportionToCut 0 means SciPy's 0.05.
type LeveneOptions struct {
	Center LeveneCenter
	// ProportionToCut is the proportion cut from each end for
	// CenterTrimmed (SciPy's proportiontocut; 0 selects 0.05).
	ProportionToCut float64
}

// Levene is Levene's test of the null hypothesis that the groups have
// equal variances: scipy.stats.levene(*samples, center, proportiontocut).
// At least two groups are required; an empty group or a NaN gives NaN.
func Levene[D Data](o *LeveneOptions, samples ...D) (TestResult, error) {
	if o == nil {
		o = &LeveneOptions{}
	}
	if o.Center < CenterMedian || o.Center > CenterTrimmed {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: Levene center %d", ErrParam, int(o.Center))
	}
	cut := o.ProportionToCut
	if cut == 0 {
		cut = 0.05
	}
	if len(samples) < 2 {
		return TestResult{qnan, qnan}, fmt.Errorf("%w: levene needs at least two groups, got %d", ErrData, len(samples))
	}
	groups, bad := groupsOf(samples)
	if bad {
		return TestResult{qnan, qnan}, nil
	}
	k := float64(len(groups))
	ntot := 0.0
	z := make([][]float64, len(groups))
	zbar := make([]float64, len(groups))
	for i, g := range groups {
		var c float64
		switch o.Center {
		case CenterMean:
			c = mean(g)
		case CenterTrimmed:
			s := slices.Clone(g)
			slices.Sort(s)
			lc := int(cut * float64(len(s)))
			if lc > len(s)-lc {
				return TestResult{qnan, qnan}, fmt.Errorf("%w: proportion to cut %v too big", ErrParam, cut)
			}
			c = mean(s[lc : len(s)-lc])
		default:
			c = medianOf(g)
		}
		z[i] = make([]float64, len(g))
		for j, v := range g {
			z[i][j] = math.Abs(v - c)
		}
		zbar[i] = mean(z[i])
		ntot += float64(len(g))
	}
	zb := 0.0
	for i, g := range groups {
		zb += float64(len(g)) * zbar[i]
	}
	zb /= ntot
	dfd := ntot - k
	numer := 0.0
	denom := 0.0
	for i, g := range groups {
		d := zbar[i] - zb
		numer += float64(len(g)) * d * d
		sq := make([]float64, len(g))
		for j := range g {
			e := z[i][j] - zbar[i]
			sq[j] = e * e
		}
		denom += sum(sq)
	}
	numer *= dfd
	dfn := k - 1
	denom *= dfn
	w := numer / denom
	return TestResult{w, F{Dfn: dfn, Dfd: dfd, Scale: 1}.SF(w)}, nil
}

// median is np.median.
func medianOf(a []float64) float64 {
	s := slices.Clone(a)
	slices.Sort(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}
