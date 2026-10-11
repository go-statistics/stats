package stats

import (
	"math"

	"github.com/go-statistics/stats/internal/dd"
)

// The null distributions of the Kolmogorov-Smirnov statistics:
//
//   - smirnovSF: the one-sided statistic D+_n (scipy.special.smirnov,
//     scipy.stats.ksone.sf), by the Birnbaum-Tingey sum of positive terms,
//     each formed in double-double;
//   - kolmogorovSF: the limit of sqrt(n)·D_n (scipy.special.kolmogorov,
//     scipy.stats.kstwobign.sf), a port of xsf's _kolmogorov;
//   - kolmogn: the two-sided statistic D_n (scipy.stats.kstwo), a port of
//     SciPy's _kolmogn (Simard & L'Ecuyer 2011: Ruben-Gambino closed
//     forms, the Durbin matrix in Marsaglia-Tsang-Wang's form, Pomeranz's
//     recursion, Pelz-Good's asymptotic series, and 2·smirnov in the upper
//     tail).

// smirnovSF is P(D+_n >= x) for the one-sided Kolmogorov-Smirnov statistic
// of a sample of size n: scipy.special.smirnov(n, x), the special cases as
// in xsf's _smirnov.
func smirnovSF(n int, x float64) float64 {
	switch {
	case x != x:
		return nan
	case x <= 0:
		return 1
	case x >= 1:
		return 0
	case n == 1:
		return 1 - x
	}
	fn := float64(n)
	nx := dd.MulFF(fn, x)
	nxfl := math.Floor(nx.Hi)
	if nx.Hi == nxfl && nx.Lo < 0 {
		nxfl--
	}
	alpha := dd.AddF(nx, -nxfl).Float()
	if alpha >= 1 {
		nxfl++
		alpha = 0
	}
	if nxfl == 0 || (nxfl == 1 && alpha == 0) {
		// x <= 1/n: cdf = x·(1+x)^(n-1)
		return 1 - x*dd.Exp(dd.MulF(log1pDD(dd.F(x)), fn-1))
	}
	if -2*fn*x*x < -7.451332191019412076235245305675398106812e2 {
		return 0
	}
	if nxfl >= fn-1 {
		// x >= 1 - 1/n: sf = (1-x)^n
		return dd.Exp(dd.MulF(log1pDD(dd.F(-x)), fn))
	}
	if n > 1000000 {
		logp := -math.Pow(6*fn*x+1, 2) / 18 / fn
		if logp < -math.Ln2 {
			return math.Exp(logp)
		}
		return 1 + math.Expm1(logp)
	}
	// Lower sum: sf = x·Σ_{j=0}^{J} C(n,j) (1-x-j/n)^(n-j) (x+j/n)^(j-1),
	// J the last j with 1-x-j/n > 0.
	last := int(fn - nxfl - 1)
	if alpha == 0 {
		last--
	}
	logn := dd.Log(fn)
	var logC dd.DD
	var s dd.DD
	for j := 0; j <= last; j++ {
		fj := float64(j)
		if j > 0 {
			logC = dd.Add(logC, dd.Sub(dd.Log(fn-fj+1), dd.Log(fj)))
		}
		// 1 - x - j/n = ((n-j) - n·x)/n and x + j/n = (n·x + j)/n
		a := dd.Sub(dd.F(fn-fj), nx)
		b := dd.AddF(nx, fj)
		e := dd.Add(logC, dd.MulF(dd.Sub(logDD(a), logn), fn-fj))
		e = dd.Add(e, dd.MulF(dd.Sub(logDD(b), logn), fj-1))
		t := dd.Exp(e)
		s = dd.AddF(s, t)
		if t != 0 && 4*float64(last+1-j)*t < 0x1p-52*s.Hi && j != last {
			break
		}
		if t == 0 && j > n/2 {
			break
		}
	}
	return clip01(dd.MulF(s, x).Float())
}

// kolmogorovSF is the survival function of the limiting distribution of
// sqrt(n)·D_n: scipy.special.kolmogorov(x) (xsf's _kolmogorov).
func kolmogorovSF(x float64) float64 {
	switch {
	case x != x:
		return nan
	case x <= math.Pi/math.Sqrt(746*8):
		return 1
	}
	if x <= 0.82 {
		w := math.Sqrt(2*math.Pi) / x
		logu8 := -math.Pi * math.Pi / (x * x)
		u := math.Exp(logu8 / 8)
		var p float64
		if u == 0 {
			p = math.Exp(logu8/8 + math.Log(w))
		} else {
			u8 := math.Exp(logu8)
			u8cub := math.Pow(u8, 3)
			p = 1 + u8cub
			p = 1 + u8*u8*p
			p = 1 + u8*p
			p = w * u * p
		}
		return clip01(1 - p)
	}
	v := math.Exp(-2 * x * x)
	vsq := v * v
	v3 := math.Pow(v, 3)
	p := 1 - v3*v3*v
	p = 1 - v3*vsq*p
	p = 1 - v3*p
	return clip01(2 * v * p)
}

// kstwoSF is P(D_n > d): scipy.stats.kstwo.sf(d, n), support (1/(2n), 1).
func kstwoSF(n int, d float64) float64 {
	switch {
	case n < 1 || d != d:
		return nan
	case d <= 0.5/float64(n):
		return 1
	case d >= 1:
		return 0
	}
	return kolmogn(n, d, false)
}

// selectProb returns the CDF or the SF, clipped to [0, 1].
func selectProb(cdfp, sfp float64, cdf bool) float64 {
	if cdf {
		return clip01(cdfp)
	}
	return clip01(sfp)
}

// logNFactorialDivNPowN is log(n!/n^n) by Stirling's series.
func logNFactorialDivNPowN(n float64) float64 {
	coeffs := [...]float64{-2.955065359477124183e-2, 6.4102564102564102564e-3,
		-1.9175269175269175269e-3, 8.4175084175084175084e-4,
		-5.952380952380952381e-4, 7.9365079365079365079e-4,
		-2.7777777777777777778e-3, 8.3333333333333333333e-2}
	rn := 1 / n
	z := rn / n
	p := 0.0
	for _, c := range coeffs {
		p = p*z + c
	}
	return math.Log(n)/2 - n + math.Log(2*math.Pi)/2 + rn*p
}

// kolmogn is SciPy's _kolmogn(n, x, cdf): the CDF (or SF) of the two-sided
// statistic D_n, for 0 < x < 1.
func kolmogn(n int, x float64, cdf bool) float64 {
	fn := float64(n)
	t := fn * x
	if t <= 1 {
		if t <= 0.5 {
			return selectProb(0, 1, cdf)
		}
		var prob float64
		if n <= 140 {
			prob = 1
			for i := 1; i <= n; i++ {
				prob *= float64(i) * (1 / fn) * (2*t - 1)
			}
		} else {
			prob = math.Exp(logNFactorialDivNPowN(fn) + fn*math.Log(2*t-1))
		}
		return selectProb(prob, 1-prob, cdf)
	}
	if t >= fn-1 {
		prob := 2 * math.Pow(1-x, fn)
		return selectProb(1-prob, prob, cdf)
	}
	if x >= 0.5 {
		prob := 2 * smirnovSF(n, x)
		return selectProb(1-prob, prob, cdf)
	}
	nxsquared := t * x
	if n <= 140 {
		if nxsquared <= 0.754693 {
			prob := kolmogDurbin(n, x)
			return selectProb(prob, 1-prob, cdf)
		}
		if nxsquared <= 4 {
			prob := kolmogPomeranz(n, x)
			return selectProb(prob, 1-prob, cdf)
		}
		prob := 2 * smirnovSF(n, x)
		return selectProb(1-prob, prob, cdf)
	}
	if !cdf {
		if nxsquared >= 370 {
			return 0
		}
		if nxsquared >= 2.2 {
			return clip01(2 * smirnovSF(n, x))
		}
	}
	var cdfprob float64
	switch {
	case nxsquared >= 18:
		cdfprob = 1
	case n <= 100000 && fn*math.Pow(x, 1.5) <= 1.4:
		cdfprob = kolmogDurbin(n, x)
	default:
		cdfprob = kolmogPelzGood(n, x)
	}
	return selectProb(cdfprob, 1-cdfprob, cdf)
}

const (
	ks2p128  = 0x1p128
	ks2m128  = 0x1p-128
	ksExp128 = 128
)

// kolmogDurbin is Pr(D_n <= d) by Durbin's matrix (Marsaglia, Tsang and
// Wang 2003): SciPy's _kolmogn_DMTW, for 1/(2n) < d < 1.
func kolmogDurbin(n int, d float64) float64 {
	nd := float64(n) * d
	k := int(math.Ceil(nd))
	h := float64(k) - nd
	m := 2*k - 1
	H := make([]float64, m*m)
	v := make([]float64, m)
	w := make([]float64, m)
	fac := 1.0
	for j := 1; j <= m; j++ {
		v[j-1] = 1 - math.Pow(h, float64(j))
		w[j-1] = fac
		fac /= float64(j)
		v[j-1] *= fac
	}
	tt := math.Pow(math.Max(2*h-1, 0), float64(m)) - 2*math.Pow(h, float64(m))
	v[m-1] = (1 + tt) * fac
	for i := 1; i < m; i++ {
		for r := i - 1; r < m; r++ {
			H[r*m+i] = w[r-i+1]
		}
	}
	for r := 0; r < m; r++ {
		H[r*m] = v[r]
	}
	for c := 0; c < m; c++ {
		H[(m-1)*m+c] = v[m-1-c]
	}
	pw := make([]float64, m*m)
	for i := 0; i < m; i++ {
		pw[i*m+i] = 1
	}
	tmp := make([]float64, m*m)
	matmul := func(dst, a, b []float64) {
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				s := 0.0
				for l := 0; l < m; l++ {
					s += a[i*m+l] * b[l*m+j]
				}
				dst[i*m+j] = s
			}
		}
	}
	expnt, hexpnt := 0, 0
	for nn := n; nn > 0; nn /= 2 {
		if nn%2 == 1 {
			matmul(tmp, pw, H)
			pw, tmp = tmp, pw
			expnt += hexpnt
		}
		matmul(tmp, H, H)
		H, tmp = tmp, H
		hexpnt *= 2
		if math.Abs(H[(k-1)*m+k-1]) > ks2p128 {
			for i := range H {
				H[i] /= ks2p128
			}
			hexpnt += ksExp128
		}
	}
	p := pw[(k-1)*m+k-1]
	fn := float64(n)
	for i := 1; i <= n; i++ {
		p = float64(i) * p / fn
		if math.Abs(p) < ks2m128 {
			p *= ks2p128
			expnt -= ksExp128
		}
	}
	if expnt != 0 {
		p = math.Ldexp(p, expnt)
	}
	return clip01(p)
}

// pomeranzBounds is SciPy's _pomeranz_compute_j1j2.
func pomeranzBounds(i, n, ll, ceilf, roundf int) (int, int) {
	var j1, j2 int
	if i == 0 {
		j1, j2 = -ll-ceilf-1, ll+ceilf-1
	} else {
		ip1div2, ip1mod2 := (i+1)/2, (i+1)%2
		switch {
		case ip1mod2 == 0 && ip1div2 == n+1:
			j1, j2 = n-ll-ceilf-1, n+ll+ceilf-1
		case ip1mod2 == 0:
			j1, j2 = ip1div2-1-ll-roundf-1, ip1div2+ll-1+ceilf-1
		default:
			j1, j2 = ip1div2-1-ll-1, ip1div2+ll+roundf-1
		}
	}
	return max(j1+2, 0), min(j2, n)
}

// kolmogPomeranz is Pr(D_n <= x) by Pomeranz's recursion: SciPy's
// _kolmogn_Pomeranz.
func kolmogPomeranz(n int, x float64) float64 {
	fn := float64(n)
	t := fn * x
	ll := int(math.Floor(t))
	f := t - float64(ll)
	g := math.Min(f, 1-f)
	ceilf, roundf := 0, 0
	if f > 0 {
		ceilf = 1
	}
	if f > 0.5 {
		roundf = 1
	}
	npwrs := 2 * (ll + 1)
	gpower := make([]float64, npwrs)
	twogpower := make([]float64, npwrs)
	onem2gpower := make([]float64, npwrs)
	gpower[0], twogpower[0], onem2gpower[0] = 1, 1, 1
	gn, tgn, omn := g/fn, 2*g/fn, (1-2*g)/fn
	for m := 1; m < npwrs; m++ {
		fm := float64(m)
		gpower[m] = gpower[m-1] * gn / fm
		twogpower[m] = twogpower[m-1] * tgn / fm
		onem2gpower[m] = onem2gpower[m-1] * omn / fm
	}
	V0 := make([]float64, npwrs)
	V1 := make([]float64, npwrs)
	V1[0] = 1
	V0s, V1s := 0, 0
	expnt := 0
	j1, j2 := pomeranzBounds(0, n, ll, ceilf, roundf)
	for i := 1; i < 2*n+2; i++ {
		k1 := j1
		V0, V1 = V1, V0
		V0s, V1s = V1s, V0s
		clear(V1)
		j1, j2 = pomeranzBounds(i, n, ll, ceilf, roundf)
		var pwrs []float64
		switch {
		case i == 1 || i == 2*n+1:
			pwrs = gpower
		case i%2 == 1:
			pwrs = twogpower
		default:
			pwrs = onem2gpower
		}
		ln2 := j2 - k1 + 1
		if ln2 > 0 {
			a := V0[k1-V0s : k1-V0s+ln2]
			start := j1 - k1
			cl := j2 - j1 + 1
			vmax := 0.0
			for q := 0; q < cl; q++ {
				c := start + q
				s := 0.0
				for r := max(0, c-ln2+1); r <= min(c, ln2-1); r++ {
					s += a[r] * pwrs[c-r]
				}
				V1[q] = s
				vmax = math.Max(vmax, s)
			}
			if 0 < vmax && vmax < ks2m128 {
				for q := range V1 {
					V1[q] *= ks2p128
				}
				expnt -= ksExp128
			}
			V1s = V0s + j1 - k1
		}
	}
	ans := V1[n-V1s]
	for m := 1; m <= n; m++ {
		if math.Abs(ans) > ks2p128 {
			ans *= ks2m128
			expnt += ksExp128
		}
		ans *= float64(m)
	}
	if expnt != 0 {
		ans = math.Ldexp(ans, expnt)
	}
	return clip01(ans)
}

// kolmogPelzGood is the Pelz-Good approximation of Pr(D_n <= x): SciPy's
// _kolmogn_PelzGood.
func kolmogPelzGood(n int, x float64) float64 {
	fn := float64(n)
	z := math.Sqrt(fn) * x
	zsq := z * z
	zthree, zfour, zsix := z*z*z, zsq*zsq, zsq*zsq*zsq
	const pi2, pi4, pi6 = math.Pi * math.Pi, math.Pi * math.Pi * math.Pi * math.Pi, math.Pi * math.Pi * math.Pi * math.Pi * math.Pi * math.Pi
	qlog := -pi2 / 8 / zsq
	if qlog < -708 {
		return 0
	}
	q := math.Exp(qlog)
	k1a := -zsq
	k1b := pi2 / 4
	k2a := 6*zsix + 2*zfour
	k2b := (2*zfour - 5*zsq) * pi2 / 4
	k2c := pi4 * (1 - 2*zsq) / 16
	k3d := pi6 * (5 - 30*zsq) / 64
	k3c := pi4 * (-60*zsq + 212*zfour) / 16
	k3b := pi2 * (135*zfour - 96*zsix) / 4
	k3a := -30*zsix - 90*math.Pow(z, 8)
	var K [4]float64
	maxk := int(math.Ceil(16 * z / math.Pi))
	for k := maxk; k > 0; k-- {
		m := float64(2*k - 1)
		msq := m * m
		mfour, msix := msq*msq, msq*msq*msq
		qp := math.Pow(q, float64(8*k))
		c := [4]float64{1, k1a + k1b*msq, k2a + k2b*msq + k2c*mfour,
			k3a + k3b*msq + k3c*mfour + k3d*msix}
		for i := range K {
			K[i] = K[i]*qp + c[i]
		}
	}
	den := [4]float64{z, 6 * zfour, 72 * math.Pow(z, 7), 6480 * math.Pow(z, 10)}
	for i := range K {
		K[i] = K[i] * q * math.Sqrt(2*math.Pi) / den[i]
	}
	q = math.Exp(-pi2 / 2 / zsq)
	sqrt3z := math.Sqrt(3) * z
	var k2extra, k3extra float64
	for k := maxk; k > 0; k-- {
		ksq := float64(k * k)
		qp := math.Pow(q, ksq)
		kspi := math.Pi * float64(k)
		k2extra += ksq * qp
		k3extra += (sqrt3z + kspi) * (sqrt3z - kspi) * ksq * qp
	}
	K[2] += k2extra * pi2 * math.Sqrt(2*math.Pi) / (-36 * zthree)
	K[3] += k3extra * pi2 * math.Sqrt(2*math.Pi) / (216 * zsix)
	s := 0.0
	for i := range K {
		s += K[i] / math.Pow(fn, float64(i)/2)
	}
	return s
}
