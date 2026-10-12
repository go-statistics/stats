package stats

import (
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-ndarray/ndarray/random"
	"github.com/go-statistics/stats/internal/dd"
)

// hypergeometricDraws reproduces numpy's Generator.hypergeometric(good,
// bad, sample, size) (numpy/random/src/distributions/
// random_hypergeometric.c): for 10 <= sample <= good+bad-10, Stadlober's
// ratio-of-uniforms method HRUA with numpy's log-factorial; otherwise the
// direct simulation of the draws with numpy's random_interval. It draws
// from g's bit generator exactly as numpy does from its own.
func hypergeometricDraws(g *random.Generator, good, bad, sample int64, size []int) (*ndarray.Array, error) {
	if good >= hypergeomMax || bad >= hypergeomMax {
		return nil, fmt.Errorf("%w: hypergeometric draws need good and bad below %d", ErrParam, int64(hypergeomMax))
	}
	a, err := ndarray.New(size...)
	if err != nil {
		return nil, err
	}
	a = a.AsType(ndarray.Int64)
	v, _ := ndarray.Data[int64](a)
	bg := g.BitGenerator()
	for i := range v {
		if sample >= 10 && sample <= good+bad-10 {
			v[i] = hypergeometricHRUA(bg, good, bad, sample)
		} else {
			v[i] = hypergeometricSample(bg, good, bad, sample)
		}
	}
	return a, nil
}

// hypergeomMax is numpy's limit on good and bad (HYPERGEOM_MAX).
const hypergeomMax = 1e9

// hypergeometricSample simulates the draws one by one.
func hypergeometricSample(bg random.BitGenerator, good, bad, sample int64) int64 {
	total := good + bad
	computed := sample
	if sample > total/2 {
		computed = total - sample
	}
	remTotal, remGood := total, good
	for computed > 0 && remGood > 0 && remTotal > remGood {
		remTotal--
		if int64(randomInterval(bg, uint64(remTotal))) < remGood {
			remGood--
		}
		computed--
	}
	if remTotal == remGood {
		remGood -= computed
	}
	if sample > total/2 {
		return remGood
	}
	return good - remGood
}

// randomInterval is numpy's random_interval: a uniform integer in [0, max]
// by masked rejection of 32-bit draws. (numpy switches to 64-bit draws
// above 2^32 - 1, which the population limit of 2·10^9 never reaches, and
// returns 0 for max = 0, which hypergeometricSample never asks.)
func randomInterval(bg random.BitGenerator, max uint64) uint64 {
	mask := max
	for s := 1; s <= 32; s *= 2 {
		mask |= mask >> s
	}
	for {
		if v := uint64(bg.Uint32()) & mask; v <= max {
			return v
		}
	}
}

// The constants of HRUA: D1 = 2 sqrt(2/e), D2 = 3 - 2 sqrt(3/e).
const (
	hruaD1 = 1.7155277699214135
	hruaD2 = 0.8989161620588988
)

// hypergeometricHRUA is numpy's hypergeometric_hrua, operation for
// operation (the explicit conversions keep Go from fusing products into
// sums, which numpy's x86-64 build does not do either).
func hypergeometricHRUA(bg random.BitGenerator, good, bad, sample int64) int64 {
	popsize := good + bad
	computed := min(sample, popsize-sample)
	minGB, maxGB := min(good, bad), max(good, bad)
	p := float64(minGB) / float64(popsize)
	q := float64(maxGB) / float64(popsize)
	mu := float64(float64(computed) * p)
	a := mu + 0.5
	v := float64(float64(float64(float64(popsize-computed)*float64(computed))*p)*q) / float64(popsize-1)
	c := math.Sqrt(v + 0.5)
	h := float64(hruaD1*c) + hruaD2
	m := int64(math.Floor(float64(float64(computed+1)*float64(minGB+1)) / float64(popsize+2)))
	gm := logFactorial(m) + logFactorial(minGB-m) + logFactorial(computed-m) + logFactorial(maxGB-computed+m)
	b := math.Min(float64(min(computed, minGB)+1), math.Floor(a+float64(16*c)))
	var k int64
	for {
		u := bg.Float64()
		w := bg.Float64()
		x := a + float64(h*(w-0.5))/u
		if x < 0 || x >= b {
			continue
		}
		k = int64(math.Floor(x))
		gp := logFactorial(k) + logFactorial(minGB-k) + logFactorial(computed-k) + logFactorial(maxGB-computed+k)
		t := gm - gp
		if float64(u*(4-u))-3 <= t {
			break
		}
		if float64(u*(u-t)) >= 1 {
			continue
		}
		if float64(2*libmLog(u)) <= t {
			break
		}
	}
	if good > bad {
		k = computed - k
	}
	if computed < sample {
		k = good - k
	}
	return k
}

// libmLog is the correctly rounded logarithm (what the C libraries return
// almost always): the double-double logarithm rounded once.
func libmLog(x float64) float64 { return dd.Log(x).Float() }

// logFactorial is numpy's logfactorial: a table of log k! below 126, the
// Stirling series to the 1/k³ term above.
func logFactorial(k int64) float64 {
	if k < int64(len(logFactTable)) {
		return logFactTable[k]
	}
	const halfLn2Pi = 0.9189385332046728
	x := float64(k)
	return float64(float64((x+0.5)*libmLog(x))-x) + (halfLn2Pi + float64((1.0/x)*(1/12.0-1/float64(float64(360.0*x)*x))))
}

// logFactTable holds log k! for k = 0..125, correctly rounded (mpmath).
var logFactTable = [...]float64{
	0.0, 0.0, 0.6931471805599453, 1.791759469228055,
	3.1780538303479458, 4.787491742782046, 6.579251212010101, 8.525161361065415,
	10.60460290274525, 12.801827480081469, 15.104412573075516, 17.502307845873887,
	19.987214495661885, 22.552163853123425, 25.19122118273868, 27.89927138384089,
	30.671860106080672, 33.50507345013689, 36.39544520803305, 39.339884187199495,
	42.335616460753485, 45.38013889847691, 48.47118135183523, 51.60667556776438,
	54.78472939811232, 58.00360522298052, 61.261701761002, 64.55753862700634,
	67.88974313718154, 71.25703896716801, 74.65823634883016, 78.0922235533153,
	81.55795945611504, 85.05446701758152, 88.58082754219768, 92.1361756036871,
	95.7196945421432, 99.33061245478743, 102.96819861451381, 106.63176026064346,
	110.32063971475739, 114.0342117814617, 117.77188139974507, 121.53308151543864,
	125.3172711493569, 129.12393363912722, 132.95257503561632, 136.80272263732635,
	140.67392364823425, 144.5657439463449, 148.47776695177302, 152.40959258449735,
	156.3608363030788, 160.3311282166309, 164.32011226319517, 168.32744544842765,
	172.3527971391628, 176.39584840699735, 180.45629141754378, 184.53382886144948,
	188.6281734236716, 192.7390472878449, 196.86618167289, 201.00931639928152,
	205.1681994826412, 209.34258675253685, 213.53224149456327, 217.73693411395422,
	221.95644181913033, 226.1905483237276, 230.43904356577696, 234.70172344281826,
	238.97838956183432, 243.2688490029827, 247.57291409618688, 251.8904022097232,
	256.22113555000954, 260.5649409718632, 264.9216497985528, 269.2910976510198,
	273.6731242856937, 278.0675734403661, 282.4742926876304, 286.893133295427,
	291.3239500942703, 295.76660135076065, 300.22094864701415, 304.6868567656687,
	309.1641935801469, 313.65282994987905, 318.1526396202093, 322.66349912672615,
	327.1852877037752, 331.7178871969285, 336.26118197919845, 340.815058870799,
	345.37940706226686, 349.95411804077025, 354.5390855194408, 359.1342053695754,
	363.73937555556347, 368.35449607240474, 372.979468885689, 377.61419787391867,
	382.25858877306, 386.91254912321756, 391.5759882173296, 396.24881705179155,
	400.93094827891576, 405.6222961611449, 410.32277652693733, 415.03230672824964,
	419.7508055995447, 424.4781934182571, 429.21439186665157, 433.9593239950148,
	438.71291418612117, 443.47508812091894, 448.2457727453846, 453.0248962384961,
	457.81238798127816, 462.6081785268749, 467.4121995716082, 472.2243839269806,
	477.04466549258564, 481.87297922988796,
}
