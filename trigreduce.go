package stats

import (
	"math"
	"math/big"
	"sync"

	"github.com/go-statistics/stats/internal/dd"
)

// Reduction modulo 2π of arguments too large for the double-double one.
//
// Below hugeReduce, z - 2πn with 2π in double-double (106 bits) leaves an
// error of about |z|·2^-106, far below the 2^-70 the callers need. Above,
// the product 2πn has more significant bits than the double-double
// carries and the remainder is noise, so large that it fell outside the
// sine table (a panic at |z| = 1e20 and beyond). There the reduction is
// done with 2π to reduceBits bits: a float64 is at most 2^1024, so the
// remainder keeps about reduceBits - 1030 > 200 correct bits, and no
// float64 lies closer than about 2^-62 to a multiple of 2π (Kahan and
// McDonald's bound, the reason Payne-Hanek reduction works in float64),
// so its relative accuracy is far beyond double-double's.

// hugeReduce is the |z| from which vmReduce reduces in big.Float.
const hugeReduce = 0x1p20

// reduceBits is the precision of 2π for the huge reduction.
const reduceBits = 1280

// twoPiBig is 2π to reduceBits bits, from Machin's formula
// π = 16 atan(1/5) - 4 atan(1/239), computed once when first needed.
var twoPiBig = sync.OnceValue(func() *big.Float {
	prec := uint(reduceBits + 64)
	atanInv := func(m int64) *big.Float {
		// atan(1/m) = Σ (-1)^k / ((2k+1) m^(2k+1))
		mm := new(big.Float).SetPrec(prec).SetInt64(m * m)
		pow := new(big.Float).SetPrec(prec).Quo(big.NewFloat(1).SetPrec(prec), new(big.Float).SetPrec(prec).SetInt64(m))
		sum := new(big.Float).SetPrec(prec).Set(pow)
		eps := new(big.Float).SetPrec(prec).SetMantExp(big.NewFloat(1), -int(prec))
		term := new(big.Float).SetPrec(prec)
		for k := int64(1); ; k++ {
			pow.Quo(pow, mm)
			term.Quo(pow, new(big.Float).SetPrec(prec).SetInt64(2*k+1))
			if term.Cmp(eps) < 0 {
				break
			}
			if k%2 == 1 {
				sum.Sub(sum, term)
			} else {
				sum.Add(sum, term)
			}
		}
		return sum
	}
	a := new(big.Float).SetPrec(prec).Mul(atanInv(5), big.NewFloat(32).SetPrec(prec))
	b := new(big.Float).SetPrec(prec).Mul(atanInv(239), big.NewFloat(8).SetPrec(prec))
	return a.Sub(a, b).SetPrec(reduceBits)
})

// reduceTwoPiHuge returns r = z - 2πn in [-π, π] as a double-double and
// n (rounded to float64, exact below 2^53 turns), for a finite z.
func reduceTwoPiHuge(z float64) (dd.DD, float64) {
	prec := uint(reduceBits + 64)
	tp := twoPiBig()
	zb := new(big.Float).SetPrec(prec).SetFloat64(z)
	q := new(big.Float).SetPrec(prec).Quo(zb, tp)
	// n = round(q): no float64 is a half-integer multiple of 2π, so the
	// direction of a tie does not arise.
	q.Add(q, big.NewFloat(0.5).SetPrec(prec))
	n, _ := q.Int(nil) // truncates toward zero
	if q.Sign() < 0 && !q.IsInt() {
		n.Sub(n, big.NewInt(1)) // floor
	}
	nb := new(big.Float).SetPrec(prec).SetInt(n)
	r := new(big.Float).SetPrec(prec).Mul(nb, tp)
	r.Sub(zb, r)
	hi, _ := r.Float64()
	lo, _ := new(big.Float).SetPrec(prec).Sub(r, new(big.Float).SetFloat64(hi)).Float64()
	nf, _ := nb.Float64()
	return dd.DD{Hi: hi, Lo: lo}, nf
}

// reduceTwoPiHugeOK reports whether z is in the range reduceTwoPiHuge
// serves.
func reduceTwoPiHugeOK(z float64) bool { return math.Abs(z) >= hugeReduce && !math.IsInf(z, 0) }
