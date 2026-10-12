package stats

import (
	"math"
	"sync/atomic"
)

// A memo of the constants that depend on the shape parameters only (1/Γ(a),
// 1/B(a,b), the t density's normalising constant, their logarithms): a
// density evaluated at 10^6 points of one distribution, or a likelihood
// summed over the data at each step of a fit, computed them at every point,
// and they cost more than the rest of the density. The memo returns the
// value the function computes, bit for bit (it is that value, stored), so
// the methods remain pure functions of their arguments.
//
// It is a small direct-mapped table of entries, each guarded by a sequence
// number (a seqlock made of atomics, so that it is safe for concurrent use
// and clean under the race detector): a reader that finds an entry being
// written, or written while it read, computes the value itself. Nothing is
// allocated.

const shapeSlots = 64

type shapeEntry struct {
	seq  atomic.Uint64 // even: stable (0: empty); odd: being written
	tag  atomic.Uint64 // which function the entry belongs to
	k1   atomic.Uint64
	k2   atomic.Uint64
	v    [4]atomic.Uint64
	_pad [3]uint64
}

var shapeTable [shapeSlots]shapeEntry

// Function tags (one per memoised function).
const (
	tagGammaInv = iota + 1
	tagGammaLog
	tagBetaInv
	tagBetaLog
	tagTConst
	tagTLogConst
)

// shapeMemo returns f(a, b), from the table when it holds it.
func shapeMemo(tag uint64, a, b float64, f func(a, b float64) [4]float64) [4]float64 {
	ka, kb := math.Float64bits(a), math.Float64bits(b)
	h := (ka*0x9e3779b97f4a7c15 ^ kb*0xc2b2ae3d27d4eb4f ^ tag*0x165667b19e3779f9) >> (64 - 6)
	e := &shapeTable[h]
	s := e.seq.Load()
	if s&1 == 0 && s != 0 && e.tag.Load() == tag && e.k1.Load() == ka && e.k2.Load() == kb {
		var r [4]float64
		for i := range r {
			r[i] = math.Float64frombits(e.v[i].Load())
		}
		if e.seq.Load() == s {
			return r
		}
	}
	r := f(a, b)
	if s&1 == 0 && e.seq.CompareAndSwap(s, s+1) {
		e.tag.Store(tag)
		e.k1.Store(ka)
		e.k2.Store(kb)
		for i := range r {
			e.v[i].Store(math.Float64bits(r[i]))
		}
		e.seq.Store(s + 2)
	}
	return r
}
