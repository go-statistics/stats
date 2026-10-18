package stats

import (
	"math"
	"sort"
	"testing"
)

// FuzzDistributionMethods drives every registered distribution with
// fuzzed parameters and a fuzzed argument through every method: none may
// panic, and a probability must come back NaN or in [0, 1] (CDF and SF,
// for the distributions whose CDF stays in [0, 1]). The seed corpus is
// each fixture instance at a few arguments, huge ones included; `go test`
// runs it, `go test -fuzz=FuzzDistributionMethods` explores beyond.
func FuzzDistributionMethods(f *testing.F) {
	type dist struct {
		name string
		e    distEntry
		keys []string
	}
	var dists []dist
	for _, pf := range loadParity(f) {
		if len(pf.Instances) == 0 {
			continue
		}
		var keys []string
		for k := range pf.Instances[0].Params {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		dists = append(dists, dist{pf.Go, distTypes[pf.Go], keys})
	}
	sort.Slice(dists, func(i, j int) bool { return dists[i].name < dists[j].name })
	for i, d := range dists {
		for _, pf := range loadParity(f) {
			if pf.Go != d.name {
				continue
			}
			for _, in := range pf.Instances {
				p := in.Params.floats()
				var ps [3]float64
				for j, k := range d.keys {
					if j < 3 {
						ps[j] = p[k]
					}
				}
				for _, x := range []float64{0.5, -1e300, 1e300, 0x1p60} {
					f.Add(uint8(i), ps[0], ps[1], ps[2], x)
				}
			}
		}
	}
	f.Fuzz(func(t *testing.T, which uint8, p0, p1, p2, x float64) {
		d := dists[int(which)%len(dists)]
		ps := [3]float64{p0, p1, p2}
		p := map[string]float64{}
		for j, k := range d.keys {
			if j < 3 {
				p[k] = ps[j]
			}
		}
		v, err := d.e.construct(p)
		if err != nil {
			return
		}
		for _, c := range distCalls(v) {
			c.f()
		}
		// the probabilities, where they are probabilities (VonMises' CDF
		// counts the turns: CDF(x + 2π) = CDF(x) + 1)
		if d.name == "VonMises" {
			return
		}
		fns := distFns(v)
		for _, k := range []string{"cdf", "sf"} {
			if r := fns[k](x); !(math.IsNaN(r) || r >= 0 && r <= 1) {
				t.Errorf("%s %v %s(%v) = %v", d.name, p, k, x, r)
			}
		}
	})
}
