module github.com/go-statistics/stats/bench

go 1.27.3

require (
	github.com/go-ndarray/ndarray v0.41.0
	github.com/go-scikit/refbench v0.4.0
	github.com/go-statistics/stats v0.1.1
)

require (
	github.com/go-fleettools/fleettools v0.81.0 // indirect
	github.com/go-numerics/optimize v0.6.0 // indirect
	github.com/go-numerics/special v0.5.3 // indirect
	github.com/go-simd/floats v0.17.0 // indirect
	github.com/go-sparse/sparse v0.3.0 // indirect
	golang.org/x/sys v0.49.0 // indirect
)

replace github.com/go-statistics/stats => ../
