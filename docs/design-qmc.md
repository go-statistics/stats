# Design: quasi-Monte Carlo (`go-statistics/stats/qmc`)

Status: design only, no code. Target: the whole `scipy.stats.qmc` surface of
SciPy **1.18.1**, plus the MATLAB `sobolset` / `haltonset` / `lhsdesign` /
`net` / `scramble` delta. It covers rows 232–238 of [coverage.md](coverage.md) §10.
This package is the QMC foundation named in the go-scikit roadmap
(`COVERAGE.md`, "Shared foundations"). Its consumers there are
`stats.qmc`, QMC integration (`qmc_quad`), optimize `shgo`, and the
differential-evolution `sobol`/`halton` initialisation.

Guiding rule: **port the reference exactly, so the reference is the equality
oracle; then measure; then optimise**. Every engine below is assigned an
*exact path* (bit-for-bit equality with SciPy fixtures) or a *tolerance path*
(stated bound, stated reason). When an engine is on the tolerance path, the
section says which dependency would move it to the exact path.

---

## 1. Sources

### Opened (and cited below by file:line)

All SciPy files were read at tag `v1.18.1` (`git show v1.18.1:<path>` in a clean clone):

| source | what was taken from it |
|---|---|
| `scipy/stats/_qmc.py` (2951 lines) | public API, validation, every engine, optimizers, `qmc_quad` hooks |
| `scipy/stats/_sobol.pyx` | `MAXDIM=21201`, `MAXDEG=18`, `_initialize_v`, `draw`, `_fast_forward`, `_cscramble` (LMS), `_fill_p_cumulative`, `_categorize` |
| `scipy/stats/_qmc_cy.pyx` | the four discrepancies, `update_discrepancy`, threaded loops, Van der Corput (plain and scrambled) |
| `scipy/stats/_sobol_direction_numbers.npz` | `poly`: `<i8`, shape (21201,); `vinit`: `<i8`, shape (21201, 18), **Fortran order** (npy headers read) |
| `scipy/_lib/_util.py` | `_transition_to_rng` (l.142), `rng_integers` (l.667–728), `_rng_spawn` (l.887–893) |
| `scipy/integrate/_quadrature.py` | `_qmc_quad_iv` (l.1076–1156), `qmc_quad` (l.1165–1350) |
| `scipy/sparse/csgraph/_min_spanning_tree.pyx` | Kruskal on CSR with a stable `argsort` (l.108–113, 140–210) |
| `LICENSES_bundled.txt` | the Joe–Kuo direction numbers are **not** listed there (grep: no match) |
| Joe & Kuo web page, https://web.maths.unsw.edu.au/~fkuo/sobol/ | data files (`new-joe-kuo-6.21201` recommended, D(6); also D(5), D(7), `joe-kuo-old.1111`), and the statement that program and direction numbers are under a BSD-style licence |
| `…/~fkuo/sobol/licence` | BSD-style, 3 clauses, © 2008 Frances Y. Kuo and Stephen Joe. It also names UNSW and the University of Waikato in the non-endorsement clause |
| Joe & Kuo, *Notes on generating Sobol′ sequences* (Aug. 2008, `joe-kuo-notes.pdf`, 3 pp.) | recurrence (2) for `m_k`, `v_k = m_k/2^k`, the point formula (3), Gray-code recursion (5), and the encoding `a_j = (a_1 … a_{s−1})_2` |
| Owen, *A randomized Halton algorithm in R*, arXiv:1706.02808 (HTML v2) | Algorithm 1 (random digit permutations; stop when `1 − b^−k < 1`). Permutations are independent per base and per digit position, and zero digits are permuted too |
| Owen, *On dropping the first Sobol′ point*, arXiv:2008.08051 (abstract only) | skipping points breaks the net structure; error can grow ∝ √n |
| Bridson, *Fast Poisson Disk Sampling in Arbitrary Dimensions*, SIGGRAPH 2007 sketch (1 p.) | grid cell size r/√n, active list, k candidates in the annulus [r, 2r], O(N) |
| MathWorks docs `sobolset`, `haltonset`, `lhsdesign` (current web pages) | properties (`Skip`, `Leap`, `PointOrder`, `ScrambleMethod`), `net`, indexing, `MatousekAffineOwen`, `RR2`. `sobolset` max dimension is 1111 and uses the **Joe–Kuo 2003** polynomials and initial values. `lhsdesign` takes `Smooth`, `Criterion` (`maximin` default, `correlation`, `none`) and `Iterations` (default 5) |
| go-ndarray `random` at **v0.33.0** (`generator.go`, `integers.go`, `scalar.go`, `pcg64.go`, `seedseq.go`, `bitgen.go`) | what already exists: `DefaultRNG`, `Generator.Spawn`, `IntegersOf(dt, low, high, endpoint, size…)`, `IntegersUint64`, `Shuffle`, `ShuffleInt64`, `Permutation`, `Uniform`, `FillUniform`, `StandardNormal`, `IntN`, `PCG64State{…, HasUint32, Uinteger}` with `State/SetState`, `SeedSequence.Spawn/NChildrenSpawned` |
| go-numerics/special **v0.2.0** | `Ndtri` (`erfinv.go:165`) and `GammaP` (`incgamma.go:51`, a double-double implementation, **not** a Cephes port) |
| go-statistics/stats `main` @ `0913742` | layout, fixture convention (`testdata/gen/*.py` → `testdata/scipy/*.json.gz`), `Rvs(g *random.Generator, …) (*ndarray.Array, error)` style |
| go-scikit/roadmap `COVERAGE.md` | consumers; Delaunay / k-d tree belong to go-geometry (item 7); csgraph belongs to go-sparse |

### Not obtained (not opened; nothing below is taken from them)

- Joe & Kuo 2008, *Constructing Sobol sequences with better two-dimensional
  projections*, SIAM J. Sci. Comput. 30:2635–2654 (paywalled). The D(6)
  criterion is used as shipped, never re-derived.
- Joe & Kuo 2003 (ACM TOMS 29:49–57) and Bratley & Fox 1988 (Alg. 659): only
  as cited in the notes and in `_sobol.pyx`.
- Owen 1995 (*Randomly permuted (t,m,s)-nets and (t,s)-sequences*) and Owen
  1998 (*Scrambling Sobol and Niederreiter–Xing points*, J. Complexity 14).
  SciPy does **not** implement nested uniform scrambling. It implements
  LMS + digital shift, so the port does not need these papers. A later
  `Scramble: Owen` option would need them first.
- Matoušek 1998 (*On the L2-discrepancy for anchored boxes*, J. Complexity 14).
  LMS is ported from `_cscramble`, not from the paper.
- Hickernell 1998 (*A generalized discrepancy and quadrature error bound*,
  Math. Comp. 67:299–322): the AMS PDF returned HTTP 403. The CD/WD/L2-star
  formulas are taken from `_qmc_cy.pyx` only. An independent check of the
  formulas is a task-7 prerequisite (§9.2).
- Zhou, Fang & Ning 2013 (mixture discrepancy) and Jin, Chen & Sudjianto 2005
  (`_perturb_discrepancy`, eqs. 19–26, with the typo fixes noted in the code):
  ported from the code only.
- Kocis & Whiten 1997 (RR2, Leap) and Hong & Hickernell 2003 (Alg. 823):
  required before the MATLAB `RR2` task (T16).
- MATLAB `scramble` and `net` reference pages (not fetched), and the MATLAB
  source (closed).
- NumPy C sources for `Generator.integers` buffering, `shuffle`,
  `multiply.reduce` and `linalg.norm`. Statements about them below are
  marked **[verify]** and become fixture tests in the first task that relies
  on them.
- SciPy `shgo` and `differential_evolution` sources (consumer contracts
  must be read when those ports start).

---

## 2. Scope

### 2.1 SciPy 1.18.1 surface (all of it)

| SciPy | Go (`package qmc`) | notes |
|---|---|---|
| `QMCEngine` (`random`, `integers`, `reset`, `fast_forward`, `d`, `num_generated`, `rng_seed`) | `Engine` interface + embedded `base` | §3 |
| `Sobol(d, scramble=True, bits=None→30, rng, optimization)`, `random_base2`, `MAXDIM` | `NewSobol` | §4.1 |
| `Halton(d, scramble=True, optimization, rng)` with `workers` | `NewHalton` | §4.2 |
| `LatinHypercube(d, scramble=True, strength∈{1,2}, optimization, rng)` | `NewLatinHypercube` | §4.3 |
| `PoissonDisk(d, radius=0.05, hypersphere∈{volume,surface}, ncandidates=30, optimization, rng, l_bounds, u_bounds)`, `fill_space` | `NewPoissonDisk` | §4.4 |
| `MultinomialQMC(pvals, n_trials, engine=None→Sobol(1,bits=30), rng)` | `NewMultinomialQMC` | §4.5 |
| `MultivariateNormalQMC(mean, cov, cov_root, inv_transform=True, engine, rng)` | `NewMultivariateNormalQMC` | §4.6 |
| `discrepancy(sample, iterative, method∈{CD,WD,MD,L2-star}, workers)` | `Discrepancy` | §5.1 |
| `update_discrepancy(x_new, sample, initial_disc)` | `UpdateDiscrepancy` | §5.2 |
| `geometric_discrepancy(sample, method∈{mindist,mst}, metric='euclidean')` | `GeometricDiscrepancy` | §5.3 |
| `scale(sample, l_bounds, u_bounds, reverse)` | `Scale` | §5.4 |
| optimization `random-cd` (n_iters=10000, n_nochange=100), `lloyd` (tol=1e-5, maxiter=10, Qhull) | `OptRandomCD`, `OptLloyd` | §4.7 |
| private but needed by consumers: `van_der_corput`, `n_primes`, `primes_from_2_to`, `_init_quad` | `VanDerCorput`, `NPrimes`, `Engine.NewLike` | §4.2, §7 |

### 2.2 MATLAB delta (sub-package `qmc/matlab`)

`sobolset(d,'Skip','Leap','PointOrder')`, `haltonset(d,'Skip','Leap')`,
`scramble(p,'MatousekAffineOwen'|'RR2'|'clear')`, `net(p,n)`, indexing
`p(i,j)`, `reduceDimensions`, `lhsdesign(n,p,'Smooth','Criterion','Iterations')`,
`lhsnorm`. `qrandstream` is a thin stream over a point set. §8.

### 2.3 Not ported, and why

- Legacy `seed=`/`RandomState` path (`_transition_to_rng`, `rng_integers`
  `randint` branch). The `rng=` Generator path is the one SciPy will keep.
  The legacy path adds `RandomState.randint`/`shuffle` parity for no new
  capability. It gets a row in coverage.md, marked "not planned".
- Multi-worker *ordering* of SciPy's discrepancy sums. SciPy adds per-thread
  partials under a mutex in completion order (`_qmc_cy.pyx:326–337`), so the
  result is not deterministic. Go is deterministic for every `Workers`
  value and equals SciPy's `workers=1` (§5.1).

---

## 3. Core API

```go
package qmc // github.com/go-statistics/stats/qmc

// Engine is scipy.stats.qmc.QMCEngine.
type Engine interface {
    D() int
    // Random returns an (n, d) float64 array in [0, 1)^d (PoissonDisk: its bounds).
    Random(n int) (*ndarray.Array, error)
    // RandomInto writes n*d values row-major into dst (len >= n*d): the
    // allocation-free path for qmc_quad, shgo and DE initialisation.
    RandomInto(dst []float64, n int) error
    Integers(lower, upper []int64, n int, endpoint bool) (*ndarray.Array, error)
    Reset()
    FastForward(n int) error
    NumGenerated() uint64
    // NewLike is type(qrng)(seed=g, **qrng._init_quad): same parameters,
    // scramble forced on, new randomisation from g (qmc_quad, §7).
    NewLike(g *random.Generator) (Engine, error)
}

// Source says where an engine's randomness comes from (SciPy's rng=).
type Source interface{ isSource() }
func FromGenerator(g *random.Generator) Source // SciPy: Generator → _rng_spawn(g, 1)[0]
func FromSeed(seed ...uint64) Source           // SciPy: int → default_rng(seed), no spawn
// nil Source = default_rng(None), OS entropy.
```

Options are structs with Go zero values meaning SciPy defaults.
`NoScramble bool` (since SciPy's default is `scramble=True`), `Bits int`
(0 → 30), `Optimization Optimization` (`OptNone`, `OptRandomCD`,
`OptLloyd`), `Workers int` (0 → 1, −1 → `runtime.NumCPU()`, other ≤0 → error,
as `_validate_workers` l.2892–2919), `Warn func(Warning)` (nil = silent).

**Warnings.** SciPy's `warnings.warn` calls become typed `Warning` values
(`WarnSobolBalance` from `_qmc.py:1865`, `WarnDuplicatePoints` from l.448,
`WarnZeroWidthDomain` for `qmc_quad`), delivered to the callback. They never
change results. **Errors.** Every SciPy `ValueError` becomes a wrapped sentinel
(`ErrDimension`, `ErrBits`, `ErrBalance`, `ErrExhausted`, `ErrBounds`,
`ErrNotUnitCube`, `ErrStrength`, `ErrResourceLimit`, …), with SciPy's message
text, so tests can match both.

**Base state** (`_initialize`, l.938–966):
- `d ≥ 0` (d = 0 is legal).
- The engine's own `*random.Generator`: a spawned child when given a
  Generator, `DefaultRNG(seed)` when given a seed.
- `rngSeed`, a deep copy taken at construction (`copy.deepcopy`, l.950).
- `numGenerated uint64`.
- The optimizer, bound to **the Generator object current at construction**.

`Reset` (l.1094–1097) replaces the engine's generator with a fresh deep copy
of `rngSeed`, but the `random-cd` optimizer keeps the *old* object. That is
how `partial(…, rng=self.rng)` at l.958/2612 behaves. The port reproduces it,
and a fixture pins it (reset → random with `random-cd`). The quirk is to be
reported upstream (§11).

**Deep copy is the one missing go-ndarray piece.** `copy.deepcopy(Generator)`
copies the bit-generator state, the buffered `uint32` and the SeedSequence,
including `n_children_spawned`. The SeedSequence matters because LHS
strength 2 spawns from the engine's generator on every call (§4.3).
go-ndarray v0.33.0 has `PCG64.State/SetState` but no `Generator.Clone`; it is
added there first (task T0).

---

## 4. Engines — algorithms and exact-path decisions

Notation: every float expression is written in SciPy's evaluation order. Go
code **must not let the compiler fuse** `a*b+c`: the Go spec allows fusion,
and the compiler does it on arm64, ppc64le and s390x. Every product that
feeds an add is wrapped in an explicit `float64(…)` conversion, which the
spec defines as a rounding point. CI runs the fixture tests under qemu arm64
(the repo already installs `qemu-user`) to catch a missing conversion.

### 4.1 Sobol — **exact**

**Direction numbers.**
- `poly[d] = 2a + 2^s + 1`, with `poly[0] = 1`, so `bit_length(poly)−1 = s`.
- `vinit[d, :s] = m_1…m_s`; columns past `s` are zero.
- Row 0 is all ones (`_sobol.pyx:107–145`, docstring recipe).
- Source: `new-joe-kuo-6.21201`, the D(6) set.

Plan:
1. `internal/sobolgen` (a Go program with a test) parses the original Joe–Kuo
   text file.
2. It writes `qmc/sobol_dn.bin`: per dimension `s`, `a`, `m_1…m_s` as
   uvarints, about 0.7 MB, loaded with `go:embed`.
3. A test decodes it and compares every entry with SciPy's npz. A committed
   copy in `testdata/scipy/`, read by a test-only npz/npy reader (zip + `<i8`,
   Fortran order).
4. The BSD licence text goes to `qmc/LICENSE-joe-kuo` and the README.
5. Decoding is lazy (`sync.Once`), only up to the requested `d`.

**`_initialize_v`** (l.245–296):
- Bratley–Fox recurrence, built from `newv = v[d,j−m]`.
- For each k, `pow2 <<= 1`; when bit `m−1−k` of p is set,
  `newv ^= pow2 * v[d,j−k−1]`.
- Then column `bits−1−c` is multiplied by `2^c`.
- All in `uint32` for `bits ≤ 32` and in `uint64` for 33–64 (`_qmc.py:1783–1788`).
  Go keeps both widths, with generics over `uint32 | uint64`.

**Draw** (`draw`, l.316–335):
- Gray-code recursion (5) of the Joe–Kuo notes.
- `l = low_0_bit(num_gen)`, the 1-based position of the lowest zero bit
  (`bits.TrailingZeros64(^x)+1`).
- `quasi[j] ^= sv[j, l−1]`, then `sample = float64(quasi) * scale` with
  `scale = 2^−bits`.
- The conversion is round-to-nearest-even in C and in Go. The multiply by a
  power of two is exact. So every value is bit-exact for all bits ≤ 64.

**State machine** (`_random`, l.1846–1886), reproduced literally:
- If `num_generated == 0`:
  - Warn when n is not a power of two.
  - n = 1 returns the shift point (SciPy returns the cached `_first_point`
    array, an alias; Go returns a copy).
  - Otherwise, the first point is followed by `n−1` draws from `num_gen = 0`.
- Else: n draws from `num_gen = num_generated − 1`.
- `total_n > 2^bits` → `ErrExhausted`, with SciPy's message (including
  "Consider increasing `bits`" unless bits = 64).
- Go compares `n > maxn − numGenerated` with `maxn` as `uint64` for bits < 64.
  For bits = 64, `2^64` does not fit, so overflow is checked with
  `bits.Add64`.
- `random_base2(m)`: `n = 2^m`. A non-power-of-two `total_n` is `ErrBalance`.
  `m` must be ≤ `bits`, which also prevents shift overflow in Go.

**`fast_forward`** (l.1932–1957): the same XOR walk without output.
`num_generated == 0` walks n−1 steps. With n = 0 that is −1, which raises
`OverflowError` in Cython's unsigned conversion. Go returns `ErrDimension`
for n = 0 at `num_generated == 0` and n < 0 anywhere. **[verify]** SciPy's
exact exception with a fixture. `reset` restores `quasi = shift`.

**Scrambling, LMS + digital shift** (`_scramble`, l.1812–1828;
`_cscramble`, l.370–396):

Draw order (fixed):
1. The shift: `rng.integers(0, 2, size=(d, bits), dtype=uint32|uint64)`, then
   `shift[d] = Σ_k b[d,k]·2^k` (LSB first, `np.dot` in exact unsigned
   arithmetic).
2. The matrices: `rng.integers(0, 2, size=(d, bits, bits), dtype=…)`. `tril`
   keeps the lower triangle, the diagonal is forced to 1, and the upper
   triangle is drawn and discarded.

Then, per (d, j):
- For p = bits−1 … 0: `row = Σ_i ltm[d,p,i]·2^(bits−1−i)`.
- `t1 = parity(row & v)`.
- `t2 |= t1 << (bits−1−p)`.

Memory and speed:
- SciPy allocates the whole `d×bits×bits` tensor: 695 MB at d = 21201,
  bits = 64.
- Go streams it one dimension at a time (bits² draws), which is valid because
  the draws are consumed in C order. **[verify]** chunked `IntegersOf` calls
  consume the same stream as one call. Expected: NumPy's `uint32`/`uint64`
  bounded fill with range ≤ 2³²−1 uses `next_uint32`, whose buffer lives in
  the bit generator (`PCG64State.HasUint32`); go-ndarray models that. A
  fixture compares chunked and whole draws.
- Go builds each `row` once per (d, p) instead of once per (d, j, p).
  `parity` is `bits.OnesCount64(row&v) & 1`.
- Result: O(d·bits²) instead of SciPy's O(d·bits³). The result is
  bit-identical, because the same GF(2) products are computed.

**Validation**: `d ≥ 0`, `d ≤ 21201` (`MAXDIM`), `bits ≤ 64`.
- SciPy fails on `bits < max degree` through a Cython memoryview extent
  mismatch. Go returns `ErrBits` in exactly the cases where SciPy raises
  (fixture: `d=50, bits=2`).
- `bits = 0` with `d ≤ 1` is accepted, as SciPy does.

### 4.2 Halton and Van der Corput — **exact**

- Bases are the first d primes (`n_primes`, l.647–683): a literal table of
  168 primes, then `primes_from_2_to(big)` in steps of 1000 from 2000. The
  sieve is ported literally (l.639–644), including its `n < 6` behaviour.
- Permutations (l.686–730):
  - `count = ceil(54 / log2(base)) − 1` rows of `arange(base)`, each shuffled
    in turn with `Generator.shuffle` (`ShuffleInt64`).
  - Generated per base, in base order, at construction.
  - Shared by all points: Owen's Algorithm 1, with the `1−b^−k<1` stop turned
    into a fixed count. `log2` is exact for base 2, and `54/log2(p)` is never
    an integer for an odd prime, so `math.Log2` cannot move the `ceil`.
- Plain digit loop (`_qmc_cy.pyx:383–390`):
  - `b2r = 1/base`.
  - `while q > 0 { seq += float64(float64(q%base)*b2r); b2r /= base; q /= base }`.
  - Sequential accumulation into a zeroed value.
- Scrambled loop (l.437–445): exactly `count` digits, digit → `perm[j][digit]`,
  same accumulation.
- `start_index = num_generated`, as `int64`. SciPy uses C `long`, 32-bit on
  Windows, so SciPy-Windows wraps past 2³¹ points where Go does not;
  documented.
- `workers`: points are independent, so goroutine chunks give bit-identical
  output for any worker count.
- Output (`np.array(sample).T.reshape(n,d)`): row-major (n, d).

**Resource bound.** The permutation tables hold `Σ_bases count·base` int64
values, about 8 GB at d = 10⁴. SciPy would simply run out of memory. Go
computes the total before allocating and returns `ErrResourceLimit` above
`Options.MaxTableBytes` (default 1 GiB). The prime sieve is bounded the same
way.

### 4.3 LatinHypercube — **exact**

- Strength 1 (l.1546–1559):
  - Scrambled: draw `uniform(size=(n,d))` (= `next_double`, row-major).
  - Then d calls to `shuffle` on rows of `tile(arange(1,n+1),(d,1))`.
  - `(perm.T − u) / n`.
  - Unscrambled: u = 0.5, with no uniform draws.
- Strength 2, OA-LHS (l.1561–1605):
  - `p = int(sqrt(n))`; requires `p` prime, `n = p²`, `d ≤ p+1`, with
    SciPy's messages.
  - The OA columns come from `np.stack(np.meshgrid(*arrays), -1)`. With the
    default `indexing='xy'`, column 0 varies fastest; a fixture with p = 3
    pins it.
  - Then `(c0 + k·c1) mod p`.
  - One `permutation(p)` per column.
  - **A new `LatinHypercube(d=1, rng=self.rng)` per call.** That spawns a
    child of the engine's generator, so `n_children_spawned` advances. Then
    `random(p)` runs for each (column, level) in order.
  - `/= p`, and the first d columns are kept.

### 4.4 PoissonDisk — **exact for `surface`, tolerance for `volume`**

Bridson 2007, as adapted in l.2087–2307:
- `cell = radius/√d`, `grid = ceil((u−l)/cell)`.
- The **grid stores candidates as float32** (l.2153), with NaN for empty
  cells. The pool keeps the float64 values.
- The first point is `uniform(l, u)`: d draws of `l + (u−l)·next_double`.
- Loop:
  1. `idx = integers(0, len(pool))`.
  2. Remove from the pool **preserving order** (`del list[idx]`).
  3. Draw `ncandidates` candidates around the centre at radius
     `radius·factor` (factor 2 for volume, 1.001 for surface).
  4. Accept a candidate when it is in limits and has no grid neighbour within
     ±2 cells closer than `radius²`. The distance is
     `Σ (cand − float32cell)²` over axis d, NaN compares false.
- Candidate sampling:
  - `surface`: `vec = standard_normal((k,d))`, `vec /= norm(vec, axis=1)`,
    `p = c + vec·r`.
  - `volume`: `x = standard_normal((k,d))`, `ssq = Σx²`,
    `fr = r·gammainc(d/2, ssq/2)^(1/d)/√ssq`, `p = c + x·fr`.
- Reductions over d use NumPy's pairwise summation, which is sequential below
  8 elements. go-simd/floats `SumPairwise` is the matching kernel.
  **[verify]** that `linalg.norm(axis=1)` is `sqrt(add.reduce(x*x))`.
- **`volume` is on the tolerance path.** It needs SciPy's `gammainc` (xsf
  Cephes `igam`) and libm `pow(·, 1/d)`, and special's `GammaP` is a
  different, double-double algorithm. Contract:
  - Every accepted point within 4 ulp of SciPy.
  - The same accept/reject sequence and point count on all fixtures. The
    fixture generator rejects any case where a candidate's nearest-neighbour
    distance falls within 1e-12 relative of `radius`, so a ulp flip cannot
    change the sequence.
  - Exact path later if go-numerics/special adds a Cephes-parity `igam` and a
    libm-parity `pow` (filed, §11).
- Quirks reproduced:
  - `num_generated` is bumped twice per call: `num_drawn` in `_random` and
    `n` in `random`.
  - `fill_space()` is `random(inf)`. In Go it is a separate `FillSpace(ctx)`
    with a cancellable context, since it can run for a very long time.
  - A candidate exactly on an upper bound indexes `grid_size`, which is an
    IndexError in SciPy. Go returns `ErrBounds` and the case goes upstream.
- **Resource bound**: `Π grid_size · d · 4` bytes, checked with overflow-safe
  multiplication, before allocating. Above `MaxTableBytes` → `ErrResourceLimit`.
  SciPy would try to allocate 64¹⁰ cells for d = 10, r = 0.05.
- **Perf**: the pool is an order-statistic structure (Fenwick tree over an
  append-only slice) so select-k-th plus delete is O(log n) and keeps SciPy's
  order. SciPy's list delete is O(n).

### 4.5 MultinomialQMC — **exact**

- Checks `min(pvals) ≥ 0` and `isclose(Σp, 1)` (rtol 1e-5, atol 1e-8).
- Default engine: `Sobol(1, scramble, bits=30, rng)`.
- Per sample: `engine.random(n_trials)`, then the cumulative sum.
- **The cumulative sum is accumulated in C `float` (float32)** and stored
  into a float64 array (`_fill_p_cumulative`, l.401–410: `cdef float tot, t`).
- **Each draw is narrowed to float32** before the binary search
  (`_find_index(…, const float value)`, l.427–439, `value > p_cum[m]`).
- Go reproduces both narrowings: `float32` accumulation and `float32(draw)`
  compared as float64.
- Output is float64 counts (`np.empty` + int assignment).

### 4.6 MultivariateNormalQMC — **exact base; tolerance for transforms that use LAPACK/BLAS/libm**

- The default engine is `Sobol(engine_dim, scramble, bits=30)` with
  `engine_dim = d` (inverse transform) or `2⌈d/2⌉` (Box–Muller). The base
  points are exact.
- Inverse transform: `ndtri(0.5 + (1−1e−10)·(u−0.5))`. Exact only if
  special's `Ndtri` is bit-equal to xsf's. The fixture measures it, and the
  bound is set to what the measurement shows (≤ 2 ulp expected). If special's
  `Ndtri` is a Cephes port, the contract becomes equality.
- `cov` given:
  - `allclose(cov, cov.T)`, then Cholesky, with a fallback to `eigh`
    (clipping eigenvalues at 0 after `≥ −1e−8`).
  - The root is `L.T` or `(V·√λ).T`, then `samples @ root + mean`.
  - LAPACK/BLAS order is unspecified, so **tolerance**: rtol 1e−13 against
    the fixture, plus an exact check of `samples @ root` against
    go-ndarray's own `MatMul` of the exact base.
  - `cov_root` given: no factorisation.
  - Neither given: `base + mean`, exact.
- Box–Muller (`inv_transform=False`): `√(−2 log u_even)·(cos, sin)(2π u_odd)`,
  interleaved, truncated to d. libm `log/cos/sin` → 2 ulp tolerance.

### 4.7 Optimizers

- **`random-cd` — exact** (l.2619–2671):
  - n = 0 or d = 0 → empty; d = 1 or n = 1 → unchanged.
  - Otherwise CD (§5.1), then up to 10 000 iterations, stopping after 100
    without change.
  - Each iteration draws three scalars, `integers(0,d−1,endpoint)`,
    `integers(0,n−1,endpoint)` and `integers(0,n−1,endpoint)` (go-ndarray
    `IntN(d)`, `IntN(n)`, `IntN(n)`: same range, same draws).
  - `_perturb_discrepancy` (l.558–612) is ported expression by expression:
    `np.prod(axis=1)` is a sequential product **[verify]**; `sum(sum_[mask])`
    is Python's left-to-right sum starting from integer 0.
  - Swap when `disc < best`.
- **`lloyd` — tolerance, deferred** (l.2678–2889):
  - Needs `scipy.spatial.Voronoi`, which is Qhull with
    `"Qbb Qc Qz QJ"` (+ `" Qx"` for d ≥ 5). Its vertex order sets the
    centroid summation order.
  - The decay is `exp(−x/root)+0.9` with `root = −maxiter/log(0.1)`, and the
    stop is on |Δ min-L1| < tol.
  - Blocked on a Voronoi in go-geometry (roadmap item 7). Contract: same
    iteration count, points within 1e−12 abs. Task T17 (last), gated.

---

## 5. Measures

### 5.1 `discrepancy` — **exact** (sequential order; *not* pairwise)

The four kernels are sequential C loops (`_qmc_cy.pyx:61–240`). The
reference summation order is a single running `double` accumulator:
- CD: `disc1` over i; `disc2` over (i, j) in row-major order.
- WD and MD: the same pattern.
- L2-star: a per-row `tmp_sum`, added to `disc2` after each row.

go-simd `SumPairwise` is **not** the oracle here; using it would break
equality. Per-term products run over k = 0…d−1 sequentially.

Formulas, transcribed exactly. `**2` on a C double is `x*x`, and integer
`n**2` is an exact integer converted to double:
- CD: `(13/12)^d − 2/n·disc1 + 1/n²·disc2`.
- WD: `−(4/3)^d + 1/n²·disc`.
- MD: `(19/12)^d − 2/n·disc1 + 1/n²·disc2`.
- L2-star: `sqrt(3^−d − (1/n)·2^(1−d)·disc1 + 1/n²·disc2)`.
- `iterative` adds 1 to n *after* the sums.

**`pow` constants.** C `pow` (glibc, Apple libm) is not guaranteed correctly
rounded but in practice is within 0.5 ulp. Go's `math.Pow` uses repeated
multiplication for integer exponents and drifts with d, so it is **never
used**. A `powCR(x float64, k int64)` helper (big.Float, 128-bit, one final
rounding) computes `(13/12)^d`, `(4/3)^d`, `(19/12)^d` and `3^−d`. `2^(1−d)`
is exact through `math.Ldexp`. Residual risk: a d at which libm pow is
misrounded. The fixture sweeps d = 1…200 and every mismatch is listed in the
test as a libm defect, never absorbed by a tolerance.

`n²`: SciPy computes it in `Py_ssize_t`, which overflows past n ≈ 3.04·10⁹.
Go computes `float64(n)*float64(n)`, the same value below that n, and an O(n²)
loop never gets there.

**Exact parallelism.**
- Rows i are split across goroutines. Each fills a buffer of per-(i, j)
  products (CD/WD/MD) or a per-row `tmp_sum` (L2-star).
- A single goroutine then adds them in (i, j) order.
- Products are O(n²·d) and the ordered adds are O(n²), so the speedup
  approaches the worker count for d ≳ 4.
- The result is bit-identical to `workers=1` for any worker count.

`ctx` variant: `DiscrepancyContext(ctx, …)`, because the work is O(n²d).

### 5.2 `update_discrepancy` — **exact**

`c_update_discrepancy` (l.249–289), with `n = len(sample)+1`:
- `disc1 = (−2/n)·Π(1+½|x−½|−½|x−½|²)`.
- `disc2 = (2/n²)·Σ_i Π(1+½|x_j−½|+½|s_ij−½|−½|x_j−s_ij|)`.
- `disc3 = (1/n²)·Π(1+|x−½|)`.
- `initial + disc1 + disc2 + disc3`, evaluated left to right.

Validation as in l.502–522.

### 5.3 `geometric_discrepancy` — **exact** (euclidean and the metrics we own)

- `pdist`: Euclidean is `sqrt(Σ(u−v)²)`, summed sequentially over d.
- Duplicate points (`== 0` anywhere) → `WarnDuplicatePoints`.
- `mindist`: minimum of the non-zero distances.
- `mst`:
  - `squareform` → csgraph → Kruskal over a **stable argsort** of the CSR
    data (`_min_spanning_tree.pyx:108–113`). The full symmetric matrix is in
    row-major order, so ties resolve by (i, j).
  - Union–find as in l.150–210. Zero-distance entries are not edges.
  - The output CSR is read in row-major order, and NumPy's pairwise mean is
    taken over it (`np.mean` of a `np.matrix`; go-simd `SumPairwise` / k).
- `metric`: `euclidean`, `sqeuclidean`, `cityblock`, `chebyshev`,
  `minkowski(p)` are ported here, and `cityblock` is also needed by Lloyd.
  The other `pdist` metrics return `ErrUnsupported` until a shared `pdist`
  exists (go-geometry), and then delegate to it.
- The Kruskal port lives here, private. When go-sparse grows csgraph, it
  moves there (one implementation, lowest shared library).

### 5.4 `scale` and `integers` — **exact**

- Forward: `sample*(u−l) + l`. Two roundings, `(u−l)` computed first,
  explicit `float64()` against fusion.
- Reverse: `(sample−l)/(u−l)`.
- Bounds: `l < u` elementwise after broadcasting to d.
- Forward checks `max ≤ 1` and `min ≥ 0` (NaN passes SciPy's check; the same
  happens in Go, **documented**). Reverse checks within the bounds.
- `Integers`: `floor(scale(random(n)))` → int64. `u_bounds=None` swaps the
  bounds, and `endpoint` adds 1 to u (`_qmc.py:1059–1083`). Only Halton
  forwards `workers`.

---

## 6. Validation plan

1. **Fixture generator** `testdata/gen/qmc.py`, in the repo's existing
   style. Output: `testdata/scipy/qmc_*.json.gz`, with float64 stored as
   exact bit patterns.
   - Run against the SciPy 1.18.1 / NumPy wheels on **linux/amd64 and
     linux/arm64**. The generator diffs the two. If they differ (C
     contraction into FMA on arm64), the amd64 output is the oracle, the
     differing cases are listed in the fixture, and Go must equal amd64 on
     every architecture.
2. **Per engine, equality cases** (seed via `default_rng(seed)` *and* via a
   parent Generator, to exercise spawn):
   - Sobol: d ∈ {0, 1, 2, 3, 10, 100, 1111, 21201}; bits ∈ {1, 2, 18, 30,
     32, 33, 53, 54, 63, 64}; scrambled and plain. Sequences of
     `random(1)`, `random(7)`, `random_base2`, `fast_forward`, `reset`,
     interleaved. The exhaustion boundary at `2^bits` (bits ≤ 20).
   - Halton: d up to 1000, scrambled and plain, `workers` ∈ {1, 4} (equal),
     `start_index` continuity.
   - LHS: strength 1 and 2, scrambled and plain, `random-cd` (including after
     `reset`), and a call made twice to pin the spawn counter.
   - PoissonDisk: surface exact; volume on the tolerance contract; custom
     bounds.
   - Multinomial and MVN as in §4.5–4.6.
3. **Measures**: discrepancy for the 4 methods × `iterative` × d ∈ 1…200 ×
   n ∈ {2, 7, 64, 513}; update_discrepancy; geometric (both methods, ties,
   duplicates); scale (both directions).
4. **Doc examples** in `_qmc.py` become Go examples, with byte-equal printed
   values: e.g. `discrepancy(space) = 0.008142039609053464`, the iterative
   pair `0.04769081147119336` / `0.008142039609053513`, and the
   Sobol `random_base2(m=3)` table.
5. **Independent properties**, not from SciPy:
   - Sobol 2^m points are a (t, m, s)-net: each elementary interval of volume
     2^−m holds exactly one point in 1-D projections and balanced counts in
     2-D.
   - Joe–Kuo notes example: `m = 1, 3, 7` with `x³+x+1` gives
     `0, .5, .75, .25, .875, .375`.
   - LHS: exactly one point per stratum per axis.
   - OA-LHS: strength-2 balance.
   - Poisson: min distance ≥ radius.
   - Halton base-2 column equals Van der Corput.
6. **Negative control**: the repo's `negative_control.sh` pattern. Flipping
   one direction number, or one bit of the shift, must make the fixture test
   fail.
7. go-ndarray assumptions marked **[verify]** each get a fixture in the task
   that relies on them: chunked `integers` streams, nested spawn keys,
   `shuffle` of int64 rows, `multiply.reduce` order, `linalg.norm(axis=1)`.

---

## 7. Consumer contracts

- **`qmc_quad`** (`_quadrature.py:1276–1350`):
  - Default `qrng = Halton(d)`.
  - `rngs = _rng_spawn(qrng.rng, n_estimates)`, spawned from the *engine's*
    (already spawned) generator.
  - For each estimate: `qrng.random(n_points)`, `x = (sample·(b−a)+a).T`,
    then `qrng = type(qrng)(seed=rngs[i], **_init_quad)`.
  - `_init_quad` forces `scramble=True` and keeps d, bits/strength,
    optimization and the PoissonDisk parameters. A Generator passed as
    `seed` is spawned again by `_initialize`; **[verify]** with a fixture
    that `_transition_to_rng` forwards it unchanged.
  - Go: `Engine.NewLike(g)` plus `QMCEngine.Generator()` (for the spawn).
    The integrate port owns `qmc_quad` itself.
- **shgo** and **DE `init='sobol'|'halton'`**: sources not opened. They need
  `NewSobol`/`NewHalton` with a `Source`, `RandomInto`, `RandomBase2`, and
  `Scale`. Their exact call sequences get checked when those ports start.
  This design only guarantees that each SciPy call has a bit-exact Go
  counterpart.

---

## 8. MATLAB delta (`qmc/matlab`) — structural, not bit-exact against MATLAB

MATLAB's RNG streams and source are not available, so the oracle is the
documented behaviour plus properties.

- **`Sobolset`**: `Dimensions ≤ 1111`, `Skip`, `Leap`, `PointOrder`
  (`standard` = formula (3), `graycode` = recursion (5)), `Net(n)`,
  `At(i, j)` (MATLAB documents indices up to 2⁵³), `ReduceDimensions`.
  - MATLAB documents the **Joe–Kuo 2003** values for its 1111 dimensions,
    which is `joe-kuo-old.1111`, not D(6).
  - The second table is embedded the same way (same licence).
  - Unscrambled points in both orders are deterministic and tested against
    formula (3) computed independently in the test.
- **`scramble(p, 'MatousekAffineOwen')`**: LMS + digital shift, reusing §4.1
  with a go-ndarray Generator. Distributional and net-property tests only.
- **`Haltonset`**: `Skip`, `Leap`, plain points exact (radical inverse).
  `RR2` (Kocis–Whiten reverse-radix permutations) is deterministic and so
  exact in principle, but **gated on obtaining Kocis & Whiten 1997** (§1).
- **`Lhsdesign`**: `Smooth` on/off, `Criterion` maximin (default)/
  correlation/none, `Iterations` (default 5): take the best of `Iterations`
  candidate designs under the criterion. `Lhsnorm`: LHS mapped through the
  normal inverse CDF with mean and covariance. Statistical tests only.
- **`Qrandstream`**: a stream over a point set, keeping state between calls
  (coverage row 237).

---

## 9. Performance plan

### 9.1 Method

Per the house rules:
- Reference: SciPy 1.18.1 at `workers=1`, single core.
- Harness: the repo's existing `perf_scipy_test.go`.
- Interleaved rounds. Runs are deferred when 1-min load > 8 (logged, not run).
- Allocation counts are reported exactly.
- Port first, measure, then optimise only where the table shows a gap,
  algorithm before SIMD.
- Any case slower than SciPy is an open defect.

### 9.2 Cases and expected levers

| case | SciPy cost | Go plan |
|---|---|---|
| Sobol construct, scrambled, d ∈ {10, 1000, 21201}, bits ∈ {30, 64} | O(d·bits³) LMS; 695 MB tensor at max | O(d·bits²) via row-ints + popcount; streamed draws (exact) |
| Sobol draw, n = 2^10…2^20, d ∈ {2, 10, 100, 1000} | 1 XOR + convert + mul per value | lanes over j: XOR-accumulate + u32/u64→f64 × 2^−bits. Exact: integer ops, exact scaling. Kernel goes in go-simd (u64→f64 needs AVX-512 or the bits ≤ 53 split) |
| Halton n = 10⁵, d ∈ {10, 100}, scrambled | per-digit `%` and `//` by a runtime base | invariant division by multiply-shift (exact integers); same float order; exact `workers` |
| LHS n = 10⁴, d = 50; OA-LHS p = 31 | shuffles dominate | `ShuffleInt64` in place, no `tile` copies |
| random-cd n = 100, d = 10 | numpy temporaries per iteration | allocation-free perturb with scratch rows |
| discrepancy ×4, n ∈ {10³, 10⁴}, d ∈ {2, 10, 50} | O(n²d) scalar C | lanes over j (each lane runs k sequentially, so products are bit-identical, no FMA), ordered scalar accumulation; exact parallel (§5.1). Kernel in go-simd |
| update_discrepancy n = 10⁵ | O(nd) | same lane scheme |
| geometric mst n = 2000 | pdist + scipy.sparse Kruskal | dense Kruskal with stable sort on (dist, i, j) |
| PoissonDisk r = 0.02, d = 2; `fill_space` | O(n) list deletes | Fenwick order-statistic pool |
| MVN n = 2^14, d = 10 | Sobol + ndtri + matmul | exact base, then go-ndarray MatMul |

SIMD rule: bit-parity kernels must not use FMA and must keep the reference
order. Only the tolerance-path engines (Lloyd, MVN with `cov`, Box–Muller,
Poisson volume) may use fused or reordered kernels, and their tests say so.

---

## 10. Security and robustness

- **Untrusted sizes.** d, n, bits, m, `n_trials`, `ncandidates`, bounds and
  radius all come from callers.
  - Every allocation size is computed in `int64` with overflow checks
    (`n·d`, `d·bits`, `Π grid`, `Σ count·base`, prime sieve).
  - On 32-bit (`GOARCH=386/arm`) `int` is 32 bits, so sizes above `MaxInt`
    → `ErrResourceLimit`, never a wrapped length.
  - CI already runs cross-architecture tests; the QMC tests join them,
    including a 386 run.
- **Bounded tables** (`MaxTableBytes`, default 1 GiB, settable): Halton
  permutations, the Poisson grid, the Sobol direction-number decode (fixed
  ≤ 21201), discrepancy buffers (chunked so memory is O(workers·n)).
- **Unbounded time**: `FillSpace`, `Discrepancy`, `GeometricDiscrepancy`
  and `random-cd` have `ctx` variants. Bomb tests assert a **refusal**:
  - Halton d = 10⁵.
  - Poisson d = 10, r = 0.01.
  - Sobol d = 21202, bits = 65.
  - `random_base2(m=70)`.
  - `Integers` with u−l overflowing int64.

  They run under `GOMEMLIMIT` and `ulimit -v`.
- **NaN/Inf inputs**: SciPy's `max() > 1` check lets NaN through; Go matches
  it and documents the behaviour. `pvals` containing NaN fail `isclose`, as in
  SciPy. Bounds containing NaN fail `l < u`.
- **Fuzz targets** (corpora committed under `qmc/testdata/fuzz/`):
  - `FuzzSobolOps` (d, bits, op sequence: random / base2 / ff / reset)
  - `FuzzHaltonOps`
  - `FuzzLHS` (strength, n, d)
  - `FuzzDiscrepancy` (bytes → bounded n×d sample, method, iterative,
    workers: parallel equals sequential)
  - `FuzzUpdateDiscrepancy`
  - `FuzzScale`
  - `FuzzPoissonParams`
  - `FuzzMultinomial` (pvals, n_trials)
  - `FuzzDirectionNumberDecode` (truncated or corrupt embedded table must
    error, never panic)

  The invariants: no panic, output in range, and `Reset` followed by
  replay is identical.
- **Embedded data integrity**: a SHA-256 of `sobol_dn.bin` is checked in a
  test, and the decoder bounds every uvarint (`s ≤ 18`, `m_k < 2^k`, `m_k`
  odd, as the notes require).
- `govulncheck` already runs in CI; no new dependencies besides go-ndarray
  and special.
- Findings filed upstream: the SciPy `reset` / `random-cd` stale RNG, the
  Poisson upper-bound IndexError, and the 32-bit `long` Halton index on
  Windows.

---

## 11. File layout

```
qmc/
  doc.go                 package doc, SciPy/MATLAB mapping
  engine.go              Engine, Source, base state, Reset/FastForward default, Integers, NewLike
  options.go errors.go warnings.go validate.go (workers, bounds)
  scale.go
  sobol.go               engine + state machine
  sobol_dirnum.go        go:embed sobol_dn.bin (+ sobol_dn_old.bin for MATLAB), lazy decode
  sobol_scramble.go      LMS + shift, streamed
  sobol_kernel*.go       draw/ff kernels (generic uint32|uint64; SIMD via go-simd)
  vdc.go primes.go halton.go
  lhs.go lhs_oa.go
  opt_randomcd.go perturb.go opt_lloyd.go
  poisson.go ordstat.go
  multinomial.go mvn.go
  discrepancy.go discrepancy_kernels.go update.go powcr.go
  geometric.go pdist.go mst.go
  LICENSE-joe-kuo
  testdata/fuzz/...
  matlab/ sobolset.go haltonset.go scramble.go lhsdesign.go qrandstream.go
internal/sobolgen/       Joe–Kuo text → sobol_dn.bin (+ test against SciPy npz)
testdata/gen/qmc.py      fixture generator (amd64 + arm64 diff)
testdata/scipy/qmc_*.json.gz, sobol_direction_numbers.npz
```

---

## 12. Tasks, in order (one PR each; each updates README, coverage.md, landing and docs site where user-visible)

- **T0.** go-ndarray: `Generator.Clone` (deepcopy: state, buffered uint32,
  SeedSequence with `n_children_spawned`). Add NumPy fixtures for deepcopy,
  nested spawn and chunked `integers(0,2,dtype=uint32|uint64)`.
- **T1.** qmc skeleton: `Engine`, `Source`, options, errors, warnings,
  `Scale`, bounds/workers validation, and the fixture generator with the
  amd64/arm64 diff.
- **T2.** Sobol direction numbers: `internal/sobolgen`, embedded table,
  Joe–Kuo licence, test against SciPy's npz.
- **T3.** Sobol unscrambled: init, draw, `fast_forward`, `random_base2`,
  `reset`, the balance warning, bits 1–64, exhaustion; fixtures.
- **T4.** Sobol scrambled: LMS + digital shift, streamed, O(d·bits²);
  fixtures; `Integers`.
- **T5.** Van der Corput, primes and Halton (scrambled, exact `workers`,
  resource bound).
- **T6.** LatinHypercube, strengths 1 and 2 (spawned inner engine).
- **T7.** Discrepancies CD/WD/MD/L2-star, `powCR`, exact parallel
  reduction, `update_discrepancy`. Prerequisite: an independent source for
  the CD/WD/MD formulas (Hickernell 1998 or Fang et al.) and an mpmath
  cross-check.
- **T8.** `random-cd` optimizer with `_perturb_discrepancy`, including the
  reset quirk.
- **T9.** `NewLike` / consumer helpers and an end-to-end `qmc_quad`-sequence
  fixture (spawn chain).
- **T10.** PoissonDisk (surface exact, volume on the tolerance contract,
  order-statistic pool, grid bound, `FillSpace(ctx)`).
- **T11.** MultinomialQMC (float32 cumulative path).
- **T12.** MultivariateNormalQMC (exact base; `Ndtri` parity measured;
  Cholesky/eigh; Box–Muller).
- **T13.** `geometric_discrepancy` (pdist subset, Kruskal port, pairwise
  mean).
- **T14.** Performance: go-simd kernels first (XOR-scale u32/u64→f64,
  no-FMA lane products) in their own go-simd PR, then use them here, with
  the perf table against SciPy.
- **T15.** MATLAB `Sobolset` / `scramble` / `Net` / `Qrandstream` and the
  `joe-kuo-old.1111` table.
- **T16.** MATLAB `Haltonset` (`RR2` gated on Kocis & Whiten 1997),
  `Lhsdesign`, `Lhsnorm`.
- **T17.** `lloyd` optimization. Gated on a go-geometry Voronoi (roadmap
  item 7).
