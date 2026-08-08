# Plan: one interpolator, four classes

Two separable pieces. The first is a defect fix and is not in question: the
scattered interpolator should compute its own barycentric weights on a cached
`Delaunay` instead of calling `LinearNDInterpolator`, and should warm-start the
simplex search from the previous step. That is 30x to 1620x faster with
bit-identical results.

The second is the collapse the fix makes thinkable: delete
`rectilinear_interpolator`, delete the `UnstructuredAuxiliary*` pair, make
`FlowMap._interpolator` a single concrete method, and move `from_points` onto
`AuxiliarySeedGrid`. Six public classes become four and one abstract member goes.
**The collapse is no longer a recommendation.** It rests on the scattered path
being ~2x bilinear at any size, which holds at a fixed neighbour count and fails
at basin scale; see "Basin scale changes which term grows" below.

## How this arose

PR #47 (issue #20) split the auxiliary stencil into two class pairs because
`FlowMap.image` and `shrink_lines` read a field between grid points, and both
were written against two monotonic axes. The split was justified on a measured
100x-to-1000x cost of scattered interpolation. That measurement was of
`scipy.interpolate.LinearNDInterpolator`, not of the method: the same interpolant
computed directly is 30x to 1620x faster, and warm-starting the simplex search
makes it flat at ~2x bilinear at any grid size. With that, nothing needs two
classes.

## Measured state

Checked on 2026-08-08 at `ab07c3e`, darwin arm64, scipy 1.18.0, numpy 2.5.1.
Grids are at 1/25 degree spacing, so a bigger side means a bigger domain and the
neighbour counts stay constant, which is what growing a real study does. The
scripts were throwaway; the numbers reproduce.

### `LinearNDInterpolator.__call__` is the wrong entry point

Its cost is O(n) per query point in the number of grid points. The same
interpolant computed directly on a cached `Delaunay` (`find_simplex`, then the
barycentric weights out of `tri.transform`) agrees to `4.4e-16`:

| grid points | `LinearNDInterpolator` | `find_simplex` + `transform` | speedup |
|---|---|---|---|
| $10^2$ | 2.09 s | 0.069 s | 30x |
| $10^4$ | 156.5 s | 0.331 s | 473x |
| $10^6$ | 15 617 s | 9.64 s | 1620x |

Per traced line, taken as 300 seeds x 1000 evaluations with the query points
scattered over the whole domain.

### Warm starting removes the remaining growth

`find_simplex` walks from a cold start on every call, so each of 300 seeds
crosses $O(\sqrt{n})$ simplices. A 3 km step against a 4.4 km cell crosses at
most one or two, so starting from the previous step's simplex and hopping across
the face with the most negative barycentric coordinate settles in one or two
array operations.

| grid | bilinear | `find_simplex` per step | warm-start walk | vs bilinear |
|---|---|---|---|---|
| $100^2 = 10^4$ | 0.002 s | 0.005 s | 0.004 s | 1.7x |
| $300^2 = 9\times10^4$ | 0.006 s | 0.050 s | 0.013 s | 2.0x |
| $1000^2 = 10^6$ | 0.024 s | 0.891 s | 0.047 s | 2.0x |

300 seeds marching 3 km steps, results identical to the cold search, zero
fallbacks over 75 000 lookups at the largest size. `find_simplex` is 98% of the
cold cost (0.868 s of 0.891 s at $10^6$); the barycentric evaluation itself is
0.012 s, so the search was the whole problem.

### Scalings

- **bilinear on axes**: $O(1)$ per query point. Measured flat, 0.087 / 0.097 /
  0.101 s per trace from $10^2$ to $10^6$.
- **barycentric, cold**: $O(\sqrt{n})$ per query point plus a cache-miss term.
  Measured log-log exponent 0.40 then 0.74.
- **barycentric, warm**: $O(1)$ per step, flat.
- **KDTree build**: $O(n\log n)$, 0.19 s at $10^6$.
- **Delaunay build including `transform`**: $O(n\log n)$, 0.006 / 0.565 / 52.3 s
  at $10^2$ / $10^4$ / $10^6$, and 0.21 GiB at $10^6$. `tri.transform` is built
  lazily on first evaluation, so it has to be forced during setup or it lands in
  the first query and is mistaken for query cost.
- **windowing** (`ftle_ridge_seeds`): $O(nk)$ in time and memory, $k$ the
  neighbours inside the radius and constant at fixed resolution. 2.26 s and
  0.34 GiB of pair indices at $10^6$. At $10^8$ that is 36 GiB, which is the
  first hard wall in the package and belongs to both layouts.

### How big the amortisation is

One seed grid, $M$ flow maps. Geometry is built once; windowing and tracing are
paid per map. The per-map workload is a full `hyperbolic_lcs`: one neighbourhood
maximum over the whole grid, then 300 seeds x 1000 interpolator evaluations.

| | $100^2 = 10^4$ | $140^2 = 1.96\times10^4$ | $1000^2 = 10^6$ |
|---|---|---|---|
| Delaunay + transform, once | 0.520 s | 0.984 s | 53.27 s |
| KDTree, once | 0.001 s | 0.003 s | 0.187 s |
| per map, windowing | 0.018 s | 0.035 s | 2.538 s |
| per map, tracing | 0.078 s | 0.088 s | 0.164 s |
| geometry is worth | 5.4 maps | 8.1 maps | 19.8 maps |

$140^2$ is the closest to the real examples (151x126 = 19 026 grid points).
There, a session of $M = 10$ takes 2.21 s cached against 11.09 s rebuilt (5.0x),
and $M = 50$ takes 7.11 s against 55.45 s (7.8x). At $10^6$, $M = 10$ is 80 s
against 562 s.

Three things follow. The **KDTree amortisation is nil**: 0.003 s to build against
0.035 s of query per map, so caching the tree saves ~2% of the windowing and is
not worth designing for. The Delaunay is the whole of it. And at example scale
the absolute saving is ~9 s over a ten-map session, against ten Parcels legs over
40 000 particles that run in ~32 s, so caching moves the diagnostic layer from a
third of the run to a fifth rather than removing it. It becomes structural only
at $10^6$.

The 300 seeds above are ~5x the 58 lines the real Cabo Verde case produces, so
realistic tracing is cheaper: the ratios go up (geometry worth ~40 maps at
$1.96\times10^4$) and the absolute savings go down.

### Basin scale changes which term grows

A basin-wide grid at 1/200 degree is about $10^8$ points, and that is on the
roadmap. Basin scaling is a *fixed domain at finer resolution*, not a bigger
domain at fixed resolution, so the window in cells grows and with it the
neighbour count $k$, as the square. Over a fixed 40x40 degree domain with a
15 km ridge-seed window:

| resolution | points | window in cells | $k$ | pair list | rolling | kd-tree |
|---|---|---|---|---|---|---|
| 1/25 | $10^6$ | 7 | 43 | 0.32 GiB | 0.14 s | 2.5 s |
| 1/50 | $4\times10^6$ | 13 | 169 | 5.05 GiB | 1.07 s | 48 s |
| 1/100 | $1.6\times10^7$ | 27 | 673 | ~86 GiB | | out of memory |

At 1/200 degree, $w \approx 54$ cells and $k \approx 2900$, so at $10^8$ points
the pair list is ~2 TB.

**The rolling window PR #47 removed is $O(nw)$; the kd-tree that replaced it is
$O(nw^2)$ in both time and memory**, because it materialises every pair. At
1/25 degree $w^2$ is 49 and nobody notices. At 1/200 degree it is 2900. Measured
window dependence of the rolling max at $4\times10^6$ points: 0.57 / 1.05 / 2.30
/ 6.12 / 18.0 s for windows of 7, 13, 27, 55, 109 cells, so linear in $w$.
`bottleneck` is not installed in the default environment; with it `move_max` is a
deque and the rolling max becomes $O(n)$, independent of the window.

Interpolation at $10^8$ diverges the same way. Bilinear is $O(1)$ and builds
nothing. The Delaunay is ~21 GiB and roughly an hour.

Spatial parallelism is needed at $10^8$ regardless, since $4\times10^8$ arm
particles is a cluster job before any diagnostic runs, but it changes memory
rather than the exponent: tiling divides the pair list and the wall clock and
leaves the $2.7\times10^{11}$ pair operations. It also treats the three
structures differently. A rolling window tiles with a $w/2$ halo and a radius
query with an $r$ halo, both cleanly. A Delaunay does not: each tile has to be
triangulated with a halo and its boundary simplices discarded, or tensor lines
seam at tile edges.

There is a way to keep the layout-free seeding without the $w^2$: bin the points
into cells of side $r$, take per-bin maxima, and use that a point can only be a
neighbourhood maximum if it is the maximum of its own 3x3 bin block. The filter
is $O(n)$ and leaves a candidate set $m \ll n$ for the exact radius test. It
works on scattered points too.

### `NeighborFlowMap` does not need the rectilinear interpolator

It needs a rectilinear *grid* for its gradient, since `_central_separation_m`
shifts along `i` and `j`. The interpolator is a separate matter. Swapping the
scattered one onto a `NeighborFlowMap` and tracing tensor lines: it runs, and
where both are defined the curves agree to 19.8 m median and 0.1 km maximum.

### The two interpolants terminate lines differently

- Exact agreement at grid nodes. Between nodes on a regular lattice barycentric
  is ~40% worse in RMS against a smooth analytic field, because every cell's four
  corners are co-circular and the diagonal is arbitrary (1582 "/" against
  1618 "\\" on a 40x40 lattice).
- One lost grid point NaNs 0.25% of cell centres under bilinear, 0.12% under
  barycentric.
- On a synthetic `NeighborFlowMap` field: 165 traced points against 328, with 163
  traced by one and not the other.
- On a real Cabo Verde auxiliary flow map: 78 identical seeds either way, curve
  separation 0.17 km median and 87 km maximum, 1340 of ~5700 points traced by one
  and not the other.

Results move. That is accepted: this is greenfield and the release notes carry it.

## A gradient from the Delaunay ring, and why not

Since the Delaunay is being built anyway, `grad F` could be fitted from a point's
Delaunay-adjacent neighbours by weighted least squares, with no auxiliary arms at
all. That would cost $n$ advected particles instead of $4n$, which at $10^8$ grid
points is the dominant term in the whole pipeline. Measured, it does not work as
a replacement.

**The truncation order drops from two to one.** Expanding the fit,

$$G - \nabla F = \tfrac12 \Big[\sum_j w_j H[d_j, d_j]\, d_j^\top\Big] M^{-1} + O(h^2),
\qquad M = \sum_j w_j d_j d_j^\top,$$

so the leading error is the **third moment** of the neighbour offsets over $M$,
which is $O(h)$ unless the stencil is centrally symmetric. The four arms are
exact antipodal pairs and the $(i \pm 1, j \pm 1)$ difference is symmetric on a
mesh uniform in metres, so both have a vanishing third moment. A Delaunay ring
does not. Measured log-log slopes on a sinusoidal map over a fixed domain refined
from 8 km to 0.5 km:

| scheme | 8 km | 1 km | 0.5 km | slope |
|---|---|---|---|---|
| arms, $s$ = 1 km | 1.94e-5 | 1.91e-5 | 1.91e-5 | 0.00 |
| arms, $s = h/2$ | 3.10e-4 | 4.78e-6 | 1.19e-6 | 2.00 |
| neighbour $(i\pm1, j\pm1)$ | 1.24e-3 | 1.91e-5 | 4.77e-6 | 2.00 |
| ring, linear fit | 3.58e-3 | 4.49e-4 | 2.25e-4 | **1.00** |
| ring, quadratic fit | 1.24e-3 | 1.91e-5 | 4.77e-6 | 2.00 |

A ring on a *rectilinear* lattice does not escape it: the four axis neighbours
pair up but Qhull splits each co-circular cell arbitrarily, and one unpaired
diagonal is enough. Measured third moment 0.051 rectilinear, 0.079 jittered,
0.134 Poisson, with the error tracking it.

**There is no crossover in any range anyone will run.** The ring linear fit
matches 1 km arms at a grid spacing of 43 m on a lattice and 19 m on a Poisson
cloud. At 1/25 degree it is 94x worse than 1 km arms, at 1/200 degree still 12x
worse. On the two-density cloud of `cabo_verde_unstructured`, which is the layout
the idea exists to serve, it is 800x worse in the coarse half, and its error
varies 3.5x across the domain purely because the spacing does.

**The knob stops being metre-valued.** A 1-ring is a topological stencil the
caller cannot set. A **radius-$r$ neighbourhood** puts it back, is second order in
$r$ (4.51e-4 / 1.10e-4 / 2.69e-5 at $r$ = 8 / 4 / 2 km), and makes $r$ worth arms
of $s \approx r/1.4$. That variant is defensible at basin resolution and nowhere
else: a 1 km radius finds a median of 0 neighbours at 4.4 km spacing, 2 at 1 km,
and 8 at 556 m, so it needs about 1/200 degree before it is possible at all.
Below the point spacing only the arms work, because they are extra particles.

Where the ring wins: lost particles. At a 5% loss rate 4.95% of grid points lose
their gradient against 18.55% for the arms, since a dropped neighbour removes a
row rather than invalidating the point.

Three further findings worth keeping whatever is decided.

- **`_separation_m` is not a general separation helper.** It frames each pair on
  that pair's own mid-latitude, which is exact for a leg that is pure east or pure
  north and wrong for an oblique one, so a ring fit built on it cannot reproduce
  even a linear map (3.6e-5 in `reference`, 4.6e-4 in `high_latitude`). Framing
  every leg of one ring on the centre point instead makes the same fit exact to
  6e-14. The per-pair frame is right for a stencil that differences pairs and
  insufficient for one that fits over a ring.
- **Inverse-distance-squared weighting is mandatory**, not a refinement. Worst-case
  condition number of $M$ on a random cloud is 18.1 weighted against 8396 uniform.
- **The FTLE tail is what moves.** At 1/200 degree the ring's median FTLE error
  beats 1 km arms (6.86e-6 against 2.13e-5 per day) while its 95th percentile is
  14x worse and its maximum 34x. Ridges live in the tail.

Recommendation: do not replace the arms with a 1-ring fit. Keep the radius-$r$
variant on the table as a basin-scale-only alternative with a stated floor on the
grid spacing, and note that it does not collapse a class either. It would be a
fourth pair, since it seeds one particle per grid point and so has its own
flow-map contract, which runs against the collapse this plan argues for.

## The change

1. `_spatial.scattered_interpolator` returns a closure over a cached `Delaunay`
   doing `find_simplex` plus the `tri.transform` barycentric weights, not a
   `LinearNDInterpolator`. Force `tri.transform` at construction so the build
   cost sits in the build.
2. Give the seam a warm-start hint: `f(points, hint) -> (values, hint)`, with
   `hint=None` doing the cold search. The hop uses `tri.neighbors` across the
   face opposite the most negative barycentric coordinate, `-1` meaning the point
   has left the convex hull.
3. Thread the hint through `_trace_half_line` and `_shrink_line_tangent`. NaN
   query points from terminated lines skip the lookup rather than fall back to a
   cold search. The RK2 midpoint is half a step away and warm-starts from the
   same hint.
4. Delete `rectilinear_interpolator`. `FlowMap._interpolator` stops being
   abstract and becomes one concrete method on the base; the three overrides go.
5. Delete `UnstructuredAuxiliarySeedGrid` and `UnstructuredAuxiliaryFlowMap`.
   Move `from_points` onto `AuxiliarySeedGrid`, keeping the reserved-dim guard
   and the arm-placement guard.
6. Cache the triangulation so it is built once per seed grid rather than once per
   flow map, which is the whole point of the amortisation (see question 1).
7. Chunk `neighborhood_maximum`'s `query_ball_point` so the pair list is
   $O(\text{block})$ rather than $O(nk)$.

Then propagate, as `AGENTS.md` requires: `__init__` exports and `__all__`,
`docs/reference.md` autosummary, `docs/api.md` (the three-pair lede, the data
model, the constructor section), `docs/architecture.md` (the sections "Two axes",
"Why the unstructured pair subclasses the auxiliary one" and "Why the
interpolation seam is abstract on `FlowMap`" all change or go, and "Where SciPy
sits" needs the new contents), `docs/notation.md`, `docs/numerics.md` (the
piecewise-linear sentence under "Stepping a tensor line", plus the scalings
above), `README.md` scope, `CHANGELOG.md`, and the tests. Coverage is gated at
100%.

Every notebook needs re-executing, **without** setting `MPLBACKEND`: `Agg` runs
the cells and commits a notebook with no figures, which is how #47 lost its plots
once already.

## Questions

1. **Where does the cached `Delaunay` live?** It is geometry, so it belongs to
   the seed grid, but `FlowMap.__init__(ds)` is public and a flow map built from
   a bare dataset has no seed to ask. Three options: build it lazily on the flow
   map and cache per instance, which rebuilds it once per flow map; cache it on
   the seed grid and have `pset_to_flowmap` hand it over, which works for the
   normal path and is absent for the hand-built one; or a module-level cache
   keyed on the points array, which is a global.

   The amortisation table above says this is not urgent. Per-instance costs
   ~1 s per flow map at example scale and only bites at $10^6$ grid points, so
   the per-instance cache is a defensible default with the seed-grid cache as a
   later optimisation. Confirm that reading, or pick one now.
2. **Does `from_axes` survive as its own constructor?** With one class and one
   interpolator it is `from_points` over the outer product of two axes, plus the
   `(i, j)` dims that let a diagnostic plot as a field. The dims are worth
   keeping. Is it two constructors on one class, or one constructor and a
   documented recipe?
3. **Is `Neighbor*` still worth keeping?** After the collapse it differs from
   `Auxiliary*` only in its gradient, and it is the stencil that ties the
   gradient step to the seed resolution and produces NaN edges. Out of scope for
   this plan, but the collapse makes the question visible.
4. **Does `image` want a warm start too?** It has arbitrary query points and one
   shot, so no hint. The curve-evolution path calls it at every vertex of a
   curve, which *is* spatially coherent, so a sorted-query warm start would help
   there. Worth it, or leave `image` cold?
5. **Does `_RESERVED_DIMS` still need `position`** and the rest, once there is
   one class and possibly a different internal dim for the two position
   components?
6. **Chunk size for the windowing**: a fixed block, or one sized from a memory
   budget the caller can set?
7. **Naming.** `examples/cabo_verde_unstructured` demonstrates a class that will
   not exist. Rename it, fold it into `cabo_verde_lcs`, or leave it as the
   point-set example? The `AGENTS.md` notebook-rank list moves with the answer.
8. **Is the ~2x tracing cost worth stating as a knob?** A caller who knows their
   grid is rectilinear can no longer ask for bilinear. Nobody has asked for it,
   and adding the option back rebuilds the seam this plan deletes.
9. **Does the collapse survive $10^8$?** At basin resolution the rectilinear path
   builds nothing and is $O(1)$, while the Delaunay is ~21 GiB, roughly an hour,
   and the only one of the three structures that does not tile cleanly. Deleting
   it would remove the path that handles the grid on the roadmap. Keep both and
   fix only the scattered one?
10. **Does the radius-$r$ ring gradient earn a place at basin scale?** It cuts
    $4\times10^8$ particles to $10^8$ where that is the binding constraint, at
    second order in $r$, but needs about 1/200 degree spacing before a 1 km
    neighbourhood exists, wants a well-definedness guard below about five
    neighbours, and is knife-edged at a radius that lands on lattice points
    (error 1.91e-5 at $r$ = 1.5 km against 4.78e-4 at 2.0 km on a 1 km lattice,
    because 0.5% variation in $\cos\phi$ flips two neighbours in or out).
11. **Where does a centre-framed separation helper live?** It belongs beside
    `_separation_m` in `grids`, and the two are correct for different stencils
    with no way to tell from the call site which one a caller wanted.
12. **The kd-tree windowing is the more urgent question**, and it is already in
    `main`'s path rather than in this plan. It is $O(nw^2)$ where the rolling
    window it replaced was $O(nw)$, and it dies at 1/100 degree over a
    40x40 degree domain on this machine. Options: bin-and-prune (keeps one rule
    for both layouts, $O(n)$ filter), chunk the query (fixes memory, keeps the
    work), add `bottleneck` and restore a rolling window for rectilinear grids
    (fastest, reintroduces a layout branch), or accept it and require tiling.
    Decide this before the collapse.
