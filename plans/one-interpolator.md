# Plan: one interpolator, four classes

Collapse the layout axis PR #47 introduced. Delete `rectilinear_interpolator`,
delete the `UnstructuredAuxiliary*` pair, make `FlowMap._interpolator` a single
concrete method, and move `from_points` onto `AuxiliarySeedGrid`. Six public
classes become four and one abstract member goes.

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
   a bare dataset has no seed to ask. Three options, none free: build it lazily
   on the flow map and cache per instance, which rebuilds it for every one of the
   $N$ flow maps in a release series and gives up the amortisation; cache it on
   the seed grid and have `pset_to_flowmap` hand it over, which works for the
   normal path and is absent for the hand-built one; or a module-level cache
   keyed on the points array, which is a global. Which?
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
