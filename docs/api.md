# API guide

The public surface of `lcs_parcels` is two sibling families, a
`SeedGrid` family and a `FlowMap` family, each with a shared abstract base and two
concrete stencils, exported from the package root:

```python
from lcs_parcels import SeedGrid, NeighborSeedGrid, AuxiliarySeedGrid, FlowMap, NeighborFlowMap, AuxiliaryFlowMap
```

`ftle_ridge_seeds`, `shrink_lines` and `prune_shrink_lines` turn a `FlowMap`'s
strain field into hyperbolic-LCS curves (see
[Hyperbolic LCS: shrink lines](#hyperbolic-lcs-shrink-lines) below), and
`FlowMap.hyperbolic_lcs()` runs them in one call.

Every adjacent same-typed argument pair on the public surface, meaning every
lon/lat pair, is **keyword-only**, so a transposed call raises `TypeError`.

A `SeedGrid` lays out reference positions and emits a particle set for Parcels; the
advected positions are ingested back into a `FlowMap`, which computes the
deformation gradient $\nabla F$ and everything downstream (Cauchy–Green $C$, its
eigen-decomposition, and the FTLE). The two families are **siblings**: neither
class is a subclass of the other, so a `SeedGrid` has no diagnostics and a `FlowMap`
emits no particle set.
Symbols and units are defined in [`notation.md`](notation.md). The type
structure and the session walkthrough are in
[`architecture.md`](architecture.md). The local east/north frame that
separations are measured in, and the tuning parameters, are in
[`numerics.md`](numerics.md). Naming follows
Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

The package contains **no Parcels code**: a `SeedGrid` emits particle sets and
ingests advected positions; Parcels (external) owns the integration.

## Data model

Each object wraps an `xr.Dataset`, held in `.ds`; these are not `xr.Dataset`
subclasses, so xarray calls go through `.ds`. Logical grid dims are `i, j`. A
**`SeedGrid` is all-coordinates** (no data variables) and carries no time: it holds the
diagnostic grid points and the reference release positions $x_0$. A **`FlowMap`
adds the advected positions** as its only data variables, plus scalar `t0`/`T`
coordinates.

| Name | Role | Dims | Kind | On |
|---|---|---|---|---|
| `lon_grid`, `lat_grid` | diagnostic grid points (every diagnostic is reported here) | `(i, j)` | coords | seed + flow map, **both stencils** |
| `lon_0`, `lat_0` | reference *release* positions $x_0$ | `(i, j)`; `(i, j, displacement)` for `Auxiliary*` | coords | seed + flow map |
| `lon`, `lat` | advected positions, the flow map $F_{t_0}^{t_1}(x_0)$ | same dims as `lon_0`/`lat_0` | data vars | flow map only |
| `displacement` | auxiliary stencil arm, `['east', 'north', 'west', 'south']` | `(displacement,)` | coord | `Auxiliary*` only |
| `t0` | release time | scalar | coord | flow map only |
| `T` | signed integration window $T = t_1 - t_0$ (`timedelta64`) | scalar | coord | flow map only |

`lon_grid`/`lat_grid` have the same meaning on both stencils: they label every
diagnostic and are what a gridded plot is drawn against. For the `Neighbor*`
classes the release position *is* the grid point, so `lon_grid` equals `lon_0`
there, and both pairs are present. For the `Auxiliary*` classes the release
positions are the four stencil arms, so `lon_0`/`lat_0` (and the advected
`lon`/`lat` on a flow map) carry the extra `displacement` dim and hold the arm
positions explicitly.

A single `FlowMap` carries `t0`/`T` as *scalar* coords; `t1` is not stored,
being recoverable as `t0 + T`.

Longitudes are stored in whatever convention they arrive in ($[-180, 180)$,
$[0, 360)$, or anything else) and are never renormalised, so what comes back
sits on the branch that went in. Only longitude *differences* and *means* are
wrapped, so a stencil straddling the antimeridian differences correctly whatever
branch its points sit on.

The `lon_grid` **axis** itself must be monotonic, which is a stricter
requirement: `FlowMap.image` and `shrink_lines` interpolate along it. Seed a
domain crossing the antimeridian on `170, 175, 180, 185`, not on
`170, 175, 180, -175`. On the latter, `hyperbolic_lcs()` raises `ValueError`
("the points in dimension 0 must be strictly ascending or descending") out of
SciPy, and `image()` does not raise at all. It reads the axis as if it were
sorted, so a point in the wrapped half of the domain comes back `NaN` or
interpolated between the wrong two grid points.

Both families expose the diagnostic grid directly, so no consumer indexes `.ds`
for it:

```text
SeedGrid.lon_grid -> xr.DataArray        # property, (i, j), degrees east
SeedGrid.lat_grid -> xr.DataArray        # property, (i, j), degrees north
FlowMap.lon_grid -> xr.DataArray     # same, on the flow map
FlowMap.lat_grid -> xr.DataArray
FlowMap.grid_image -> xr.Dataset     # abstract property (per-stencil)
```

- **`grid_image`** is the flow map image of the diagnostic grid points,
  $F_{t_0}^{t_1}(x_{\mathrm{grid}})$: `lon`/`lat` (degrees) on `(i, j)`, one
  advected position per grid point whatever stencil it was released with. For
  `NeighborFlowMap` that is the advected positions unchanged; for
  `AuxiliaryFlowMap` it is the centroid of the four advected arms, with the
  longitude averaged on the circle so four arms straddling the antimeridian
  average to a position between them, taken with `skipna=False` so a single
  lost arm makes the whole grid point `NaN`.

Both `SeedGrid` and `FlowMap` have a terse `repr`, one line for the seed grid and
two for the flow map; display `.ds` to see the dataset itself.

```pycon
>>> seed
<NeighborSeedGrid 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00>
>>> flowmap
<NeighborFlowMap 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00,
                 t0 2020-01-01T00:00:00, T +7.0 days>
```

## Constructors and round-trip

A seed is built, emitted, ingested into a flow map, and (optionally) collapsed
back to a seed. Two calls cross between the families, and neither object holds a
reference to the other.

```text
SeedGrid.from_axes(*, lon, lat) -> Self                       # classmethod (abstract)
SeedGrid.to_parcels_pset() -> tuple[list, list]               # concrete (base)
SeedGrid.pset_to_flowmap(*, lon, lat, t0, t1) -> FlowMap      # concrete (base)
FlowMap.to_seed() -> SeedGrid                                 # concrete (base)
```

- **`from_axes(*, lon, lat)`** builds a seed grid from 1-D lon/lat axes,
  broadcast into 2-D fields on `(i, j)` (lon varying along
  `i`, lat along `j`) and stored as the diagnostic grid `lon_grid`/`lat_grid`.
  No time is recorded: `t0` and the window `T` enter only at
  `pset_to_flowmap`. `NeighborSeedGrid.from_axes` stores those same points as the
  reference positions `lon_0`/`lat_0`. `AuxiliarySeedGrid.from_axes` also
  takes a keyword-only `aux_separation_m` (the controlled arm separation $s$ in
  metres; default `1000.0`), lays out the fixed four-arm
  `displacement = ['east', 'north', 'west', 'south']` stencil at $\pm s$ about
  each grid point, in that point's own local east/north frame so the east–west
  and north–south arm spans are $2s$ at every latitude, and stores those arms
  explicitly as `lon_0`/`lat_0` on `(i, j, displacement)`. It raises
  `ValueError` if $s$ would span 90 degrees of longitude or more at any grid
  point, which happens closer to a pole than about $0.64\,s$: 640 m for the
  default $s = 1$ km, 32 km for $s = 50$ km. The longitude increment an eastward
  offset of $s$ metres needs grows without bound towards the pole, so the
  east–west arms are rejected rather than approximated.
- **`to_parcels_pset()`** flattens the *reference* release positions to plain
  `(lon, lat)` lists (a 2-tuple) over the `particle` index (`('i', 'j')`, plus
  `'displacement'` for `AuxiliarySeedGrid`). The auxiliary arms are emitted directly
  from the explicit `lon_0`/`lat_0`. The 2-tuple is unpacked before it is fed
  back in, since `pset_to_flowmap` is keyword-only:

  ```python
  lon_0, lat_0 = seed.to_parcels_pset()
  # ... advect (lon_0, lat_0) with Parcels, collect (lon1, lat1) ...
  flowmap = seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)
  ```

- **`pset_to_flowmap(*, lon, lat, t0, t1)`** reattaches the flat advected
  positions onto the seed's `particle` index as the advected `lon`/`lat`,
  leaving the reference `lon_0`/`lat_0` (and any auxiliary geometry) untouched,
  and produce the paired concrete `FlowMap`. It records the release time `t0`
  and the derived signed window `T = t1 - t0` as scalar coordinates (`t1` itself
  is not stored). Lost particles arrive as `NaN` and propagate through the
  diagnostics as `NaN`. Backward integration is selected by passing a `t1`
  before `t0` (negative `T`); no separate direction flag exists. A **zero
  window** (`t1 == t0`) is rejected with `ValueError`, since the FTLE's $1/|T|$
  would divide by zero.
- **`to_seed()`** drops the advected `lon`/`lat` and the scalar `t0`/`T`
  coords, recovering the paired `SeedGrid`; the lossless inverse of
  `pset_to_flowmap`. Re-emitting reproduces the same flat particle set. For
  `Auxiliary*` this rebuilds from the carried arms, needing neither the original
  axes nor `aux_separation_m`.

The reusable-template workflow (same grid, sweep `t1`, or re-release at a new
`t0`) is therefore `flowmap.to_seed()` then
`seed.pset_to_flowmap(lon=..., lat=..., t0=..., t1=...)`, with no shared state:
the seed is a spatial template and every release passes its own `(t0, t1)`.

## Operators

The diagnostics live on `FlowMap`; a `SeedGrid` has none (it carries no
advected positions or window).

```text
FlowMap.deformation_gradient() -> xr.DataArray   # abstract (per-stencil)
FlowMap.cauchy_green() -> xr.DataArray            # concrete (base)
FlowMap.cg_eigen() -> xr.Dataset                  # concrete (base)
FlowMap.ftle() -> xr.DataArray                    # concrete (base)
```

- **`deformation_gradient()`** returns $\nabla F = \partial(\text{lon},
  \text{lat}) / \partial(\text{lon}_0, \text{lat}_0)$ as advected separations
  (numerator, from the ingested outputs) over reference separations
  (denominator). Each separation is taken in metres in the local east/north
  frame of the two points it connects (see
  [`notation.md`](notation.md#the-local-east-north-frame)). The accuracy of a
  separation is limited by the span of the stencil, not by the size or position
  of the domain; the error series is in
  [`numerics.md`](numerics.md#the-local-east-north-frame). Dims `i`, `j`,
  `row`, `col`, a *set* rather than an order. The package is label-based, so the axis
  order the call returns is not part of the contract (today it is
  `('row', 'col', 'i', 'j')`, and that may change).
  `row`/`col` are dimension coordinates valued `['x', 'y']` and
  `gradF.sel(row=a, col=b) = dF_a / dx0_b`. There is **no** `comp` coord on the
  tensor. The stencil is per-subclass:
  - `NeighborFlowMap`: central difference against neighbours
    `(i +/- 1, j +/- 1)`; domain-edge cells are `NaN` by construction.
  - `AuxiliaryFlowMap`: per-point four-arm central difference (east-west,
    north-south over `2s`); defined at every grid point, no `NaN` edges.
- **`cauchy_green()`** returns $C = (\nabla F)^\top \nabla F$, symmetric, on the same
  dim set `i`, `j`, `row`, `col` (again in no contractual order, though as shipped it
  is `('row', 'i', 'j', 'col')`). Select by label, and `.transpose()` yourself
  if you need a particular layout.
- **`cg_eigen()`** is the eigen-decomposition of $C$ via `np.linalg.eigh`. Returns a
  `Dataset` with `lambda` on `(i, j, eig)` (eigenvalues **ascending**,
  $0 < \lambda_1 \le \lambda_2$, `eig = [0, 1]`) and `xi` on
  `(i, j, comp, eig)` (orthonormal eigenvectors, `comp = ['x', 'y']`).
- **`ftle()`** returns $\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ using the
  *largest* eigenvalue $\lambda_2$ and the recorded signed window `T`. Dims
  `(i, j)`, units 1/second. Only $|T|$ enters, so backward and forward
  integration of the same map give the same FTLE.

All four are reported at the diagnostic grid points and carry
`lon_grid`/`lat_grid` as their position coords, and only those. The release
positions `lon_0`/`lat_0` are *not* carried through: on an `AuxiliaryFlowMap`
they are one stencil arm, which would mislabel an `(i, j)` quantity.

A single `NaN` (lost particle or missing stencil point) propagates
`gradF -> C -> eigen -> ftle` with no special-casing.

## Hyperbolic LCS: shrink lines

Three module-level functions (in `lcs_parcels.tensorlines`, exported from the
package root) turn a `FlowMap`'s strain field into LCS **curves**, following
Haller (2015) §5.1 / Table 1 ($n = 2$):

```python
from lcs_parcels import ftle_ridge_seeds, prune_shrink_lines, shrink_lines

ftle = flowmap.ftle()
seeds = ftle_ridge_seeds(ftle)                                              # seed points
lines = shrink_lines(flowmap, seed_lon=seeds["lon"], seed_lat=seeds["lat"])  # curves
lines = prune_shrink_lines(lines, ftle=ftle)                                # drop duplicates
```

A repelling LCS is a **shrink line**, a curve tangent to the weak-stretch
eigenvector $\xi_1$ of $C$, solving the tensor-line ODE $\dot r = \xi_1(r)$.
Attracting LCS need no separate call: by the forward–backward duality (Haller &
Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)) they are the shrink
lines of the *backward* flow, so `shrink_lines` of a **forward** `FlowMap` gives
repelling LCS and of a **backward** one gives attracting LCS.

```text
ftle_ridge_seeds(ftle, *, window_m=30_000.0, quantile=None,
                 ftle_min=None) -> xr.Dataset
shrink_lines(flowmap, *, seed_lon, seed_lat, min_anisotropy=1.15,
             step_m=3_000.0, line_length_m=1_500_000.0) -> xr.Dataset
prune_shrink_lines(lines, *, ftle=None, window_m=30_000.0) -> xr.Dataset
```

Every tuning parameter is stated in the units of the thing itself: metres for
the three lengths, a dimensionless eigenvalue ratio for the guard, the field's
own units for `ftle_min`. The three lengths, the ratio and `quantile` mean the
same thing at any resolution and over any window; `ftle_min` is an absolute
threshold, so it does not.

- **`ftle_ridge_seeds(ftle)`** picks seed points at strong local maxima of an
  FTLE field: grid points that are the maximum over a square window of side
  `window_m` **metres** (a windowed local maximum on the raw value) *and* at or
  above a magnitude floor. `window_m` is converted to an odd cell count per
  dimension from the field's own `lon_grid`/`lat_grid` spacing (the default 30 km
  is 7 cells on a $1/25^\circ$ grid at $20^\circ$N). The window reaches
  `window_m / 2` to either side of its own grid point, so two seed points can be
  as close as about `window_m / 2`, not `window_m`; the returned
  `min_seed_separation_m` attribute is that distance computed on this grid, off
  its smallest cell, so it is a floor rather than a typical spacing. That floor
  bounds *strict* maxima: a plateau of exactly equal values makes every one of
  its cells a windowed maximum, and those can be adjacent. A `window_m` spanning
  fewer than three cells in either dimension emits a `UserWarning`: a one-cell
  window makes every point a windowed maximum, so the local-maximum test stops
  selecting and only the floor is left.

  The floor is set one of two ways, and passing both raises `ValueError`.
  `quantile` is a quantile of this field, in $[0, 1]$; `ftle_min` is an absolute
  value in the units of `ftle` (1/s for `flowmap.ftle()`). Passing neither uses
  `quantile=0.90`, the top decile. Use `ftle_min` when several runs (other
  windows, other regions) have to be compared against one threshold, which a
  per-field quantile cannot give.

  The field must carry `lon_grid`/`lat_grid`; `flowmap.ftle()` does. NaN cells
  never qualify.

  Returns an `xr.Dataset` with `lon`/`lat` (degrees) on a `seed` dim, one entry
  per seed point, and a `seed` index coordinate. Its `attrs` record what the
  selection did: `selector` (`"quantile"` or `"ftle_min"`) and the
  `ftle_threshold` it resolved to, plus `window_m`, the odd cell counts
  `window_cells_i`/`window_cells_j` it became, the median grid spacings
  `grid_spacing_i_m`/`grid_spacing_j_m` it was measured against, and
  `min_seed_separation_m`.
- **`shrink_lines(flowmap, seed_lon=..., seed_lat=...)`** integrates the
  $\xi_1$ tensor line through each seed point, in both directions, on the flow
  map's rectilinear grid. It interpolates the tensor $C$ (via `scipy`'s
  `RegularGridInterpolator`) and re-diagonalises at each step, which is robust
  to the eigenvector sign ambiguity, then orients each step to the running
  heading. `line_length_m` is a **cap**, not the achieved length: it is traced
  half in each direction, with $n = $ `line_length_m / (2 * step_m)` steps per
  direction (rounded, at least 1), so a line that runs the full budget has
  $2n + 1$ points. A line stops early where $\xi_1$ stops being a well-defined
  direction, where $\lambda_2 / \lambda_1$ falls below `min_anisotropy`, or
  where it leaves the grid or hits a NaN cell. Returns an `xr.Dataset` with
  `lon`/`lat` on dims `(line, point)`, one `line` per seed point; every row is
  the same length, NaN-filled past termination, and a seed point that cannot be
  traced at all is an all-NaN row.
- **`prune_shrink_lines(lines, *, ftle=None, window_m=...)`** drops shrink
  lines that duplicate a stronger one, which is what several seeds on one ridge
  produce. Rows that are NaN at every point are dropped first and cover
  nothing. The remaining lines are ranked by the FTLE integrated along them
  (`ftle_mean * length_m`, the FTLE interpolated from `ftle` at each line's
  points), or by `length_m` alone when no `ftle` is given, and walked from the
  strongest down. A candidate is dropped once one
  of its points falls within `window_m / 2` of an already-kept line's nearest
  point *and* the arc length of its uncovered stretch is below `window_m`. A
  line sharing nothing with a stronger line is always kept, whatever its
  length, and a kept-or-dropped decision is made for the whole line, never a
  part of it. Returns `lines` restricted to the kept rows, the `line` coord
  keeping its **original labels**, plus `length_m` (m) and, with `ftle`,
  `ftle_mean` (1/s) on `line`. The attributes `window_m`,
  `tube_radius_m`, `min_new_length_m`, `n_lines_in` and `n_lines_dropped`
  record what was run. Why the score is a line integral and why the tube
  radius and minimum new length are `window_m / 2` and `window_m` are in
  [`numerics.md`](numerics.md#why-pruning-is-a-run-length-in-a-tube).

**`min_anisotropy`** is a floor on $\lambda_2 / \lambda_1$, the ratio of the two
Cauchy–Green eigenvalues, and so dimensionless. It is a **well-definedness
guard, not an LCS selector**. The magnitude floor in `ftle_ridge_seeds`
(`quantile` or `ftle_min`) is what selects. At the default 1.15 a
1% error in $C$ swings $\xi_1$ by about 2 degrees; at a ratio of 1.05 by 6
degrees. Being a ratio it carries no $T$, no grid scale and no stretching rate,
so 1.15 means the same thing for a six-hour laboratory flow and a six-month
basin-scale one. Its useful range starts just above 1: $C$ is positive
semi-definite, so $\lambda_2 \ge \lambda_1 \ge 0$ always, and any
`min_anisotropy` at or below 1 makes the guard unsatisfiable, so it never fires,
and lines then stop only by leaving the grid or hitting a NaN cell.

This layer interpolates on the axis-aligned `lon_grid`/`lat_grid` axes, so (like
`NeighborFlowMap`) it assumes a rectilinear flow map, and the `lon_grid` axis
must be monotonic. The traced lines themselves are unconstrained: a line steps
by adding a longitude increment to its current longitude, so it crosses the
antimeridian on the branch its seed point came in on.

### One call: `FlowMap.hyperbolic_lcs()`

```text
FlowMap.hyperbolic_lcs(*, window_m=None, quantile=None, ftle_min=None,
                       min_anisotropy=None, step_m=None,
                       line_length_m=None) -> xr.Dataset
```

Runs `ftle()`, `ftle_ridge_seeds`, `shrink_lines` and `prune_shrink_lines` in
one call, computing the FTLE exactly once and pruning
at the same `window_m` the seeding used. Every parameter is optional and only
the ones actually passed are forwarded, so the defaults stay in
`ftle_ridge_seeds`/`shrink_lines`/`prune_shrink_lines`, including the rule that
`quantile` and `ftle_min` are mutually exclusive, which raises `ValueError`
from `ftle_ridge_seeds` when both are passed here. There is no direction
argument: the flow map already carries $\mathrm{sign}(T)$, so a forward map
yields repelling LCS and a backward one attracting LCS, and the returned
dataset says which family it holds. `hyperbolic_lcs()` extracts hyperbolic
(repelling and attracting) LCS only.

The result is the pruned `shrink_lines` dataset, `lon`/`lat` on `(line, point)`
with no all-NaN row, `ftle_mean` (1/s) and `length_m` (m) on `line`, and the
`ftle` field on `(i, j)` the seed points were picked from carried along, so the
curves can be plotted over it without recomputing an eigendecomposition. The
`line` coordinate labels the seed indices that survived pruning rather than
every seed's original index. The ridge-selection and pruning attributes ride
along too, so `lcs.attrs["selector"]`, `lcs.attrs["ftle_threshold"]`,
`lcs.attrs["min_seed_separation_m"]` and `lcs.attrs["n_lines_dropped"]` are
readable off the result without re-running the seeding or the pruning:

```python
lcs = forward.hyperbolic_lcs()
lcs["ftle"].plot(x="lon_grid", y="lat_grid")
plt.plot(lcs["lon"].T, lcs["lat"].T)
```

`lon_grid`/`lat_grid` are 2-D non-dimension coords (the layout that lets the
grid be curvilinear), so a gridded field is drawn against them by passing
`x="lon_grid", y="lat_grid"`; `.plot()` on its own falls back to the logical
`i`/`j` axes.

Ridge-finding itself takes a *field*, not a flow map, so to pick seed points
from a smoothed or masked FTLE, or to see the lines before pruning, run the
steps by hand.

## Evolving a material curve

An extracted LCS is a **material** curve, so its later positions are fixed by the
flow: $\mathcal{M}(t) = F_{t_0}^{t}(\mathcal{M}(t_0))$ (Haller 2015, Eq. 5).
`FlowMap.image` applies the stored flow map to arbitrary reference points, so a
curve is evolved by interpolating that map at the curve's vertices, with no
second advection.

```text
FlowMap.image(*, lon_0, lat_0) -> xr.Dataset
```

- **`image(*, lon_0, lat_0)`** interpolates `grid_image` (the advected positions
  on the diagnostic grid) at reference points `lon_0`/`lat_0` (`DataArray`s on any
  shared dims, e.g., the `(line, point)` grid of `shrink_lines`), returning their
  advected positions $F_{t_0}^{t_1}(x_0)$ as an `xr.Dataset` with `lon`/`lat` on
  the input dims, the same structure a `shrink_lines` curve has, so an evolved
  curve plots the same way and can be passed back into `image`. The requested
  reference positions ride along as `lon_0`/`lat_0` coords on the output. Points
  off the grid, in a NaN (land/edge) cell, or NaN themselves map to NaN.
  Rectilinear grids only, like `shrink_lines`, with a monotonic `lon_grid` axis.

  The advected longitudes may arrive on any branch. Each is re-anchored on the
  branch of the grid point it came from before the interpolation, so an
  advection that hands positions back wrapped to $[-180, 180)$ is read
  correctly. The returned longitudes are on the branch `lon_0` was given in.

An LCS is evolved in its **coherent** direction, where perturbations decay: an
attracting LCS forward in time, a repelling one backward. Advect the grid to a
few horizons and call `image` at each to carry the curve through them; see
[`examples/cabo_verde_lcs_evolution.py`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_lcs_evolution.py).

## Output metadata

Every array the package returns carries `name`, `long_name`, and `units`, and no
two quantities come back under the same `name`. That metadata *is* the axis
label, the title, and the colorbar caption of a vanilla `.plot()`, so a returned
field needs no hand-set labels to be readable. The only keyword a gridded
diagnostic needs is `x="lon_grid", y="lat_grid"` to be drawn geographically
rather than against the logical `i`/`j` axes.

| Returned by | `name` | `long_name` | `units` |
|---|---|---|---|
| `deformation_gradient()` | `deformation_gradient` | deformation gradient grad F of the flow map | `1` |
| `cauchy_green()` | `cauchy_green` | right Cauchy-Green strain tensor C = (grad F)^T grad F | `1` |
| `cg_eigen()["lambda"]` | `lambda` | eigenvalue lambda of the Cauchy-Green tensor | `1` |
| `cg_eigen()["xi"]` | `xi` | eigenvector xi of the Cauchy-Green tensor | `1` |
| `ftle()` | `ftle` | finite-time Lyapunov exponent | `1/s` |
| `image()`, `grid_image` | `lon` / `lat` | longitude/latitude of the advected position F(x_0) | `degrees_east` / `degrees_north` |
| `ftle_ridge_seeds()` | `lon` / `lat` | longitude/latitude of the FTLE ridge seed | `degrees_east` / `degrees_north` |
| `shrink_lines()` | `lon` / `lat` | longitude/latitude along the shrink line | `degrees_east` / `degrees_north` |
| `prune_shrink_lines()` | `ftle_mean` | mean FTLE along the shrink line | `1/s` |
| `prune_shrink_lines()` | `length_m` | arc length of the shrink line | `m` |
| `hyperbolic_lcs()` | `lon` / `lat` | longitude/latitude along the repelling (or attracting) LCS | `degrees_east` / `degrees_north` |
| `hyperbolic_lcs()` | `ftle_mean` | mean FTLE along the shrink line | `1/s` |
| `hyperbolic_lcs()` | `length_m` | arc length of the shrink line | `m` |

Units are SI: the FTLE is `1/s`, not 1/day. Convert for display in the plotting
code, where the conversion is visible, and re-set `units` when you do.

The coordinates carry the same metadata, set once at construction
(`from_axes`, `pset_to_flowmap`) and unchanged through the diagnostics:

| Coord | `long_name` | `units` |
|---|---|---|
| `i` / `j` | logical grid index along i / j |  |
| `lon_grid` / `lat_grid` | longitude / latitude | `degrees_east` / `degrees_north` |
| `lon_0` / `lat_0` | longitude/latitude of the reference release position x_0 | `degrees_east` / `degrees_north` |
| `displacement` | auxiliary stencil arm |  |
| `t0` | release time t0 |  |
| `T` | signed integration window T = t1 - t0 |  |
| `row` / `col` | tensor row / column index |  |
| `comp` | eigenvector component |  |
| `eig` | Cauchy-Green eigenpair, ascending: 0 is the weak-stretch lambda_1, 1 is lambda_max = lambda_2 |  |
| `seed` | FTLE ridge seed index |  |
| `line` | shrink line index, one per seed point |  |
| `point` | point index along the shrink line |  |

The index and label coords (`i`, `j`, `displacement`, `row`, `col`, `comp`,
`eig`, `seed`, `line`, `point`) carry no `units`: their values are logical indices or
string labels, so there is no unit to give. `t0` and `T` carry none either,
they are `datetime64`/`timedelta64`, so the dtype already holds the unit.

## References

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

Haller, G. & Sapsis, T. (2011). *Lagrangian coherent structures and the smallest
finite-time Lyapunov exponent.* Chaos, 21, 023115.
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
