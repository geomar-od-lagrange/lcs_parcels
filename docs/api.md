# API reference: seeds and flow maps

The public surface of `lcs_parcels` is two sibling families — a time-free
`Seed` family and a `FlowMap` family — each with a shared abstract base and two
concrete stencils, exported from the package root:

```python
from lcs_parcels import Seed, NeighborSeed, AuxiliarySeed, FlowMap, NeighborFlowMap, AuxiliaryFlowMap
```

Plus two functions that turn a `FlowMap`'s strain field into hyperbolic-LCS
curves — `ftle_ridge_seeds` and `shrink_lines` (see
[Hyperbolic LCS: shrink lines](#hyperbolic-lcs-shrink-lines) below), which
`FlowMap.hyperbolic_lcs()` runs in one call.

Every adjacent same-typed argument pair on the public surface — every
lon/lat pair — is **keyword-only**, so a transposed call raises `TypeError`.

A `Seed` lays out reference positions and emits a particle set for Parcels; the
advected positions are ingested back into a `FlowMap`, which computes the
deformation gradient $\nabla F$ and everything downstream (Cauchy–Green $C$, its
eigen-decomposition, and the FTLE). The two families are **siblings, not an
inheritance pair**: neither class is a subclass of the other, so a `Seed` has no
diagnostics and a `FlowMap` emits no particle set.
Symbols and units are defined in [`notation.md`](notation.md); the type
structure and the session walkthrough are in
[`architecture.md`](architecture.md), and the local east/north frame separations
are measured in, together with the tuning parameters, is in
[`numerics.md`](numerics.md). Naming follows
Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

The package contains **no Parcels code**: a `Seed` emits particle sets and
ingests advected positions; Parcels (external) owns the integration.

## Data model

Each object wraps an `xr.Dataset`, held in `.ds` — these are not `xr.Dataset`
subclasses, so xarray calls go through `.ds`. Logical grid dims are `i, j`. A
**`Seed` is time-free
and all-coordinates** (no data variables): it holds the diagnostic grid points
and the reference release positions $x_0$. A **`FlowMap` adds the advected
positions** as its only data variables, plus scalar `t0`/`T` coordinates.

| Name | Role | Dims | Kind | On |
|---|---|---|---|---|
| `lon_grid`, `lat_grid` | diagnostic grid points (every diagnostic is reported here) | `(i, j)` | coords | seed + flow map, **both stencils** |
| `lon_0`, `lat_0` | reference *release* positions $x_0$ | `(i, j)`; `(i, j, displacement)` for `Auxiliary*` | coords | seed + flow map |
| `lon`, `lat` | advected positions, the flow map $F_{t_0}^{t_1}(x_0)$ | same dims as `lon_0`/`lat_0` | data vars | flow map only |
| `displacement` | auxiliary stencil arm, `['east', 'north', 'west', 'south']` | `(displacement,)` | coord | `Auxiliary*` only |
| `t0` | release time | scalar | coord | flow map only |
| `T` | signed integration window $T = t_1 - t_0$ (`timedelta64`) | scalar | coord | flow map only |

`lon_grid`/`lat_grid` carries the same meaning on both stencils: it labels every
diagnostic and is what a gridded plot is drawn against. For the `Neighbor*`
classes the release position *is* the grid point, so `lon_grid` equals `lon_0`
there, and both pairs are present. For the `Auxiliary*` classes the release
positions are the four stencil arms, so `lon_0`/`lat_0` (and the advected
`lon`/`lat` on a flow map) carry the extra `displacement` dim and hold the arm
positions explicitly.

A single `FlowMap` carries `t0`/`T` as *scalar* coords; `t1` is not stored,
being recoverable as `t0 + T`.

Longitudes are stored in whatever convention they arrive in — $[-180, 180)$,
$[0, 360)$, or anything else — and are never renormalised, so what comes back
sits on the branch that went in. Only longitude *differences* and *means* are
wrapped, so a stencil straddling the antimeridian differences correctly whatever
branch its points sit on.

The `lon_grid` **axis** itself must be monotonic, which is a stricter
requirement: `FlowMap.image` and `shrink_lines` interpolate along it. Seed a
domain crossing the antimeridian on `170, 175, 180, 185`, not on
`175, 178, -179, -176`. On the latter, `hyperbolic_lcs()` raises `ValueError`
("the points in dimension 0 must be strictly ascending or descending") out of
SciPy, and `image()` does not raise at all — it reads the axis as if it were
sorted, so a point in the wrapped half of the domain comes back `NaN` or
interpolated between the wrong two grid points.

Both families expose the diagnostic grid directly, so no consumer indexes `.ds`
for it:

```text
Seed.lon_grid -> xr.DataArray        # property, (i, j), degrees east
Seed.lat_grid -> xr.DataArray        # property, (i, j), degrees north
FlowMap.lon_grid -> xr.DataArray     # same, on the flow map
FlowMap.lat_grid -> xr.DataArray
FlowMap.grid_image -> xr.Dataset     # abstract property (per-stencil)
```

- **`grid_image`** — the flow map image of the diagnostic grid points,
  $F_{t_0}^{t_1}(x_{\mathrm{grid}})$: `lon`/`lat` (degrees) on `(i, j)`, one
  advected position per grid point whatever stencil it was released with. For
  `NeighborFlowMap` that is the advected positions unchanged; for
  `AuxiliaryFlowMap` it is the centroid of the four advected arms — the
  longitude averaged on the circle, so four arms straddling the antimeridian
  average between themselves — taken with `skipna=False` so a single lost arm
  makes the whole grid point `NaN`.

Both `Seed` and `FlowMap` have a terse one-line `repr`; `.ds` remains how the
dataset itself is displayed.

```pycon
>>> seed
<NeighborSeed 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00>
>>> flowmap
<NeighborFlowMap 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00, t0 2020-01-01T00:00:00, T +7.0 days>
```

## Constructors and round-trip

A seed is built, emitted, ingested into a flow map, and (optionally) collapsed
back to a seed. The two families are connected by two crossings; neither holds a
reference to the other.

```text
Seed.from_axes(*, lon, lat) -> Self                       # classmethod (abstract)
Seed.to_parcels_pset() -> tuple[list, list]               # concrete (base)
Seed.pset_to_flowmap(*, lon, lat, t0, t1) -> FlowMap      # concrete (base)
FlowMap.to_seed() -> Seed                                 # concrete (base)
```

- **`from_axes(*, lon, lat)`** — build a time-free seed from 1-D lon/lat axes
  (length `Ni`, `Nj`), broadcast into 2-D fields on `(i, j)` (lon varying along
  `i`, lat along `j`) and stored as the diagnostic grid `lon_grid`/`lat_grid`.
  No time is recorded — the seed is time-free; `t0` and the window `T` enter
  only at `pset_to_flowmap`. `NeighborSeed.from_axes` stores those same points
  as the reference positions `lon_0`/`lat_0`. `AuxiliarySeed.from_axes` also
  takes a keyword-only `aux_separation_m` (the controlled arm separation $s$ in
  meters; default `1000.0`), lays out the fixed four-arm
  `displacement = ['east', 'north', 'west', 'south']` stencil at $\pm s$ about
  each grid point — in that point's own local east/north frame, so the east–west
  and north–south arm spans are $2s$ at every latitude — and stores those arms
  explicitly as `lon_0`/`lat_0` on `(i, j, displacement)`. It raises
  `ValueError` if $s$ would span 90 degrees of longitude or more at any grid
  point, which happens closer to a pole than about $0.64\,s$: 640 m for the
  default $s = 1$ km, 32 km for $s = 50$ km. There is no east there, so the
  east–west arms are rejected rather than approximated.
- **`to_parcels_pset()`** — flatten the *reference* release positions to plain
  `(lon, lat)` lists (a 2-tuple) over the `particle` index (`('i', 'j')`, plus
  `'displacement'` for `AuxiliarySeed`). The auxiliary arms are emitted directly
  from the explicit `lon_0`/`lat_0`. The 2-tuple is unpacked before it is fed
  back in, since `pset_to_flowmap` is keyword-only:

  ```python
  lon0, lat0 = seed.to_parcels_pset()
  # ... advect (lon0, lat0) with Parcels, collect (lon1, lat1) ...
  flowmap = seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)
  ```

- **`pset_to_flowmap(*, lon, lat, t0, t1)`** — reattach the flat advected
  positions onto the seed's `particle` index as the advected `lon`/`lat`,
  leaving the reference `lon_0`/`lat_0` (and any auxiliary geometry) untouched,
  and produce the paired concrete `FlowMap`. It records the release time `t0`
  and the derived signed window `T = t1 - t0` as scalar coordinates (`t1` itself
  is not stored). Lost particles arrive as `NaN` and propagate naturally.
  Backward integration is selected purely by passing `t1` before `t0` (negative
  `T`); no separate direction flag exists. A **zero window** (`t1 == t0`) is
  rejected with `ValueError`, since the FTLE's $1/|T|$ would divide by zero.
- **`to_seed()`** — the lossless inverse of `pset_to_flowmap`: drop the advected
  `lon`/`lat` and the scalar `t0`/`T` coords, recovering the paired time-free
  `Seed`. Re-emitting reproduces the same flat particle set. For `Auxiliary*`
  this rebuilds from the carried arms, needing neither the original axes nor
  `aux_separation_m`.

The reusable-template workflow (same grid, sweep `t1`, or re-release at a new
`t0`) is therefore `flowmap.to_seed()` then
`seed.pset_to_flowmap(lon=..., lat=..., t0=..., t1=...)`, with no shared state:
the seed is a spatial template and every release passes its own `(t0, t1)`.

## Operators

The diagnostics live on `FlowMap`; a time-free `Seed` has none (it carries no
advected positions or window).

```text
FlowMap.deformation_gradient() -> xr.DataArray   # abstract (per-stencil)
FlowMap.cauchy_green() -> xr.DataArray            # concrete (base)
FlowMap.cg_eigen() -> xr.Dataset                  # concrete (base)
FlowMap.ftle() -> xr.DataArray                    # concrete (base)
```

- **`deformation_gradient()`** — $\nabla F = \partial(\text{lon},
  \text{lat}) / \partial(\text{lon}_0, \text{lat}_0)$ as advected separations
  (numerator, from the ingested outputs) over reference separations
  (denominator). Each separation is taken in metres in the local east/north
  frame of the two points it connects (see
  [`notation.md`](notation.md#the-local-east-north-frame)). What limits the
  accuracy of a separation is the span of the stencil, not the size or position
  of the domain; the error series is in
  [`numerics.md`](numerics.md#the-local-east-north-frame). Dims `i`, `j`,
  `row`, `col` — a *set*, not an order: the
  package is label-based, so the axis order the call returns is not part of the
  contract (today it is `('row', 'col', 'i', 'j')`, and that may change).
  `row`/`col` are dimension coordinates valued `['x', 'y']` and
  `gradF.sel(row=a, col=b) = dF_a / dx0_b`. There is **no** `comp` coord on the
  tensor. The stencil is per-subclass:
  - `NeighborFlowMap`: central difference against neighbours
    `(i +/- 1, j +/- 1)`; domain-edge cells are legitimately `NaN`.
  - `AuxiliaryFlowMap`: per-point four-arm central difference (east-west,
    north-south over `2s`); defined at every grid point, no `NaN` edges.
- **`cauchy_green()`** — $C = (\nabla F)^\top \nabla F$, symmetric, on the same
  dim set `i`, `j`, `row`, `col` (again in no contractual order — as shipped it
  is `('row', 'i', 'j', 'col')`). Select by label, and `.transpose()` yourself
  if you need a particular layout.
- **`cg_eigen()`** — eigen-decomposition of $C$ via `np.linalg.eigh`. Returns a
  `Dataset` with `lambda` on `(i, j, eig)` (eigenvalues **ascending**,
  $0 < \lambda_1 \le \lambda_2$, `eig = [0, 1]`) and `xi` on
  `(i, j, comp, eig)` (orthonormal eigenvectors, `comp = ['x', 'y']`).
- **`ftle()`** — $\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ using the
  *largest* eigenvalue $\lambda_2$ and the recorded signed window `T`. Dims
  `(i, j)`, units 1/second. Only $|T|$ enters, so backward and forward
  integration of the same map give the same FTLE.

All four are reported at the diagnostic grid points and carry
`lon_grid`/`lat_grid` as their position coords — and only those. The release
positions `lon_0`/`lat_0` are *not* carried through: on an `AuxiliaryFlowMap`
they are one stencil arm, which would mislabel an `(i, j)` quantity.

A single `NaN` (lost particle or missing stencil point) propagates
`gradF -> C -> eigen -> ftle` with no special-casing.

## Hyperbolic LCS: shrink lines

Two module-level functions (in `lcs_parcels.tensorlines`, exported from the
package root) turn a `FlowMap`'s strain field into LCS **curves**, following
Haller (2015) §5.1 / Table 1 ($n = 2$):

```python
from lcs_parcels import ftle_ridge_seeds, shrink_lines

seed_lon, seed_lat = ftle_ridge_seeds(flowmap.ftle())                    # start points
lines = shrink_lines(flowmap, seed_lon=seed_lon, seed_lat=seed_lat)      # curves
```

A repelling LCS is a **shrink line** — a curve tangent to the weak-stretch
eigenvector $\xi_1$ of $C$, solving the tensor-line ODE $\dot r = \xi_1(r)$.
Attracting LCS need no separate call: by the forward–backward duality (Haller &
Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)) they are the shrink
lines of the *backward* flow, so `shrink_lines` of a **forward** `FlowMap` gives
repelling LCS and of a **backward** one gives attracting LCS.

```text
ftle_ridge_seeds(ftle, *, window_m=30_000.0,
                 quantile=0.90) -> tuple[np.ndarray, np.ndarray]
shrink_lines(flowmap, *, seed_lon, seed_lat, min_anisotropy=1.15,
             step_m=3_000.0, line_length_m=1_500_000.0) -> xr.Dataset
```

Every tuning parameter is stated in the units of the thing itself — metres for
the three lengths, a dimensionless eigenvalue ratio for the guard — so the same
call means the same thing at any resolution and over any window:

- **`ftle_ridge_seeds(ftle)`** — start points at strong local maxima of an FTLE
  field: grid points that are the maximum over a square neighbourhood of side
  `window_m` **metres** (a windowed local maximum on the raw value) *and* at or
  above the `quantile` magnitude floor. `window_m` is converted to an odd cell
  count per dimension from the field's own `lon_grid`/`lat_grid` spacing (the
  default 30 km is 7 cells on a $1/25^\circ$ grid at $20^\circ$N). The window
  reaches `window_m / 2` to either side of its own grid point, so two seeds can
  be as close as about `window_m / 2`, not `window_m` — halve it to read off the
  minimum seed spacing. The field must carry `lon_grid`/`lat_grid`; `flowmap.ftle()`
  does. Returns `(lon, lat)` 1-D arrays. NaN cells never qualify.
- **`shrink_lines(flowmap, seed_lon=..., seed_lat=...)`** — integrate the
  $\xi_1$ tensor line through each seed, both directions, on the flow map's
  rectilinear grid. It interpolates the tensor $C$ (via `scipy`'s
  `RegularGridInterpolator`) and re-diagonalises at each step — robust to the
  eigenvector sign ambiguity — and orients each step to the running heading.
  `line_length_m` is a **cap**, not the achieved length: it is traced half in
  each direction, with $n = $ `line_length_m / (2 * step_m)` steps per direction
  (rounded, at least 1), so a line that runs the full budget has $2n + 1$
  points. A line stops early where $\xi_1$ stops being a well-defined direction
  — where $\lambda_2 / \lambda_1$ falls below `min_anisotropy` — or where it
  leaves the grid or hits a NaN cell. Returns an `xr.Dataset` with `lon`/`lat`
  on dims `(line, point)`, one `line` per seed; every row is the same length,
  NaN-filled past termination, and a seed that cannot be traced at all is an
  all-NaN row.

**`min_anisotropy`** is a floor on $\lambda_2 / \lambda_1$, the ratio of the two
Cauchy–Green eigenvalues, and so dimensionless. It is a **well-definedness
guard, not an LCS selector** — `quantile` is what selects. At the default 1.15 a
1% error in $C$ swings $\xi_1$ by about 2 degrees; at a ratio of 1.05 by 6
degrees. Being a ratio it carries no $T$, no grid scale and no stretching rate,
so 1.15 means the same thing for a six-hour laboratory flow and a six-month
basin-scale one. Its useful range starts just above 1: $C$ is positive
semi-definite, so $\lambda_2 \ge \lambda_1 \ge 0$ always, and any
`min_anisotropy` at or below 1 makes the guard unsatisfiable — it never fires,
and lines then stop only by leaving the grid or hitting a NaN cell.

This layer interpolates on the axis-aligned `lon_grid`/`lat_grid` axes, so (like
`NeighborFlowMap`) it assumes a rectilinear flow map, and the `lon_grid` axis
must be monotonic. The traced lines themselves are unconstrained: a line steps
by adding a longitude increment to its current longitude, so it crosses the
antimeridian on its seed's branch.

### One call: `FlowMap.hyperbolic_lcs()`

```text
FlowMap.hyperbolic_lcs(*, window_m=None, quantile=None, min_anisotropy=None,
                       step_m=None, line_length_m=None) -> xr.Dataset
```

Runs the three steps — `ftle()`, `ftle_ridge_seeds`, `shrink_lines` — in one
call, computing the FTLE exactly once. Every parameter is optional and only the
ones actually passed are forwarded, so the defaults stay in
`ftle_ridge_seeds`/`shrink_lines`. There is no direction argument: the flow map
already carries $\mathrm{sign}(T)$, so a forward map yields repelling LCS
and a backward one attracting LCS, and the returned dataset says which it holds.
The name says *hyperbolic* because it extracts hyperbolic (repelling and
attracting) LCS only.

The result is the `shrink_lines` dataset — `lon`/`lat` on `(line, point)` —
with the `ftle` field on `(i, j)` that the seeds were picked from riding along,
so the curves can be plotted over it without recomputing an eigendecomposition:

```python
lcs = forward.hyperbolic_lcs()
lcs["ftle"].plot(x="lon_grid", y="lat_grid")
plt.plot(lcs["lon"].T, lcs["lat"].T)
```

`lon_grid`/`lat_grid` are 2-D non-dimension coords (the layout that lets the
grid be curvilinear), so a gridded field is drawn against them by naming them;
`.plot()` on its own falls back to the logical `i`/`j` axes.

Ridge-finding itself takes a *field*, not a flow map, so to pick ridges from a
smoothed or masked FTLE, drive the three steps by hand.

## Evolving a material curve

An extracted LCS is a **material** curve, so its later positions are fixed by the
flow: $\mathcal{M}(t) = F_{t_0}^{t}(\mathcal{M}(t_0))$ (Haller 2015, Eq. 5).
`FlowMap.image` applies the stored flow map to arbitrary reference points, so
evolving a curve is just interpolating that map at the curve's vertices — no
second advection.

```text
FlowMap.image(*, lon0, lat0) -> xr.Dataset
```

- **`image(*, lon0, lat0)`** — interpolate `grid_image` (the advected positions
  on the diagnostic grid) at reference points `lon0`/`lat0` (`DataArray`s on any
  shared dims, e.g. the `(line, point)` grid of `shrink_lines`), returning their
  advected positions $F_{t_0}^{t_1}(x_0)$ as an `xr.Dataset` with `lon`/`lat` on
  the input dims — the same structure a `shrink_lines` curve has, so an evolved
  curve is drop-in plottable and can itself be re-fed. The requested reference
  positions ride along as `lon_0`/`lat_0` coords on the output. Points off the
  grid, in a NaN (land/edge) cell, or NaN themselves map to NaN. Rectilinear
  grids only, like `shrink_lines`, with a monotonic `lon_grid` axis.

  The advected longitudes may arrive on any branch. Each is re-anchored on the
  branch of the grid point it came from before the interpolation, so an
  advection that hands positions back wrapped to $[-180, 180)$ is read
  correctly. The returned longitudes are on the branch `lon0` was given in.

An LCS is evolved in its **coherent** direction, where perturbations decay: an
attracting LCS forward in time, a repelling one backward. Advecting the grid to a
few horizons and calling `image` at each carries the curve through them —
see [`examples/cabo_verde_lcs_evolution.py`](../examples/cabo_verde_lcs_evolution.py).

## Output metadata

Every array the package returns carries `name`, `long_name`, and `units`, and no
two quantities come back under the same `name`. That metadata *is* the axis
label, the title, and the colorbar caption of a vanilla `.plot()`, so a returned
field needs no hand-set labels to be readable — the only keyword a gridded
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
| `shrink_lines()` | `lon` / `lat` | longitude/latitude along the shrink line | `degrees_east` / `degrees_north` |
| `hyperbolic_lcs()` | `lon` / `lat` | longitude/latitude along the repelling (or attracting) LCS | `degrees_east` / `degrees_north` |

Units are SI: the FTLE is `1/s`, not 1/day. Converting for display is the
reader's call and belongs in the plotting code, where it is visible; re-set
`units` when you do.

The coordinates carry the same treatment, set once at construction
(`from_axes`, `pset_to_flowmap`) and riding unchanged through the diagnostics:

| Coord | `long_name` | `units` |
|---|---|---|
| `i` / `j` | logical grid index along i / j | — |
| `lon_grid` / `lat_grid` | longitude / latitude | `degrees_east` / `degrees_north` |
| `lon_0` / `lat_0` | longitude/latitude of the reference release position x_0 | `degrees_east` / `degrees_north` |
| `displacement` | auxiliary stencil arm | — |
| `t0` | release time t0 | — |
| `T` | signed integration window T = t1 - t0 | — |
| `row` / `col` | tensor row / column index | — |
| `comp` | eigenvector component | — |
| `eig` | Cauchy-Green eigenpair, ascending: 0 is the weak-stretch lambda_1, 1 is lambda_max = lambda_2 | — |
| `line` | shrink line index, one per seed point | — |
| `point` | point index along the shrink line | — |

The index and label coords (`i`, `j`, `displacement`, `row`, `col`, `comp`,
`eig`, `line`, `point`) carry no `units`: their values are logical indices or
string labels, so there is no unit to give. `t0` and `T` carry none either —
they are `datetime64`/`timedelta64`, so the dtype already holds the unit.

## References

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

Haller, G. & Sapsis, T. (2011). *Lagrangian coherent structures and the smallest
finite-time Lyapunov exponent.* Chaos, 21, 023115.
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
