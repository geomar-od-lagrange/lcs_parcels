<!--
API reference for the seed / flow-map diagnostic layer in
src/lcs_parcels/grids.py. Kept in sync with the code; symbols are defined in
docs/notation.md and the workflow/diagrams in docs/architecture.md.
-->

# API reference: seeds and flow maps

The public surface of `lcs_parcels` is two sibling families — a time-free
`Seed` family and a `FlowMap` family — each with a shared abstract base and two
concrete stencils, exported from the package root:

```python
from lcs_parcels import Seed, NeighborSeed, AuxiliarySeed, FlowMap, NeighborFlowMap, AuxiliaryFlowMap
```

Plus two functions that turn a `FlowMap`'s strain field into hyperbolic-LCS
curves — `ftle_ridge_seeds` and `shrink_lines` (see
[Hyperbolic LCS: shrink lines](#hyperbolic-lcs-shrink-lines) below).

A `Seed` lays out reference positions and emits a particle set for Parcels; the
advected positions are ingested back into a `FlowMap`, which computes the
deformation gradient $\nabla F$ and everything downstream (Cauchy–Green $C$, its
eigen-decomposition, and the FTLE). The two families are **siblings, not an
inheritance pair**: a `FlowMap` is not a kind of `Seed` (it emits nothing to
Parcels) and a `Seed` is not a kind of `FlowMap` (it has no advected positions
or window). Shared computation lives in module-level helpers, not a common base.
Symbols and units are defined in [`notation.md`](notation.md); the class diagram
and session flow are in [`architecture.md`](architecture.md). Naming follows
Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

The package contains **no Parcels code**: a `Seed` emits particle sets and
ingests advected positions; Parcels (external) owns the integration.

## Data model

Each object wraps an `xr.Dataset` held in `.ds` (composition, *not* an
`xr.Dataset` subclass), with logical grid dims `i, j`. A **`Seed` is time-free
and all-coordinates** (no data variables): it holds only the reference release
positions $x_0$, plus the auxiliary arm geometry and centres for
`AuxiliarySeed`. A **`FlowMap` adds the advected positions** as its only data
variables, plus scalar `t0`/`T` coordinates.

| Name | Role | Dims | Kind | On |
|---|---|---|---|---|
| `lon_0`, `lat_0` | reference *release* positions $x_0$ | `(i, j)` (see note) | coords | seed + flow map |
| `lon_c`, `lat_c` | grid-point *centres* (diagnostics reported here) | `(i, j)` | coords | `Auxiliary*` only |
| `lon`, `lat` | advected positions, the flow map $F_{t_0}^{t_1}(x_0)$ | same dims as `lon_0`/`lat_0` | data vars | flow map only |
| `t0` | release time | scalar | coord | flow map only |
| `T` | signed integration window $T = t_1 - t_0$ (`timedelta64`) | scalar | coord | flow map only |

Note: for the `Auxiliary*` classes the release positions are the four stencil
arms, so `lon_0`/`lat_0` (and the advected `lon`/`lat` on a flow map) all carry
the extra `displacement` dim, i.e. `(i, j, displacement)` — the arms are stored
*explicitly*, so the dataset is self-sufficient (no metric needed to recover
them). The grid-point centres on which diagnostics are reported are kept as a
separate pair `lon_c`/`lat_c` on `(i, j)`. A single `FlowMap` carries `t0`/`T`
as *scalar* coords; `t1` is not stored, being recoverable as `t0 + T`. See the
*Position convention* in the module docstring.

## Constructors and round-trip

A seed is built, emitted, ingested into a flow map, and (optionally) collapsed
back to a seed. The two families are connected by two crossings; neither holds a
reference to the other.

```python
Seed.from_axes(lon, lat) -> Self                          # classmethod (abstract)
Seed.to_parcels_pset() -> tuple[list, list]               # concrete (base)
Seed.pset_to_flowmap(lon, lat, *, t0, t1) -> FlowMap      # concrete (base)
FlowMap.to_seed() -> Seed                                 # concrete (base)
```

- **`from_axes(lon, lat)`** — build a time-free seed from 1-D lon/lat axes
  (length `Ni`, `Nj`), broadcast into curvilinear 2-D reference fields on
  `(i, j)`. No time is recorded — the seed is time-free; `t0` and the window `T`
  enter only at `pset_to_flowmap`. `AuxiliarySeed.from_axes` also takes a
  keyword-only `aux_separation_m` (the controlled arm separation $s$ in meters;
  default `1000.0`), lays out the fixed four-arm
  `displacement = ['east', 'north', 'west', 'south']` stencil at $\pm s$ about
  each centre, and stores those arms explicitly as `lon_0`/`lat_0` on
  `(i, j, displacement)` plus the centres `lon_c`/`lat_c` on `(i, j)`.
- **`to_parcels_pset()`** — flatten the *reference* release positions to plain
  `(lon, lat)` lists (a 2-tuple) over the `particle` index (`('i', 'j')`, plus
  `'displacement'` for `AuxiliarySeed`). The auxiliary arms are emitted directly
  from the explicit `lon_0`/`lat_0`.
- **`pset_to_flowmap(lon, lat, *, t0, t1)`** — reattach the flat advected
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
`seed.pset_to_flowmap(..., t0=..., t1=...)`, with no shared state: the seed is a
spatial template and every release passes its own `(t0, t1)`.

## Operators

The diagnostics live on `FlowMap`; a time-free `Seed` has none (it carries no
advected positions or window).

```python
FlowMap.deformation_gradient() -> xr.DataArray   # abstract (per-stencil)
FlowMap.cauchy_green() -> xr.DataArray            # concrete (base)
FlowMap.cg_eigen() -> xr.Dataset                  # concrete (base)
FlowMap.ftle() -> xr.DataArray                    # concrete (base)
```

- **`deformation_gradient()`** — $\nabla F = \partial(\text{lon},
  \text{lat}) / \partial(\text{lon}_0, \text{lat}_0)$ as advected separations
  (numerator, from the ingested outputs) over reference separations
  (denominator), both taken in the shared single-reference-latitude meters frame
  (see *Sphere metric convention*). Dims `(i, j, row, col)` with `row`/`col`
  dimension coordinates valued `['x', 'y']` and
  `gradF.sel(row=a, col=b) = dF_a / dx0_b`. There is **no** `comp` coord on the
  tensor. The stencil is per-subclass:
  - `NeighborFlowMap`: central difference against neighbours
    `(i +/- 1, j +/- 1)`; domain-edge cells are legitimately `NaN`.
  - `AuxiliaryFlowMap`: per-point four-arm central difference (east-west,
    north-south over `2s`); defined at every grid point, no `NaN` edges.
- **`cauchy_green()`** — $C = (\nabla F)^\top \nabla F$, symmetric, same dims
  `(i, j, row, col)`.
- **`cg_eigen()`** — eigen-decomposition of $C$ via `np.linalg.eigh`. Returns a
  `Dataset` with `lambda` on `(i, j, eig)` (eigenvalues **ascending**,
  $0 < \lambda_1 \le \lambda_2$, `eig = [0, 1]`) and `xi` on
  `(i, j, comp, eig)` (orthonormal eigenvectors, `comp = ['x', 'y']`).
- **`ftle()`** — $\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ using the
  *largest* eigenvalue $\lambda_2$ and the recorded signed window `T`. Dims
  `(i, j)`, units 1/second. Only $|T|$ enters, so backward and forward
  integration of the same map give the same FTLE.

A single `NaN` (lost particle or missing stencil point) propagates
`gradF -> C -> eigen -> ftle` with no special-casing.

## Hyperbolic LCS: shrink lines

Two module-level functions (in `lcs_parcels.tensorlines`, exported from the
package root) turn a `FlowMap`'s strain field into LCS **curves**, following
Haller (2015) §5.1 / Table 1 ($n = 2$):

```python
from lcs_parcels import ftle_ridge_seeds, shrink_lines

seed_lon, seed_lat = ftle_ridge_seeds(flowmap.ftle())     # start points
lines = shrink_lines(flowmap, seed_lon, seed_lat)          # xr.Dataset of curves
```

A repelling LCS is a **shrink line** — a curve tangent to the weak-stretch
eigenvector $\xi_1$ of $C$, solving the tensor-line ODE $\dot r = \xi_1(r)$.
Attracting LCS need no separate call: by the forward–backward duality (Haller &
Sapsis 2011) they are the shrink lines of the *backward* flow, so `shrink_lines`
of a **forward** `FlowMap` gives repelling LCS and of a **backward** one gives
attracting LCS.

```python
ftle_ridge_seeds(ftle, *, window=7, quantile=0.90) -> tuple[np.ndarray, np.ndarray]
shrink_lines(flowmap, seed_lon, seed_lat, *, lambda_max_min=1.1,
             step_m=3000.0, n_steps=250) -> xr.Dataset
```

- **`ftle_ridge_seeds(ftle)`** — start points at strong local maxima of an FTLE
  field: grid points that are the maximum over a `window x window` neighbourhood
  (a windowed local maximum on the raw value) *and* at or above the `quantile`
  magnitude floor. Returns `(lon, lat)` 1-D arrays. NaN cells never qualify.
- **`shrink_lines(flowmap, seed_lon, seed_lat)`** — integrate the $\xi_1$ tensor
  line through each seed, both directions, on the flow map's rectilinear grid.
  It interpolates the tensor $C$ (via `scipy`'s `RegularGridInterpolator`) and
  re-diagonalises at each step — robust to the eigenvector sign ambiguity — and
  orients each step to the running heading. A line stops where $\lambda_2 <$
  `lambda_max_min` (a low guard against the rare degenerate $\lambda_1 \approx
  \lambda_2$ points, *not* an LCS selector), or where it leaves the grid / hits a
  NaN cell; a seed that cannot be traced yields an all-NaN line. Returns an
  `xr.Dataset` with `lon`/`lat` on dims `(line, point)` — one `line` per seed,
  fixed `2 * n_steps + 1` points, NaN past termination.

This layer interpolates on the axis-aligned `lon_0`/`lat_0` grid, so (like
`NeighborFlowMap`) it assumes a rectilinear flow map. The tight ODE loop is the
one place the package leaves the label-based xarray API for NumPy/SciPy.

## Reference

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
