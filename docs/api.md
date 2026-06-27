<!--
API reference for the particle-grid diagnostic layer in
src/lcs_parcels/grids.py. Kept in sync with the code; symbols are defined in
docs/notation.md and the workflow/diagrams in docs/architecture.md.
-->

# API reference: particle grids

The public surface of `lcs_parcels` is two concrete grid classes and their
shared abstract base, exported from the package root:

```python
from lcs_parcels import ParticleGrid, NeighborGrid, AuxiliaryGrid
```

Both concrete classes implement the same workflow: seed a grid, emit a particle
set for Parcels, ingest the advected positions, then compute the deformation
gradient $\nabla F$ and everything downstream (Cauchy–Green $C$, its
eigen-decomposition, and the FTLE). Symbols and units are defined in
[`notation.md`](notation.md); the class diagram and session flow are in
[`architecture.md`](architecture.md). Naming follows Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

The package contains **no Parcels code**: it emits particle sets and ingests
advected positions; Parcels (external) owns the integration.

## Data model

Each grid wraps an `xr.Dataset` held in `.ds` (composition, *not* an
`xr.Dataset` subclass). The dataset has logical grid dims `i, j` and carries:

| Name | Role | Dims | Kind |
|---|---|---|---|
| `lon_0`, `lat_0` | reference *release* positions $x_0$ | `(i, j)` (see note) | coords |
| `lon`, `lat` | advected positions, the flow map $F_{t_0}^{t_1}(x_0)$ | same dims as `lon_0`/`lat_0` | data vars |
| `t0` | release time, recorded at seeding | scalar/ensemble | coord |
| `T` | signed integration window $T = t_1 - t_0$ (`timedelta64`; 0 on a seed) | scalar/ensemble | coord |

Note: for `AuxiliaryGrid` the release positions are the four stencil arms, so
`lon_0`/`lat_0` and the advected `lon`/`lat` all carry the extra `displacement`
dim, i.e. `(i, j, displacement)` — the arms are stored *explicitly*, so the
dataset is self-sufficient (no metric needed to recover them). The grid-point
centres on which diagnostics are reported are kept as a separate pair
`lon_c`/`lat_c` on `(i, j)` (coords). See the *Position convention* in the module
docstring.

## Constructors and round-trip

```python
ParticleGrid.from_axes(lon, lat, *, t0) -> Self            # classmethod (abstract)
ParticleGrid.to_parcels_pset() -> tuple[list, list]        # abstract
ParticleGrid.from_parcels_pset_lon_lat(seed, lon, lat, *, t1) -> Self  # classmethod (abstract)
```

- **`from_axes(lon, lat, *, t0)`** — build a seed grid from 1-D lon/lat axes
  (length `Ni`, `Nj`), broadcast into curvilinear 2-D reference fields on
  `(i, j)`. The advected `lon`/`lat` are seeded equal to the reference (the
  identity $F_{t_0}^{t_0}(x_0) = x_0$) and `T = 0`. The grid *owns* its release
  time `t0`. `AuxiliaryGrid.from_axes` also takes `aux_separation_m` (the
  controlled arm separation $s$ in meters; default `1000.0`), lays out the fixed
  four-arm `displacement = ['east', 'north', 'west', 'south']` stencil at
  $\pm s$ about each centre, and stores those arms explicitly as
  `lon_0`/`lat_0` on `(i, j, displacement)` plus the centres `lon_c`/`lat_c` on
  `(i, j)`.
- **`to_parcels_pset()`** — flatten the *reference* release positions to plain
  `(lon, lat)` lists over the `particle` index (`('i', 'j')`, plus
  `'displacement'` for `AuxiliaryGrid`). `AuxiliaryGrid` emits its explicit
  `lon_0`/`lat_0` arms directly, so it is trivially robust on an already-ingested
  grid (the reference positions are never overwritten).
- **`from_parcels_pset_lon_lat(seed, lon, lat, *, t1)`** — reattach the flat
  advected positions onto the seed's `particle` index as the advected
  `lon`/`lat`, leaving the reference `lon_0`/`lat_0` untouched, and record the
  derived signed window `T = t1 - seed.t0`. Only `t1` is passed; the seed owns
  `t0`. Lost particles arrive as `NaN` and propagate naturally. Backward
  integration is selected purely by passing `t1` before `t0` (negative `T`); no
  separate direction flag exists.

## Operators

```python
ParticleGrid.deformation_gradient() -> xr.DataArray   # abstract (per-stencil)
ParticleGrid.cauchy_green() -> xr.DataArray           # concrete (base)
ParticleGrid.cg_eigen() -> xr.Dataset                 # concrete (base)
ParticleGrid.ftle() -> xr.DataArray                   # concrete (base)
```

- **`deformation_gradient()`** — $\nabla F = \partial(\text{lon},
  \text{lat}) / \partial(\text{lon}_0, \text{lat}_0)$ as advected separations
  (numerator, from the ingested outputs) over reference separations
  (denominator), both taken in the shared single-reference-latitude meters frame
  (see *Sphere metric convention*). Dims `(i, j, row, col)` with `row`/`col`
  dimension coordinates valued `['x', 'y']` and
  `gradF.sel(row=a, col=b) = dF_a / dx0_b`. There is **no** `comp` coord on the
  tensor. The stencil is per-subclass:
  - `NeighborGrid`: central difference against neighbours `(i +/- 1, j +/- 1)`;
    domain-edge cells are legitimately `NaN`.
  - `AuxiliaryGrid`: per-point four-arm central difference (east-west,
    north-south over `2s`); defined at every grid point, no `NaN` edges.
- **`cauchy_green()`** — $C = (\nabla F)^\top \nabla F$, symmetric, same dims
  `(i, j, row, col)`.
- **`cg_eigen()`** — eigen-decomposition of $C$ via `np.linalg.eigh`. Returns a
  `Dataset` with `lambda` on `(i, j, eig)` (eigenvalues **ascending**,
  $0 < \lambda_1 \le \lambda_2$, `eig = [0, 1]`) and `xi` on
  `(i, j, comp, eig)` (orthonormal eigenvectors, `comp = ['x', 'y']`).
- **`ftle()`** — $\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ using the
  *largest* eigenvalue $\lambda_2$. Dims `(i, j)`, units 1/second. Only $|T|$
  enters, so backward and forward integration of the same map give the same
  FTLE.

A single `NaN` (lost particle or missing stencil point) propagates
`gradF -> C -> eigen -> ftle` with no special-casing.

## Reference

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
</content>
