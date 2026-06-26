<!--
Plan for the scaffolding PR. This file plans the API; it also drives two doc
deliverables: a notation reference (`docs/notation.md`) and the replacement of
the `docs/api.md` sketch with a real API doc once the code exists.
-->

# LCS-Parcels API design plan

Status: draft for review.

This plan turns the sketch in `docs/api.md` into a concrete design. It follows
Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137–162, [doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322),
for naming and notation.

## What this PR delivers (pedagogical scaffolding)

This PR is scaffolding, built to be read and then filled in during a human pair
programming session. It provides:

- class definitions, method **signatures**, and **type hints**;
- **docstrings** stating intent, units, shapes, and the relevant Haller equation;
- **tests** that pin the intended behavior (shapes, dims, round-trip, simple
  analytic cases);
- the notation doc.

It deliberately does **not** provide the numerical implementation. Method bodies
are placeholders (`raise NotImplementedError`). Tests are real and **left red**:
no `xfail`, no `skip`. A true red suite is the intended starting point for the
human implementation session.

## Scope

This package is the **diagnostic layer** that sits on top of trajectory
integration. It contains **no Parcels code**: it emits particle sets and ingests
results, nothing more. A seed grid owns its release time `$t_0$`; ingest is given
the end time `$t_1$` and derives the signed window `$T = t_1 - t_0$`, so
direction (forward/backward) is implied by `$\operatorname{sign}(T)$` and this
package never needs a separate direction flag. See
[`plans/timing-design.md`](timing-design.md).

We provide:

1. Seeding — build seed positions from a structured grid.
2. Round-trip — emit a particle set, ingest advected lon/lat back onto the grid.
3. Operators — deformation gradient → Cauchy–Green → eigen-analysis.
4. Diagnostics — FTLE and eigen-derived scalar fields.

## Notation (Haller 2015)

Full reference lives in `docs/notation.md`; summary here.

| Symbol | Meaning | Eq. |
|---|---|---|
| $v(x, t)$ | velocity field, $x = (x^1, x^2)$ in 2D | 2 |
| $F_{t_0}^{t}(x_0) = x(t; t_0, x_0)$ | flow map: initial position $\to$ position at time $t$ | 3 |
| $\nabla F_{t_0}^{t_1}(x_0)$ | deformation gradient ($2\times2$ in 2D) | 4, 9 |
| $C(x_0) = (\nabla F)^\top \nabla F$ | right Cauchy–Green strain tensor | 6 |
| $C\,\xi_i = \lambda_i\,\xi_i,\ 0 < \lambda_1 \le \lambda_2,\ \xi_1 \perp \xi_2$ | eigen-decomposition of $C$ | 7 |
| $\Lambda = \tfrac{1}{t_1 - t_0}\log\sqrt{\lambda_{\max}}$ | FTLE (uses the **largest** eigenvalue) | §4.1 |
| $\eta^\pm$, shrink/stretch/shear lines | geometric LCS from $\xi_{1,2}$ | 10, 11, Table 1 |

Naming consequence: the sketch's "F ($2\times2$)" is really $\nabla F$. We reserve
`F` / `flow_map` for the map itself (a 2-component vector field) and
`deformation_gradient` / `gradF` for the $2\times2$ tensor.

## Data structures

The two differentiation methods are modeled as **two explicit classes**, not as
runtime inference from dims (no `"displacement" in ds.dims` sniffing). Both are
first-class — neither is a default or a fallback.

Each class is a **composition wrapper** around an `xr.Dataset` (it holds a `.ds`;
it does not subclass `xr.Dataset`, which xarray discourages). Operators are
methods on these classes. All internal access uses the high-level, label-based
xarray API (`.isel`, `.sel`, named dims), never positional indexing.

```
ParticleGrid (ABC, composition wrapper around .ds)
├── NeighborGrid     # stencil = neighboring grid points (i±1, j±1)
└── AuxiliaryGrid    # stencil = per-point displacement grid (Haller Eq. 9)
```

> Note: `NeighborGrid` **is** the SPASSO / d'Ovidio approach (confirmed against
> the [SPASSO source](https://github.com/OceanCruises/SPASSO), `src/Diagnostics.py`):
> particles are seeded once on a single regular grid at spacing `delta0` (one
> per output cell), and FTLE is computed by neighbour-differencing the
> final-position maps with `np.gradient` over that same grid — no auxiliary
> stencil. `AuxiliaryGrid` is the Haller (Eq. 9) auxiliary-grid alternative.
> Their naming worth echoing: `delta0` (grid spacing), final vs initial
> separation, integration window in days.

### Common state (base `ParticleGrid`)
- `.ds`: `xr.Dataset` with logical dims `i, j` and data vars `lon(i, j)`,
  `lat(i, j)` (2D lon/lat, so curvilinear/non-rectangular grids work). These are
  the $x_0$ on which diagnostics are defined.
- Release time `t0` (`datetime64`), recorded at seeding so the grid owns its own
  `t0`; see [`plans/timing-design.md`](timing-design.md).
- Resolution metadata (`dlon`, `dlat`) as coords/attrs.

### `NeighborGrid`
- No extra dims. The deformation gradient is differenced against neighboring
  grid points `(i±1, j±1)`.

### `AuxiliaryGrid`
- Adds a *fixed* four-arm displacement stencil on a single `displacement` dim
  (`displacement = ['east', 'north', 'west', 'south']`) with offsets
  `dx(displacement)`, `dy(displacement)` in **meters** — Haller's auxiliary grid
  (Eq. 9). No center point (it would duplicate the grid position) and no diagonal
  corners (unused by central differencing); the shape is enforced in `from_axes`,
  not left arbitrary. Decouples diagnostic resolution from the gradient step.

## Round-trip (emit / ingest)

Kept deliberately minimal:

- `to_parcels_pset() -> tuple[list, list]` — flatten to plain `(lon, lat)` lists
  for Parcels. Internally `.stack(particle=('i','j'))` or
  `(...,'displacement')`; the `particle` MultiIndex is the lossless inverse.
- `from_parcels_pset_lon_lat(seed, lon, lat, *, t1) -> ParticleGrid` — factory
  that reattaches advected positions to the `particle` coord, `.unstack()`s back
  to the grid, and records `t0` (from the seed) and the derived signed window
  `T = t1 - t0` (`timedelta64`, needed for FTLE). Only the end time `t1`
  (`datetime64`) is passed; the seed owns `t0`. See
  [`plans/timing-design.md`](timing-design.md).

Multiple release times `t0` and multiple integration times `T` are just extra
broadcast dimensions — handled by xarray, no special machinery. Particle loss
arrives as NaN and propagates naturally; cells with a missing stencil point
yield NaN `$\nabla F$`.

## Operators (methods)

- `deformation_gradient() -> DataArray` — `$\nabla F$`, dims `(i, j, row, col)`.
- `cauchy_green() -> DataArray` — `$C = (\nabla F)^\top \nabla F$`, same dims.
- `cg_eigen() -> Dataset` — eigenvalues `λ` on `(i, j, eig)` and eigenvectors
  `ξ` on `(i, j, comp, eig)`, via
  `xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row','col']], ...)`.
- `ftle() -> DataArray` — `$\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$`.
- (optional) generalized Green–Lagrange `$E_\lambda$` (Eq. 8).

## Tensor representation

2×2 tensors and 2-vectors are **single DataArrays with component dims**, not
scalar vars (`F11, F12, …`):
- `gradF`, `C`: dims `(i, j, row, col)`, labeled coord `comp = ['x', 'y']`.
- eigen output: `λ` on `(i, j, eig)`, `ξ` on `(i, j, comp, eig)`.

This keeps the eigen step a vectorized one-liner.

## Sphere metric (note, not a blocker)

Haller's math is Cartesian; our grid is lon/lat. Separations are formed in a
local tangent frame in **meters** ($dx = R\cos\phi\,d\lambda$, $dy = R\,d\phi$;
$\lambda$ = longitude, $\phi$ = latitude). Because we operate in the
tiny-separation regime, a flat-tangent `$\cos\phi$` approximation is adequate in
practice; this is a convention, not a correctness blocker. The metric only
converts lon/lat separations to meters: the *advected* separations forming the
numerator of $\nabla F$ are measured from the ingested Parcels outputs, not
recomputed analytically from $R$ and $\phi$.

## v1 scope vs later

**v1 (this PR):** seed → deformation gradient → `$C$` → eigen → FTLE and
eigen-derived scalar fields, as scaffolding (signatures, types, docstrings,
tests; no implementation). Both `NeighborGrid` and `AuxiliaryGrid` are
first-class.

**Deferred (separate module):** the geometric LCS layer — shrink/stretch/shear
lines, elliptic (vortex) LCS via closed shear lines (Eqs. 10–11, Table 1). This
is tensor-line ODE integration with eigenvector orientation/desingularization, a
different problem from pointwise grid algebra.

## Resolved review questions

1. Differentiation modes — **both first-class**, no default; modeled as two
   classes (`NeighborGrid`, `AuxiliaryGrid`).
2. Metric — **meters-based** local tangent frame; not critical (tiny-separation
   regime).
3. v1 scope — **stop at FTLE + eigen fields**; defer tensor-line / elliptic LCS.
4. Tensor storage — **component-dim DataArrays**.
5. Where operators live — **composition wrapper classes**, not `xr.Dataset`
   subclassing and not an accessor.
6. Timing input — the seed **owns `t0`** (set in `from_axes`); ingest takes `t1`
   and derives signed `T = t1 - t0`. Direction is `sign(T)`. See
   [`plans/timing-design.md`](timing-design.md).
7. Auxiliary stencil — **fixed four arms** `['east', 'north', 'west', 'south']`
   on one `displacement` dim, enforced in `from_axes` (no center, no diagonals).
8. Deformation gradient — the numerator (advected separations) is taken from the
   **ingested outputs**; the metric only converts the initial separations
   (denominator) to meters.
