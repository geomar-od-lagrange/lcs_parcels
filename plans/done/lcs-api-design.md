<!--
Plan for the scaffolding PR. This file plans the API; it also drives two doc
deliverables: a notation reference (`docs/notation.md`) and the replacement of
the `docs/api.md` sketch with a real API doc once the code exists.
-->

# LCS-Parcels API design plan

> **Superseded by [`plans/seed-flowmap-design.md`](seed-flowmap-design.md) for
> the class structure and timing.** The single `ParticleGrid` hierarchy
> (`NeighborGrid`/`AuxiliaryGrid`) described below has been split into two
> sibling families: a time-free `Seed` family (`Seed`, `NeighborSeed`,
> `AuxiliarySeed`) and a `FlowMap` family (`FlowMap`, `NeighborFlowMap`,
> `AuxiliaryFlowMap`). The seed carries no time; both `t0` and `t1` enter at
> ingest (`Seed.pset_to_flowmap(lon, lat, *, t0, t1)`). The signed-`T`,
> scalar-at-boundary core is unchanged in substance. Read the seed/flow-map note
> for the current design; the prose below has been corrected for the split but is
> otherwise a historical record.

Status: implemented. The design below is realized in
[`src/lcs_parcels/grids.py`](../src/lcs_parcels/grids.py) with a green test
suite (`tests/`). This document records the design rationale; the live API
reference is [`docs/api.md`](../docs/api.md) and the symbol table is
[`docs/notation.md`](../docs/notation.md).

This plan turns the original sketch into a concrete design. It follows
Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137–162, [doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322),
for naming and notation.

## What this design delivers

The realized package provides:

- class definitions, method **signatures**, and **type hints**;
- **docstrings** stating intent, units, shapes, and the relevant Haller equation;
- the numerical implementation of the operator chain (deformation gradient
  through FTLE);
- **tests** that pin the behavior (shapes, dims, round-trip, analytic cases);
- the notation doc and the API doc.

## Scope

This package is the **diagnostic layer** that sits on top of trajectory
integration. It contains **no Parcels code**: it emits particle sets and ingests
results, nothing more. A `Seed` is time-free; ingest
(`pset_to_flowmap(lon, lat, *, t0, t1)`) is given both the release time $t_0$ and
the end time $t_1$ and derives the signed window $T = t_1 - t_0$, so direction
(forward/backward) is implied by $\operatorname{sign}(T)$ and this package never
needs a separate direction flag. See
[`plans/timing-design.md`](timing-design.md).

We provide:

1. Seeding — build seed positions from a structured grid.
2. Round-trip — emit a particle set, ingest advected lon/lat back onto the grid.
3. Operators — deformation gradient -> Cauchy–Green -> eigen-analysis.
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

> Superseded structure (see [`plans/seed-flowmap-design.md`](seed-flowmap-design.md)):
> the single `ParticleGrid` hierarchy below is now two sibling families. Read
> `Seed`/`NeighborSeed`/`AuxiliarySeed` (time-free, emit) wherever the text says
> `ParticleGrid`/`NeighborGrid`/`AuxiliaryGrid` *before* advection, and
> `FlowMap`/`NeighborFlowMap`/`AuxiliaryFlowMap` (advected, diagnostics) *after*.

```
ParticleGrid (ABC, composition wrapper around .ds)
├── NeighborGrid     # stencil = neighboring grid points (i +/- 1, j +/- 1)
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
- `.ds`: `xr.Dataset` with logical dims `i, j`. A `Seed` carries only the
  reference release positions; the advected positions appear on the `FlowMap`
  produced at ingest (2D lon/lat, so curvilinear/non-rectangular grids work):
  - reference coordinates `lon_0(i, j)`, `lat_0(i, j)` — the release positions
    $x_0$ (for `NeighborSeed` the grid points themselves; `AuxiliarySeed`
    elaborates these onto a `displacement` arm dim and keeps the diagnostic
    centres as a separate `lon_c`/`lat_c` pair — see below). These live on both
    the seed and the flow map.
  - advected data variables `lon(i, j)`, `lat(i, j)` — the flow map
    $F_{t_0}^{t_1}(x_0)$ (the particle's actual position, as in Parcels). These
    exist **only on the `FlowMap`**, set at ingest. A `Seed` has no advected
    positions and no identity-seeded `lon`/`lat`; the old $F_{t_0}^{t_0}(x_0) = x_0$
    fiction is gone.
- The release time `t0` (`datetime64`) and the signed window `T` (`timedelta64`)
  live **only on the `FlowMap`**, both derived at ingest from the `(t0, t1)`
  arguments — the seed is time-free; see
  [`plans/timing-design.md`](timing-design.md).
- Resolution metadata (`dlon`, `dlat`) as coords/attrs.

### `NeighborSeed` / `NeighborFlowMap`
- No extra dims. The deformation gradient is differenced against neighboring
  grid points `(i +/- 1, j +/- 1)`.

### `AuxiliarySeed` / `AuxiliaryFlowMap`
- Adds a *fixed* four-arm displacement stencil on a single `displacement` dim
  (`displacement = ['east', 'north', 'west', 'south']`) placed at $\pm s$ meters
  about each centre (`aux_separation_m`) — Haller's auxiliary grid (Eq. 9). The
  arms are stored **explicitly** as the reference release positions
  `lon_0(i, j, displacement)`, `lat_0(i, j, displacement)` (so the dataset is
  self-sufficient — no metric convention is needed to recover where particles
  started), and the advected `lon`/`lat` share those dims. The grid-point
  **centres** `lon_c(i, j)`, `lat_c(i, j)` — where diagnostics are reported and
  the anchor for downstream LCS work — are kept separately. No center arm (it
  would duplicate the grid position) and no diagonal corners (unused by central
  differencing); the shape is enforced in `from_axes`, not left arbitrary.
  Decouples diagnostic resolution from the gradient step.

## Round-trip (emit / ingest)

Kept deliberately minimal:

- `Seed.to_parcels_pset() -> tuple[list, list]` — flatten the **reference**
  positions $x_0$ (`lon_0`, `lat_0`) to plain lists for Parcels. Internally
  `.stack(particle=('i','j'))` or `(...,'displacement')`; the `particle`
  MultiIndex is the lossless inverse.
- `Seed.pset_to_flowmap(lon, lat, *, t0, t1) -> FlowMap` — ingest factory
  (a behaviour of the seed) that reattaches advected positions to the `particle`
  coord as the flow map `lon`/`lat` (leaving the reference `lon_0`/`lat_0`
  untouched), `.unstack()`s back to the grid, and records `t0` and the derived
  signed window `T = t1 - t0` (`timedelta64`, needed for FTLE) on the returned
  `FlowMap`. **Both** the release time `t0` and the end time `t1` (`datetime64`)
  are passed here — the seed is time-free. `FlowMap.to_seed()` collapses an
  advected flow map back to a time-free seed. See
  [`plans/timing-design.md`](timing-design.md).

A single `FlowMap` carries scalar `t0`/`T`. A release series (sweep `t0`) or a
window sweep (`T`) is an **external loop**: each release is emitted, advected
independently, and ingested via `pset_to_flowmap(..., t0=, t1=)` to a
scalar-`(t0, T)` `FlowMap`, then the per-run flow maps are assembled into the
`(i, j, t0, T)` cube with `xr.concat`/`combine_by_coords` (self-aligning, since
each carries its own scalar `t0`/`T`). The time-free seed cannot carry a `t0`
dimension, so this is no longer "extra broadcast dimensions on the seed". Particle
loss arrives as NaN and propagates naturally; cells with a missing stencil point
yield NaN $\nabla F$.

## Operators (methods)

- `deformation_gradient() -> DataArray` — `$\nabla F$`, dims `(i, j, row, col)`.
- `cauchy_green() -> DataArray` — `$C = (\nabla F)^\top \nabla F$`, same dims.
- `cg_eigen() -> Dataset` — eigenvalues $\lambda$ on `(i, j, eig)` and
  eigenvectors $\xi$ on `(i, j, comp, eig)`, via
  `xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row','col']], ...)`.
- `ftle() -> DataArray` — `$\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$`.
- (optional) generalized Green–Lagrange `$E_\lambda$` (Eq. 8).

## Tensor representation

$2\times 2$ tensors and 2-vectors are **single DataArrays with component dims**, not
scalar vars (`F11, F12, …`):
- `gradF`, `C`: dims `(i, j, row, col)`, with `row, col = ['x', 'y']` (dimension
  coordinates indexing the two tensor axes). The tensors carry **no** `comp`
  coord — `comp` is reserved for the eigenvector component dim.
- eigen output: $\lambda$ on `(i, j, eig)`, $\xi$ on `(i, j, comp, eig)` with
  `comp = ['x', 'y']`.

This keeps the eigen step a vectorized one-liner.

## Sphere metric (note, not a blocker)

Haller's math is Cartesian; our grid is lon/lat. Separations are formed in a
**single** local tangent frame in **meters**, anchored at the grid centroid
($dx = R\cos\phi_{\mathrm{ref}}\,d\lambda$, $dy = R\,d\phi$; $\lambda$ =
longitude, $\phi$ = latitude). The cosine factor uses the *one* grid reference
latitude $\phi_{\mathrm{ref}} = \overline{\phi_0}$ for every point, **not** a
per-point $\cos\phi$: a single shared frame is what makes the off-diagonal
$\nabla F$ terms exact (a per-point cosine corrupts them by a cosine ratio). In
the tiny-separation regime this flat-tangent approximation is adequate; it is a
convention, not a correctness blocker. The metric only converts lon/lat
separations to meters: the *advected* separations forming the numerator of
$\nabla F$ are measured from the ingested Parcels outputs, not recomputed
analytically from $R$ and $\phi$.

## v1 scope vs later

**v1 (implemented):** seed -> deformation gradient -> $C$ -> eigen -> FTLE and
eigen-derived scalar fields (signatures, types, docstrings, tests, and the
numerical implementation). Both stencils (`Neighbor*` and `Auxiliary*`, across
the `Seed`/`FlowMap` families) are first-class.

**Deferred (separate module):** the geometric LCS layer — shrink/stretch/shear
lines, elliptic (vortex) LCS via closed shear lines (Eqs. 10–11, Table 1). This
is tensor-line ODE integration with eigenvector orientation/desingularization, a
different problem from pointwise grid algebra.

## Resolved review questions

1. Differentiation modes — **both first-class**, no default; modeled as the two
   stencils (`Neighbor*`, `Auxiliary*`) on each of the `Seed`/`FlowMap`
   families.
2. Metric — **meters-based** local tangent frame; not critical (tiny-separation
   regime).
3. v1 scope — **stop at FTLE + eigen fields**; defer tensor-line / elliptic LCS.
4. Tensor storage — **component-dim DataArrays**.
5. Where operators live — **composition wrapper classes**, not `xr.Dataset`
   subclassing and not an accessor.
6. Timing input — the seed is **time-free**; ingest
   (`pset_to_flowmap(lon, lat, *, t0, t1)`) takes **both** `t0` and `t1` and
   derives signed `T = t1 - t0` on the `FlowMap`. Direction is `sign(T)`. See
   [`plans/timing-design.md`](timing-design.md).
7. Auxiliary stencil — **fixed four arms** `['east', 'north', 'west', 'south']`
   on one `displacement` dim, enforced in `from_axes` (no center, no diagonals).
8. Deformation gradient — the numerator (advected separations) is taken from the
   **ingested outputs**; the metric only converts the initial separations
   (denominator) to meters.
