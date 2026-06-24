# LCS-Parcels API design plan

Status: draft for review.

This plan turns the sketch in `docs/api.md` into a concrete design. It follows
Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech. 47:137–162
for naming and notation.

## Scope

This package is the **diagnostic layer** that sits *on top of* trajectory
integration. Parcels does the advection; we provide:

1. Seeding (build seed positions from a structured grid).
2. Reshaping advected positions back onto that grid (the "roundtrip").
3. Operators: flow map gradient → Cauchy–Green → eigen-analysis.
4. Diagnostics: FTLE and eigen-derived scalar fields.

Parcels advection itself is treated as a thin adapter at the boundary, not a
core responsibility.

## Notation (Haller 2015)

| Symbol | Meaning | Eq. |
|---|---|---|
| `v(x, t)` | velocity field, `x = (x¹, x²)` in 2D | 2 |
| `F_{t0}^{t}(x0) = x(t; t0, x0)` | flow map: initial position → position at time `t` | 3 |
| `∇F_{t0}^{t1}(x0)` | deformation gradient (2×2 in 2D) | 4, 9 |
| `C(x0) = (∇F)ᵀ ∇F` | right Cauchy–Green strain tensor | 6 |
| `C ξᵢ = λᵢ ξᵢ`, `0 < λ₁ ≤ λ₂`, `ξ₁ ⟂ ξ₂` | eigen-decomposition of `C` | 7 |
| `Λ = (1/(t1−t0)) log √λ_max` | FTLE (uses the **largest** eigenvalue) | §4.1 |
| `η±`, shrink/stretch/shear lines | geometric LCS from `ξ₁,₂` | 10, 11, Table 1 |

Naming consequence: the sketch's "F (2×2)" is really `∇F`. We reserve `F` /
`flow_map` for the map itself (a 2-component vector field) and
`deformation_gradient` / `gradF` for the 2×2 tensor.

## Data structures

All structures are thin conventions over `xarray` objects, not new classes where
a Dataset will do.

### Particle grid
- `xr.Dataset` with logical dims `i, j` and data vars `lon(i, j)`, `lat(i, j)`.
- Curvilinear-capable (lon/lat are 2D), so non-rectangular grids work.
- Carries resolution metadata (`dlon`, `dlat`) as attrs/coords.
- This is the set of `x0` on which FTLE, `C`, eigen-fields are defined.

### Displacement grid (optional) — the differentiation stencil
- Dims `(di, dj)` with displacements `dx(di, dj)`, `dy(di, dj)` **in meters**.
- This *is* Haller's auxiliary-grid finite-difference stencil (Eq. 9): each
  `(i, j)` point gets its own tight stencil at controlled separation δ.
- Presence/absence of this grid selects the differentiation mode (below).

### Particle set
- Flattened view: `.stack(particle=('i','j'))` or
  `.stack(particle=('i','j','di','dj'))`.
- The `particle` MultiIndex is the inverse map, so grid ↔ set is lossless.
- Parcels consumes plain 1D `lon/lat` arrays; we stash the MultiIndex, run
  advection, reattach the output to the `particle` coord, then `.unstack()`.

## Differentiation modes for `∇F`

The operators are **polymorphic over the presence of the displacement grid**:

1. **Auxiliary grid (recommended default)** — displacement grid present. Each
   `(i, j)` has its own 4-point stencil at separation δ. Decouples diagnostic
   resolution from gradient step; less noisy; degrades gracefully under
   particle loss. ~4–5× particle count. Haller Eq. 9.
2. **Neighbor differencing (fallback)** — no displacement grid. Stencil is the
   neighboring grid points `(i±1, j±1)`. Fewer particles, but welds FTLE
   resolution to the differentiation step and degrades at boundaries.

## Sphere metric (correctness requirement)

Haller's math is Cartesian. Our grid is lon/lat. `∇F` must be formed from
separations in a **local tangent frame in meters** — both the initial stencil
and the final displacements — using `dx = R cos φ dλ`, `dy = R dφ`. Forming
`∇F` directly from degree differences gives wrong eigenvalues (hence wrong FTLE
and ξ directions). Decision: displacement `dx, dy` are in meters; a metric/
projection helper converts to lon/lat for seeding and back for differentiation.

## Tensor representation

Hold 2×2 tensors and 2-vectors as **single DataArrays with component dims**,
not as scalar vars (`F11, F12, …`):
- `gradF`, `C`: dims `(i, j, row, col)` with labeled coord `comp = ['x','y']`.
- eigen output: `λ` on `(i, j, eig)` and `ξ` on `(i, j, comp, eig)`.

This makes the eigen step a one-liner:
`xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row','col']], ...)`
and keeps the operator algebra vectorized.

## Operators (layered)

- **L0 Seeding**: `particle_grid (+ displacement_grid) → particle_set`.
- **L1 Advection**: Parcels (external adapter). Returns final positions.
- **L2 Flow map**: reshape advected positions to the grid; `dX/dy` partials →
  `deformation_gradient` (`∇F`), mode-polymorphic.
- **L3 Tensors**: `cauchy_green` (`C`); `cg_eigen` (`λ₁, λ₂, ξ₁, ξ₂`);
  optionally generalized Green–Lagrange `E_λ` (Eq. 8).
- **L4 Diagnostics**: `ftle = (1/|T|) log √λ_max`; further eigen-derived scalar
  fields.

## Cross-cutting concerns

- **Release time `t0` and integration time `T`**: carried as coords so FTLE time
  series are natural.
- **Forward vs backward integration**: forward → repelling LCS, backward →
  attracting LCS (Haller §3.5 duality). Cheap: a direction/sign on advection.
- **Particle loss**: beached / out-of-domain particles return as NaN/deleted;
  `unstack` must tolerate gaps; any cell with a missing stencil point yields a
  NaN `∇F`. Auxiliary-grid mode is more robust here.

## v1 scope vs later

**v1 (this plan):** L0–L4 — seed → flow map → `C` → eigen → FTLE and
eigen-derived scalar fields, with auxiliary-grid as the primary path and the
sphere metric handled correctly.

**Deferred (separate module):** the geometric LCS layer — shrink/stretch/shear
lines, elliptic (vortex) LCS via closed shear lines (Eqs. 10–11, Table 1). This
is tensor-line ODE integration with eigenvector orientation/desingularization,
a different problem from pointwise grid algebra.

## Open questions (for review)

1. Auxiliary grid as the recommended default, neighbor-differencing as fallback?
2. Metric handling: meters-based local tangent frame (preferred) vs degree-space
   with `cos φ` correction?
3. v1 scope: stop at FTLE + eigen fields, defer tensor-line / elliptic LCS?
4. Tensor storage: component-dim DataArrays (preferred) vs named scalar vars?
