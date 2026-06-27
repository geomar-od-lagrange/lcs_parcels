<!--
Design note for splitting the single ParticleGrid into two lifecycle types,
Seed and FlowMap. Supersedes the single-class framing in
plans/lcs-api-design.md and refines the timing model in plans/timing-design.md.
Once implemented, docs/api.md and docs/architecture.md replace their
ParticleGrid descriptions with the Seed/FlowMap pair.
-->

# Seed / FlowMap lifecycle split

Status: decided (design); not yet implemented.

Follows the notation of [docs/notation.md](../docs/notation.md) and Haller (2015),
*Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech. 47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## Context

The current design models a particle grid as a single `ParticleGrid` ABC
(`NeighborGrid`, `AuxiliaryGrid`) that serves *both* lifecycle states:
before advection (a seed to release) and after (an advected flow-map sample to
diagnose). To keep that one object internally consistent it carries two
fictions: a seed's advected `lon`/`lat` are seeded *equal* to the reference
`lon_0`/`lat_0` (the identity flow map $F_{t_0}^{t_0}(x_0)=x_0$), and its window
is $T=0$.

This conflates two different responsibilities, and it lets the diagnostic
methods run on an un-advected seed and return plausible garbage: `ftle()` on a
seed is $\tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ with $T=0$ (a divide-by-zero)
and `deformation_gradient()` is the identity everywhere. Nothing errors. This
is exactly the "infer state from data" pattern the project conventions warn
against ([AGENTS.md](../AGENTS.md): *model distinct concepts as distinct
types*).

It also collides with the round-trip data model: `to_parcels_pset` stacks the
grid into a flat `particle` index, which only works while the seed's dims are
the spatial/stencil dims. The moment `t0`/`T` become real dimensions (an
ensemble cube; see [plans/timing-design.md](timing-design.md)), the stack is
ill-defined — but those dims are exactly what diagnostics want.

We resolve both by splitting the lifecycle into two independent types.

## The two type families

The stencil axis (Neighbor vs Auxiliary) and the lifecycle axis (pre- vs
post-advection) are orthogonal. We keep the stencil split and add the lifecycle
split, giving two small ABCs with two concrete classes each. They are
*siblings*, not an inheritance pair: a `FlowMap` is not a kind of `Seed` (it
emits nothing), so inheriting would re-merge the two lifecycle states the split
exists to separate. Shared code lives at module level (see below), not on a
common base.

```
Seed (ABC)                       FlowMap (ABC)
├── NeighborSeed                 ├── NeighborFlowMap
└── AuxiliarySeed                └── AuxiliaryFlowMap
```

- **`Seed`** — the Parcels *boundary* object. Holds only the reference release
  positions (and, for Auxiliary, the arm geometry). It is **time-free**: no
  `t0`, no `T`, no advected positions. Lays out and emits particles; ingests
  the advected positions back through `pset_to_flowmap`.
- **`FlowMap`** — the *diagnostics* object. Holds reference + advected
  positions, the release time `t0`, and the signed window `T`. Computes the
  deformation gradient through FTLE. Never talks to Parcels.

This is not over-engineering: the split *removes* the identity/`T=0` fictions
rather than adding indirection, and it maps onto the existing "keep Parcels at
the boundary" convention — `Seed` is the boundary, `FlowMap` is the math.

## The two crossings (no composition)

The types are connected by two factories; **neither holds a reference to the
other**.

```python
Seed.pset_to_flowmap(lon, lat, *, t0, t1) -> FlowMap   # boundary -> diagnostics
FlowMap.to_seed()                          -> Seed      # diagnostics -> boundary
```

- **`Seed.pset_to_flowmap`** replaces today's `from_parcels_pset_lon_lat`
  classmethod. The name states both ends of the transformation (pset ->
  flowmap), and it marks this as a *data-consuming* factory, unlike the
  self-contained `to_parcels_pset`/`to_seed` converters. It is a *behaviour of
  the seed*: it needs exactly what the seed already owns — the reference
  positions and the `particle` MultiIndex — so it lives there, not as a
  `FlowMap` constructor reaching into a foreign object. The window enters here
  as arguments: `t0` (release time) and `t1` (end time), from which it derives
  the signed `T = t1 - t0` and lands `t0`/`T` on the `FlowMap`. It keeps emit
  and ingest symmetric on one type, so a round-trip test stays on a single
  object: `fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t0=..., t1=...)`.
  The `lon`/`lat` argument is the advected flat arrays pulled off the returned
  Parcels `ParticleSet` — never a Parcels object itself, keeping Parcels at the
  boundary.
- **`FlowMap.to_seed`** answers "does a `FlowMap` need a back-link to its
  `Seed`?" — no; it reconstructs one. Ingest never overwrites the reference
  `lon_0`/`lat_0` (nor, for Auxiliary, the arm geometry or centres), so a
  `FlowMap` still carries a pristine launcher. `to_seed` is a *lossless drop* of
  the advected `lon`/`lat`, `t0`, and `T` — yielding a time-free seed. For
  Auxiliary this is better than re-running `from_axes`: it rebuilds from the
  carried arms, needing neither the original 1-D axes nor `aux_separation_m`
  (which a reloaded or `concat`-ed `FlowMap` no longer has).

The re-run workflow (same grid, sweep `t1`, or re-release at a new `t0`) is
therefore "`to_seed()` then `pset_to_flowmap(..., t0=..., t1=...)`", with no
shared state: the seed is a reusable spatial template and every release passes
its own `(t0, t1)`.

## Lifecycle invariant makes the stacking problem disappear

Only `Seed` has `to_parcels_pset` and the `particle` stack. `FlowMap` has **no
emit method at all**. So a `FlowMap` is free to carry `t0`/`T` as full
dimensions — a single experiment *or* an assembled ensemble cube
`(i, j, t0, T)` built by `xr.concat`/`combine_by_coords` over single FlowMaps —
and the "stack breaks with a `T` dim" problem *cannot occur*, because nothing
ever stacks a `FlowMap`. The discipline from
[plans/timing-design.md](timing-design.md) ("scalar at the boundary, dims in
the cube") becomes a structural type invariant instead of a convention.

Because each single `FlowMap` carries its own scalar `t0`/`T`, cube assembly is
self-aligning: `xr.concat([fm.ds for fm in runs], dim="T")` promotes each
scalar `T` into the `T` axis at the right slice. No manual alignment.

## Stacking rule for `to_parcels_pset`

`to_parcels_pset` stacks over whatever spatial dims the *release* carries:

- `(i, j)` always,
- `displacement` for Auxiliary.

Neither `t0` nor `T` is ever stackable: the seed is time-free (it carries no
`t0`), and `T` does not exist until ingest. Build the stack list from
`lon_0.dims` — `("i", "j")` for Neighbor, `(..., "displacement")` for
Auxiliary. Emit is a **2-tuple** `(lon, lat)`.

### The seed carries no time

The release time is a property of a release *event*, not of the grid of
positions: the same seed can be released at many `t0`. So `Seed` carries no time
at all, and emit stays the 2-tuple above — no broadcasting `lon_0` across a
`t0` dim, no per-particle release time, no 3-tuple. Both ends of the window
enter at ingest, `pset_to_flowmap(lon, lat, *, t0, t1)`, which derives the
signed `T = t1 - t0` and lands `t0`/`T` on the `FlowMap`.

A release series (sweep `t0`) is therefore an *external loop* over scalar `t0`:
each release is advected independently and ingested to a scalar-`(t0, T)`
`FlowMap`, then assembled with `xr.concat`/`combine_by_coords` into the
`(i, j, t0, T)` cube — the same self-aligning assembly the lifecycle-invariant
section relies on. This matches "scalar at the boundary, dims in the cube":
independent FTLE runs are naturally separate Parcels psets anyway, so nothing is
lost by not packing a `t0` dim into a single emit.

## Dataset schemas

Reference positions stay **coordinates**; advected positions are **data
variables**; `t0`/`T` are coordinates (so they propagate onto derived fields).

| | dims | coords | data vars |
|---|---|---|---|
| `NeighborSeed` | `i, j` | `lon_0, lat_0 (i,j)` | — |
| `AuxiliarySeed` | `i, j, displacement` | `lon_0, lat_0 (i,j,displacement)`, `lon_c, lat_c (i,j)`, `displacement` | — |
| `NeighborFlowMap` | `i, j` (+ `t0, T`) | `lon_0, lat_0 (i,j)`, `t0, T` | `lon, lat (i,j…)` |
| `AuxiliaryFlowMap` | `i, j, displacement` (+ `t0, T`) | `lon_0, lat_0 (i,j,displacement)`, `lon_c, lat_c (i,j)`, `displacement`, `t0, T` | `lon, lat (i,j,displacement…)` |

A `Seed` is "all coordinates" (no data vars) — its payload is the release
positions $x_0$. A `FlowMap` adds the advected $F_{t_0}^{t_1}(x_0)$ as the only
data vars and the derived signed `T`.

## Class structure

The two crossings and the diagnostic chain are stencil-agnostic and live on the
bases, wired to their stencil partner by a paired class attribute. Only
`from_axes`, the Auxiliary arm layout, and `deformation_gradient` are genuinely
per-stencil.

- **`Seed` (ABC)**: `from_axes` (abstract classmethod, per stencil);
  `to_parcels_pset` (concrete — generic stack over `lon_0.dims`,
  coord-stripped via `reset_coords(drop=True)`); `pset_to_flowmap` (concrete —
  reattach advected onto the `particle` index, `unstack`, set `t0` and
  `T = t1 - t0`, construct `self._flowmap_cls(ds)`). Concrete seeds set
  `_flowmap_cls`.
- **`FlowMap` (ABC)**: `deformation_gradient` (abstract, per stencil);
  `cauchy_green`, `cg_eigen`, `ftle` (concrete on the base, unchanged from
  today); `to_seed` (concrete — drop advected `lon`/`lat`, `t0`, and `T`,
  construct `self._seed_cls(ds)`). Concrete flow maps set `_seed_cls`.
- **Shared infrastructure** stays module-level functions, used by both families
  without duplication: `EARTH_RADIUS_M`, `_lonlat_to_meters`,
  `_meters_to_lonlat`, `_assemble_tensor`, and the centroid metric helpers
  (`_reference_lonlat`, `_to_meters`). The metric is needed by `AuxiliarySeed`
  (arm layout in meters) *and* both `FlowMap`s (the grad F denominator), so it
  cannot belong to either base alone. Module-level keeps it shared with no
  adapter layer.

Keep everything in `src/lcs_parcels/grids.py` (the module still holds particle
grids); no file split.

## Migration (follow through — leave nothing half-done)

- **`src/lcs_parcels/grids.py`** — implement the four classes + two ABCs as
  above; delete `ParticleGrid` and the identity/`T=0` seeding.
- **`src/lcs_parcels/__init__.py`** — export `Seed, NeighborSeed,
  AuxiliarySeed, FlowMap, NeighborFlowMap, AuxiliaryFlowMap`; drop the
  `ParticleGrid`/`NeighborGrid`/`AuxiliaryGrid` names.
- **`tests/conftest.py`** — the `advected_grid` helper becomes
  `seed = NeighborSeed.from_axes(...); fm = seed.pset_to_flowmap(lon_out, lat_out, t0=..., t1=...)`.
- **`tests/test_grids.py`** — split into seed-shape tests (no `lon`/`lat`, no
  `T` on a seed) and flow-map-shape tests; drop the identity (`lon == lon_0`)
  and `T == t0 - t0` assertions (they no longer exist on a seed).
- **`tests/test_roundtrip.py`** — `seed.to_parcels_pset()` then
  `seed.pset_to_flowmap(..., t0=, t1=)`; identity round-trip asserts
  `fm.ds["lon"] ≈ seed.ds["lon_0"]`; the emit-ingest-emit losslessness test
  re-emits via `fm.to_seed().to_parcels_pset()` and asserts the time-free seed
  carries no `t0`/`T`.
- **`tests/test_operators.py`** — diagnostics now run on `FlowMap`s; the
  backward-integration and NaN-knockout tests carry over unchanged in substance.
- **`tests/test_lcs_parcels.py`** — assert the two new hierarchies instead of
  the `ParticleGrid` base.
- **`examples/example_grid_pset.py`** (jupytext `.py`/`.md`/`.ipynb` triple) —
  rewrite to the `Seed` → `pset_to_flowmap` → `FlowMap` flow and
  re-`jupytext --sync`;
  commit the `.ipynb` code-only.
- **`AGENTS.md`** — update the boundary convention: emit via
  `Seed.to_parcels_pset()` (a 2-tuple `(lon, lat)`) and ingest via
  `Seed.pset_to_flowmap(lon, lat, *, t0, t1)` returning a `FlowMap`; the seed is
  **time-free**, ingest takes both `t0` and `t1` and derives `T = t1 - t0`.
  Replaces the current "the seed grid owns its release time `t0`" wording.
- **`docs/api.md`, `docs/architecture.md`, `docs/notation.md`** — replace the
  `ParticleGrid` description and the class diagram with the Seed/FlowMap pair
  and the two crossings.
- **`plans/lcs-api-design.md`, `plans/timing-design.md`** — mark the
  single-`ParticleGrid` framing as superseded by this note; the timing model
  (release time enters at ingest, derive signed `T`, scalar-at-boundary) is
  unchanged in substance, now enforced structurally by a time-free `Seed`.

## Verification

- `pixi run -e dev pytest` green (seed-shape, flow-map-shape, round-trip
  including `to_seed` re-emit, operators, backward integration, NaN
  propagation).
- `ruff` clean.
- `jupytext --sync examples/example_grid_pset.py` then execute the notebook;
  confirm the printed identity round-trip and the FTLE field come out as the
  prose claims (no claimed result that hasn't actually run).
