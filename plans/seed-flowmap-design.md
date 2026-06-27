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
split, giving two small ABCs with two concrete classes each:

```
Seed (ABC)                       FlowMap (ABC)
├── NeighborSeed                 ├── NeighborFlowMap
└── AuxiliarySeed                └── AuxiliaryFlowMap
```

- **`Seed`** — the Parcels *boundary* object. Holds only the reference release
  positions and the release time. Lays out and emits particles; ingests the
  advected positions back. No advected positions, no `T`.
- **`FlowMap`** — the *diagnostics* object. Holds reference + advected positions
  and the signed window. Computes the deformation gradient through FTLE. Never
  talks to Parcels.

This is not over-engineering: the split *removes* the identity/`T=0` fictions
rather than adding indirection, and it maps onto the existing "keep Parcels at
the boundary" convention — `Seed` is the boundary, `FlowMap` is the math.

## The two crossings (no composition)

The types are connected by two factories; **neither holds a reference to the
other**.

```python
Seed.pset_to_flowmap(lon, lat, *, t1) -> FlowMap   # boundary -> diagnostics
FlowMap.to_seed(*, t0=None)            -> Seed      # diagnostics -> boundary
```

- **`Seed.pset_to_flowmap`** replaces today's `from_parcels_pset_lon_lat`
  classmethod. The name states both ends of the transformation (pset ->
  flowmap), and it marks this as a *data-consuming* factory, unlike the
  self-contained `to_parcels_pset`/`to_seed` converters. It is a *behaviour of
  the seed*: it needs exactly what the seed already owns — the reference
  positions, the `particle` MultiIndex, and `t0` — so it lives there, not as a
  `FlowMap` constructor reaching into a foreign object. It keeps emit and ingest
  symmetric on one type, so a round-trip test stays on a single object:
  `fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t1=...)`. The `lon`/`lat`
  argument is the advected flat arrays pulled off the returned Parcels
  `ParticleSet` — never a Parcels object itself, keeping Parcels at the
  boundary.
- **`FlowMap.to_seed`** answers "does a `FlowMap` need a back-link to its
  `Seed`?" — no; it reconstructs one. Ingest never overwrites the reference
  `lon_0`/`lat_0` (nor, for Auxiliary, the arm geometry or centres), so a
  `FlowMap` still carries a pristine launcher. `to_seed` is a *lossless drop* of
  the advected `lon`/`lat` and `T`. For Auxiliary this is better than re-running
  `from_axes`: it rebuilds from the carried arms, needing neither the original
  1-D axes nor `aux_separation_m` (which a reloaded or `concat`-ed `FlowMap` no
  longer has). The optional `t0=` override is free and safe because the arms are
  in absolute lon/lat — moving `t0` does not move them; a scalar or array `t0`
  there is the entry point to the release-series ensemble.

The re-run workflow (same grid, sweep `t1`, or re-release at a new `t0`) is
therefore "`to_seed()` then `pset_to_flowmap(..., t1=...)`", with no shared
state.

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

`to_parcels_pset` stacks over whatever non-scalar dims the *release* carries:

- `(i, j)` always,
- `displacement` for Auxiliary,
- `t0` **when `t0` is a release dimension** (a release series), broadcasting the
  reference positions across `t0` so every particle carries its own release
  time. Scalar `t0` stays a scalar coord and is not stacked.

`T` is never stackable — it does not exist until ingest. Concretely, build the
stack list from `lon_0.dims` plus `t0` when `t0` is a dimension, broadcasting
`lon_0`/`lat_0` across `t0` first. This generalizes today's hardcoded
`("i","j")` / `(...,"displacement")`.

### Decision to confirm — emit the per-particle release time

If `t0` can be stacked into `particle`, Parcels needs a per-particle release
time at emit. Recommended: make emit uniform,

```python
Seed.to_parcels_pset() -> tuple[list[float], list[float], list[np.datetime64]]
#   (lon, lat, time);  time = per-particle t0 (scalar t0 -> all equal)
```

Rationale: the release time is something the seed already owns and Parcels needs
regardless, so emitting it is *more correct* even for a single `t0`, and the
release-series case then falls out for free — Parcels supports non-lockstep
release times. The cost is a 3-tuple return that the tests and the example must
unpack. The minimal alternative (keep the 2-tuple, support only scalar `t0`,
defer release-series) is cheaper now but leaves the dim-`t0` path half-built.
Given the conventions' "follow through; don't half-migrate", the 3-tuple is the
recommendation — **confirm before implementing**, as it is the one scope choice
here.

## Dataset schemas

Reference positions stay **coordinates**; advected positions are **data
variables**; `t0`/`T` are coordinates (so they propagate onto derived fields).

| | dims | coords | data vars |
|---|---|---|---|
| `NeighborSeed` | `i, j` (+ `t0`) | `lon_0, lat_0 (i,j)`, `t0` | — |
| `AuxiliarySeed` | `i, j, displacement` (+ `t0`) | `lon_0, lat_0 (i,j,displacement)`, `lon_c, lat_c (i,j)`, `displacement`, `t0` | — |
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
  `to_parcels_pset` (concrete — generic stack over `lon_0.dims` (+ `t0`),
  coord-stripped via `reset_coords(drop=True)`); `pset_to_flowmap` (concrete —
  reattach advected onto the `particle` index, `unstack`, set `T = t1 - t0`,
  construct `self._flowmap_cls(ds)`). Concrete seeds set `_flowmap_cls`.
- **`FlowMap` (ABC)**: `deformation_gradient` (abstract, per stencil);
  `cauchy_green`, `cg_eigen`, `ftle` (concrete on the base, unchanged from
  today); `to_seed` (concrete — drop advected `lon`/`lat` and `T`, optional
  `t0` override, construct `self._seed_cls(ds)`). Concrete flow maps set
  `_seed_cls`.
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
  `seed = NeighborSeed.from_axes(...); fm = seed.pset_to_flowmap(lon_out, lat_out, t1=...)`.
- **`tests/test_grids.py`** — split into seed-shape tests (no `lon`/`lat`, no
  `T` on a seed) and flow-map-shape tests; drop the identity (`lon == lon_0`)
  and `T == t0 - t0` assertions (they no longer exist on a seed).
- **`tests/test_roundtrip.py`** — `seed.to_parcels_pset()` then
  `seed.pset_to_flowmap(..., t1=)`; identity round-trip asserts `fm.ds["lon"] ≈
  seed.ds["lon_0"]`; the emit-ingest-emit losslessness test re-emits via
  `fm.to_seed().to_parcels_pset()`. Unpack the 3-tuple if the emit-time decision
  lands.
- **`tests/test_operators.py`** — diagnostics now run on `FlowMap`s; the
  backward-integration and NaN-knockout tests carry over unchanged in substance.
- **`tests/test_lcs_parcels.py`** — assert the two new hierarchies instead of
  the `ParticleGrid` base.
- **`examples/example_grid_pset.py`** (jupytext `.py`/`.md`/`.ipynb` triple) —
  rewrite to the `Seed` → `pset_to_flowmap` → `FlowMap` flow and
  re-`jupytext --sync`;
  commit the `.ipynb` code-only.
- **`AGENTS.md`** — update the boundary convention: emit via
  `Seed.to_parcels_pset()` and ingest via
  `Seed.pset_to_flowmap(lon, lat, *, t1)` returning a `FlowMap`; the seed owns
  `t0`, ingest takes `t1` and derives `T = t1 - t0`.
- **`docs/api.md`, `docs/architecture.md`, `docs/notation.md`** — replace the
  `ParticleGrid` description and the class diagram with the Seed/FlowMap pair
  and the two crossings.
- **`plans/lcs-api-design.md`, `plans/timing-design.md`** — mark the
  single-`ParticleGrid` framing as superseded by this note; the timing model
  (own `t0`, derive signed `T` at ingest, scalar-at-boundary) is unchanged in
  substance, now enforced structurally.

## Verification

- `pixi run -e dev pytest` green (seed-shape, flow-map-shape, round-trip
  including `to_seed` re-emit, operators, backward integration, NaN
  propagation).
- `ruff` clean.
- `jupytext --sync examples/example_grid_pset.py` then execute the notebook;
  confirm the printed identity round-trip and the FTLE field come out as the
  prose claims (no claimed result that hasn't actually run).
