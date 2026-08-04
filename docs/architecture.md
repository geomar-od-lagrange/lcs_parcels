# Architecture: seeds and flow maps

Visual overview of the diagnostic layer defined in
[`src/lcs_parcels/grids.py`](../src/lcs_parcels/grids.py). Naming and notation
follow Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
Timing conventions follow [`plans/timing-design.md`](../plans/timing-design.md):
a `Seed` is **time-free**; ingest (`pset_to_flowmap`) is given both the release
time `t0` and the end time `t1`, and the signed window `T = t1 - t0` is derived
and stored on the resulting `FlowMap`. Symbols live in
[`docs/notation.md`](notation.md).

The package contains no Parcels code: a `Seed` *emits* a particle set
(`to_parcels_pset`) and *ingests* the advected positions back into a `FlowMap`
(`pset_to_flowmap`). Parcels itself sits outside this package and owns the
integration; direction (the sign of $T$) follows from `t1` relative to `t0`.

## Class diagram

The lifecycle is split into two **sibling** families — a time-free `Seed` and a
`FlowMap` — that are *not* an inheritance pair: a `FlowMap` is not a kind of
`Seed` (it emits nothing to Parcels). Within each family the two
finite-difference strategies for the deformation gradient $\nabla F$ are modeled
as two explicit subclasses (`Neighbor*`, `Auxiliary*`), not inferred at runtime
from the dataset dimensions. Each class *wraps* an `xr.Dataset` (held in `.ds`)
by composition; none subclasses `xr.Dataset`. The two families are linked only
by the paired class attributes `_flowmap_cls` (seed to its flow map) and
`_seed_cls` (flow map to its seed), and by the two crossing methods.

```mermaid
classDiagram
    class Seed {
        <<abstract>>
        +ds : xr.Dataset
        +_flowmap_cls : type[FlowMap]
        +__init__(ds)
        +lon_grid : xr.DataArray
        +lat_grid : xr.DataArray
        +__repr__() str
        +from_axes(kw lon, lat) Self*
        +to_parcels_pset() tuple
        +pset_to_flowmap(kw lon, lat, t0, t1) FlowMap
    }

    class FlowMap {
        <<abstract>>
        +ds : xr.Dataset
        +_seed_cls : type[Seed]
        +__init__(ds)
        +lon_grid : xr.DataArray
        +lat_grid : xr.DataArray
        +__repr__() str
        +grid_image() xr.Dataset*
        +deformation_gradient() xr.DataArray*
        +cauchy_green() xr.DataArray
        +cg_eigen() xr.Dataset
        +ftle() xr.DataArray
        +image(kw lon0, lat0) xr.Dataset
        +lcs(kw window_m, quantile, ftle_min_per_day, step_m, line_length_m) xr.Dataset
        +to_seed() Seed
    }

    class NeighborSeed {
        +from_axes(kw lon, lat) Self
    }
    class AuxiliarySeed {
        +from_axes(kw lon, lat, aux_separation_m) Self
    }
    class NeighborFlowMap {
        +grid_image() xr.Dataset
        +deformation_gradient() xr.DataArray
    }
    class AuxiliaryFlowMap {
        +grid_image() xr.Dataset
        +deformation_gradient() xr.DataArray
    }

    Seed <|-- NeighborSeed
    Seed <|-- AuxiliarySeed
    FlowMap <|-- NeighborFlowMap
    FlowMap <|-- AuxiliaryFlowMap

    Seed ..> FlowMap : pset_to_flowmap (_flowmap_cls)
    FlowMap ..> Seed : to_seed (_seed_cls)

    note for Seed "Time-free, all-coordinates: diagnostic grid points\nlon_grid/lat_grid and reference release positions lon_0/lat_0 (x_0);\nno t0, no T, no advected lon/lat, no data vars.\nEmits a particle set and ingests it back."
    note for FlowMap "Holds reference + advected positions (lon/lat = F(x_0), the\nonly data vars) plus scalar t0/T. cauchy_green / cg_eigen / ftle / image / lcs\nare concrete on the base, defined via deformation_gradient() and grid_image."
    note for NeighborSeed "Stencil = neighbouring grid points (i +/- 1, j +/- 1).\nNo extra dims beyond (i, j); lon_0/lat_0 equal lon_grid/lat_grid.\nSPASSO / d'Ovidio approach."
    note for AuxiliarySeed "Stencil = fixed four arms east/north/west/south on one\ndisplacement dim, stored explicitly as the reference positions lon_0/lat_0\naround the grid points lon_grid/lat_grid; no centre arm, no diagonals;\ndecouples the gradient step from seed resolution."
```

In the diagram, a trailing `*` marks an abstract member (each concrete subclass
overrides it) and `kw` marks the point past which arguments are keyword-only —
in Python, the bare `*` of `from_axes(*, lon, lat)`. `from_axes` is a
classmethod (constructor); `lon_grid`, `lat_grid` and `grid_image` are
properties, drawn with parentheses only because the diagram has no notation for
one; the rest are instance methods.

`from_axes`, `deformation_gradient` and `grid_image` are the per-stencil seam:
subclasses differ only in the stencil they lay down, in how they
finite-difference $\nabla F$, and in how the advected positions collapse onto
the diagnostic grid. Everything else — emit/ingest on `Seed`, and the diagnostic
chain on `FlowMap` (the Cauchy–Green tensor $C = (\nabla F)^\top \nabla F$, its
eigen-decomposition $C\,\xi_i = \lambda_i\,\xi_i$, and the FTLE
$\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$) — is shared base-class
behaviour.

### Why `lon_grid`/`lat_grid` is one canonical pair

The obvious layout gives the diagnostic grid location no name of its own: it is
already `lon_0`/`lat_0` for the neighbour stencil, and for the auxiliary one,
where `lon_0` lives on `(i, j, displacement)`, it needs a second pair. One
concept under two names forces every downstream consumer to sniff the dataset
for which name is present — the exact runtime introspection the design rules
ban. Carrying `lon_grid`/`lat_grid` on both stencils removes the branch by
construction rather than hiding it behind a helper that branches internally.

`_grid` rather than `_c` for "centre": once the pair exists on both stencils,
"centre" is a misnomer for the neighbour case, which has no arms to be the
centre of. `_grid` says what the point is without knowing the auxiliary story,
which matters because this is the name on every plot axis and in every repr.

For the neighbour stencil `lon_grid` equals `lon_0`, and the value is stored
twice rather than aliased. That is the same call already made for the explicit
auxiliary arms: store the value, do not make a consumer reconstruct it from a
convention. The cost is one duplicated coordinate; the benefit is that a
dataset read off disk is self-describing.

The accessors are concrete one-liners on the two base classes, not abstract
properties overridden four times. Once the coordinate carries one name on both
stencils, `return self.ds["lon_grid"]` branches on nothing, so four identical
overrides would be dead duplication. `grid_image` *is* abstract and overridden,
because there the type genuinely carries information: the neighbour flow map
passes its advected positions through, the auxiliary one takes the centroid of
its four arms (with `skipna=False`, so one lost arm makes the grid point NaN,
matching what the deformation gradient does).

Diagnostics are labelled by `lon_grid`/`lat_grid` and *only* by that pair. The
release positions are dropped in `FlowMap._stencil_meters`, because on an
auxiliary flow map the surviving `lon_0` is one stencil arm — a point about `s`
metres from the grid point the diagnostic describes. Introducing a canonical
coordinate without removing the competing wrong one would have left the trap in
place.

`FlowMap.image` interpolates along axes relabelled `lon_grid`/`lat_grid`, so its
result would naturally come back carrying the caller's arbitrary reference points
under the grid-point name. It renames them to `lon_0`/`lat_0` before returning:
they are the $x_0$ that were mapped, not diagnostic grid points, and one name for
two quantities is the defect the canonical pair exists to remove.

### Why lon/lat pairs are keyword-only

Every public entry point that takes an adjacent lon/lat pair takes it
keyword-only: `from_axes`, `pset_to_flowmap`, `image`, and `shrink_lines`' seed
pair. Two same-typed adjacent arguments are the classic silent-swap footgun —
transposing them does not raise, it returns a plausible-looking field somewhere
off the coast of nowhere. The one thing a positional call buys, brevity, is
worth less than an error at the call site.

The rule is about same-typed neighbours, not about keywords for their own sake:
`shrink_lines(flowmap, ...)` keeps `flowmap` positional, since it is the only
argument of its type and there is nothing to transpose it with. The cost is that
the star-unpacking idiom is unavailable —
`pset_to_flowmap(*seed.to_parcels_pset(), ...)` does not work, and the 2-tuples
that `to_parcels_pset` and `ftle_ridge_seeds` return are unpacked into named
locals first. That is a fair trade for a call form whose entire risk was that it
hid which value went where.

### Why the tuning parameters are physical

The three tunable quantities of the tensor-line layer are stated in the units of
the thing itself — `window_m` and `line_length_m` in metres, `ftle_min_per_day`
as a stretching rate — and converted internally against the field's own grid
spacing and the flow map's own window, so the same call means the same thing at
any resolution and over any horizon. Expressed the natural implementation way
instead, each would depend on something other than what the caller is asking
for: a neighbourhood as a cell count depends on grid resolution, a line as a
step count depends on the step size, and a degeneracy guard as a raw
Cauchy–Green eigenvalue floor silently tightens as $|T|$ grows, turning a guard
into a selector.
`quantile` is left alone: making it absolute would change ridge selection from
relative-to-this-field to an absolute threshold, which is a science decision
rather than a units one.

### Why the metadata is attached at construction

`name`, `long_name` and `units` are stamped on the coordinates once, in
`from_axes` and `pset_to_flowmap`, and ride through the diagnostics untouched,
rather than being re-applied defensively in each operator. The consequence is
that a `FlowMap` built by hand from an unlabelled dataset stays unlabelled —
which is consistent with `FlowMap.__init__` already accepting, say, a
non-axis-aligned grid that `NeighborFlowMap` cannot correctly difference.

The metadata is not decoration: for the reader who verifies the science by
displaying datasets and plotting fields, the `long_name` *is* the plot title and
the `units` *is* the colorbar caption. It is also why no two quantities share a
`name` — `deformation_gradient()` and `cauchy_green()` are built by the same
helper but come back as `deformation_gradient` and `cauchy_green`, since two
distinct quantities under one name silently collide the moment they are merged
into a `Dataset`.

## Flow chart: a typical session

Defining a seed, generating a particle set, advecting it with Parcels
(external), ingesting the result into a flow map, and estimating the FTLE. Nodes
on the package side are method calls; the dashed box is the external Parcels step
that this package does not drive. `to_seed()` closes the loop, recovering a
time-free seed for re-release.

```mermaid
flowchart TD
    axes["1-D lon/lat axes"]
    seed["seed : NeighborSeed<br/>.ds: lon_grid/lat_grid + lon_0/lat_0 (i,j)<br/>coords only, time-free"]
    pset["flat particle set<br/>(lon0, lat0)"]

    subgraph parcels ["Parcels (external, not driven by this package)"]
        advect["ParticleSet(lon0, lat0)<br/>execute(advection) from t0 to t1"]
        out["advected (lon1, lat1)<br/>flat order preserved"]
    end

    fm["flow map : NeighborFlowMap<br/>.ds: lon_grid/lat_grid + lon_0/lat_0 (ref)<br/>+ lon/lat (advected) on (i,j)<br/>scalar coords t0, signed T"]
    gradF["grad F<br/>(i,j,row,col) on lon_grid/lat_grid"]
    C["C = (grad F)^T grad F<br/>(i,j,row,col)"]
    eig["lambda, xi<br/>(ascending lambda, orthonormal xi)"]
    ftle(["FTLE field, 1/s<br/>Lambda(i,j) = (1 / |T|) log sqrt(lambda_max)"])

    axes -->|"from_axes(lon=lon, lat=lat)"| seed
    seed -->|"to_parcels_pset()"| pset
    pset --> advect
    advect --> out
    out -->|"seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)"| fm
    fm -->|"deformation_gradient()<br/>Haller Eq. 9"| gradF
    gradF -->|"cauchy_green()<br/>Eq. 6"| C
    C -->|"cg_eigen()<br/>Eq. 7"| eig
    eig -->|"ftle()<br/>Sec. 4.1"| ftle
    fm -.->|"to_seed() (drop advected lon/lat, t0, T)"| seed

    style parcels stroke-dasharray: 5 5
```

The last four steps (`deformation_gradient` -> `cauchy_green` -> `cg_eigen` ->
`ftle`) are the concrete base-class chain invoked under the hood by a single
`fm.ftle()` call; they are drawn explicitly to show where each Haller quantity
enters.

The `Auxiliary*` pair follows the identical workflow; the only differences are
that the particle set is additionally stacked over the four-arm `displacement`
dim, and `deformation_gradient` differences across that per-point stencil rather
than against neighbouring grid points. Backward integration (attracting LCS) is
selected purely by passing `t1` before `t0` at ingest (so `T = t1 - t0` is
negative); no separate direction flag exists, and a zero window (`t1 == t0`) is
rejected with `ValueError`.

## Extracting LCS: shrink lines

Downstream of the FTLE, the geometric layer
([`src/lcs_parcels/tensorlines.py`](../src/lcs_parcels/tensorlines.py)) turns the
strain field into LCS **curves**. The extraction itself is deliberately *not*
implemented on `FlowMap` (which stays a gridded-diagnostics object) but as two
free functions that consume a `FlowMap`'s xarray outputs — keeping the one new
external dependency (`scipy`, for grid interpolation) at the boundary:

- `ftle_ridge_seeds(ftle)` — start points at the FTLE ridge tops (windowed local
  maxima above a quantile floor);
- `shrink_lines(flowmap, seed_lon=..., seed_lat=...)` — integrates the $\xi_1$
  tensor lines ($\dot r = \xi_1(r)$, Haller Table 1) through those seeds,
  returning an `xr.Dataset` of polylines on `(line, point)`.

Repelling vs. attracting is just *which* flow map is passed: forward gives
repelling LCS, backward gives attracting LCS (the same forward/backward duality
that selects the FTLE's sign of `T`). `FlowMap.lcs()` runs the whole chain:

```python
lcs = forward.lcs()      # repelling; backward.lcs() for attracting
```

`lcs()` is convenience, not a new abstraction. It exists because a caller who
just wants the curves should not have to know that the FTLE is computed twice if
they wire the three steps naively — it evaluates the field once and hands it
down. Ridge-finding deliberately keeps taking a *field* rather than a
`FlowMap`: passing the flow map would hide a recomputation, and a caller who
wants to smooth or mask the FTLE before picking ridges must be able to. The
convenience is therefore bought exactly once, in `lcs()`, and the three
functions underneath stay independently callable.

`lcs()` returns the curves *and* the FTLE field they were seeded from in one
`Dataset`. The first plot anyone makes is the curves over that field, and the
alternative is re-running an eigendecomposition of the whole grid to recover
something `lcs()` had in hand; the `(line, point)` and `(i, j)` dims coexist
without conflict. Its parameters all default to `None` and only those the caller
set are forwarded, so `ftle_ridge_seeds` and `shrink_lines` remain the single
owners of their defaults. There is no direction argument — the flow map already
carries $\operatorname{sign}(T)$, and the returned dataset announces repelling
or attracting in its own `long_name` attributes.

`lcs()` being a method while the extraction lives in `tensorlines` means `grids`
would import `tensorlines`, which already imports the metric helpers from
`grids`. The two names are therefore imported inside `lcs()` rather than at
module level. The alternative — moving the metres frame into a third module —
would touch every import in the package to buy back two lines, so the deferred
import stands until something else needs that module to exist.

## Reprs

`Seed` and `FlowMap` carry terse one-line reprs, defined on the base classes and
reading the `lon_grid`/`lat_grid` accessors, so neither concrete class overrides
anything and a future stencil gets a correct repr for free:

```
<NeighborSeed 6x5 grid, lon -25.000..-20.000, lat 15.000..20.000>
<NeighborFlowMap 6x5 grid, lon -25.000..-20.000, lat 15.000..20.000, t0 2020-01-01T00:00:00, T +7.00 days (forward/repelling)>
```

They are summaries, not dataset dumps: `.ds` remains how the dataset is
displayed, and a repr that is visibly *not* an xarray repr is itself a useful
signal that these objects wrap a dataset rather than subclass one.
