# Architecture

Why the diagnostic layer in
[`src/lcs_parcels/grids.py`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/src/lcs_parcels/grids.py) is shaped the way it
is: what the types are, how they compose, and which alternatives were rejected.
The companion document [`docs/numerics.md`](numerics.md) covers the other half —
why the numbers that come out of these types are right: the local east/north
frame each separation is measured in, the units the tuning parameters are stated
in, and the well-definedness guard. Symbols live in
[`docs/notation.md`](notation.md); naming and notation follow Haller (2015),
*Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech. 47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

Timing conventions follow [`plans/timing-design.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/plans/done/timing-design.md):
a `SeedGrid` carries no time, and ingest (`pset_to_flowmap`) is given both the release
time $t_0$ and the end time $t_1$, and the signed window $T = t_1 - t_0$ is
derived and stored on the resulting `FlowMap`.

The package contains no Parcels code: a `SeedGrid` *emits* a particle set
(`to_parcels_pset`) and *ingests* the advected positions back into a `FlowMap`
(`pset_to_flowmap`). Parcels itself sits outside this package and owns the
integration; direction (the sign of $T$) follows from $t_1$ relative to $t_0$.

## Structure

### Two sibling families, not an inheritance pair

The lifecycle is split into a `SeedGrid` family and a `FlowMap` family.
They are *siblings*: a `FlowMap` is not a kind of `SeedGrid`, because it emits
nothing to Parcels, and a `SeedGrid` is not a kind of `FlowMap`, because it has no
advected positions and no window. Neither base inherits from the other, and
there is no common base above both, because there is almost nothing the two
families would share in one. The separation helpers are module-level functions
called only from the `FlowMap` side (`AuxiliarySeedGrid.from_axes` inverts the same
relation inline to place its arms). The two families share `__init__`, which
stores the dataset on `.ds`, and the `lon_grid`/`lat_grid` accessors, which read
it back; each is one line, written identically on `SeedGrid` and on `FlowMap`. The
repr's grid summary is a module-level function that both reprs call.

A `SeedGrid` holds coordinates only: the diagnostic grid points
`lon_grid`/`lat_grid` and the reference release positions `lon_0`/`lat_0`
($x_0$). No $t_0$, no $T$, no advected positions, no data variables. A
`FlowMap` adds the advected positions `lon`/`lat` — the flow map image
$F(x_0)$, and its only data variables — plus the scalar coordinates `t0` and
signed `T`.

Each class *wraps* an `xr.Dataset`, held in `.ds`, by composition; none
subclasses `xr.Dataset`. The public API is therefore the handful of methods that
have a meaning for a seed or a flow map, with `.ds` as the escape hatch into
plain xarray.

The two families are linked only by the paired class attributes
`_flowmap_cls` (seed to its flow map) and `_seed_cls` (flow map back to its
seed), and by the two crossing methods `SeedGrid.pset_to_flowmap` and
`FlowMap.to_seed`. A new stencil is therefore a `SeedGrid` subclass and a `FlowMap`
subclass that name each other, and nothing else changes.

### The two stencils

Within each family, the two finite-difference strategies for the deformation
gradient $\nabla F$ are modelled as two explicit subclasses — `NeighborSeedGrid` /
`NeighborFlowMap` and `AuxiliarySeedGrid` / `AuxiliaryFlowMap` — rather than
inferred at runtime from the dataset's dimensions. The type carries that
information, so nothing inspects the dataset for a `displacement` dim to decide
what to do.

- **Neighbour stencil** (`Neighbor*`): the stencil is the neighbouring grid points
  $(i \pm 1, j \pm 1)$. No dims beyond `(i, j)`, and `lon_0`/`lat_0` equal
  `lon_grid`/`lat_grid`. This is the SPASSO approach (see `src/Diagnostics.py`
  in [SPASSO](https://github.com/OceanCruises/SPASSO)), and it ties the gradient
  step to the seed resolution.
- **Auxiliary stencil** (`Auxiliary*`): each grid point carries four arms —
  east, north, west, south — on a single `displacement` dim, stored explicitly
  as the reference positions `lon_0`/`lat_0` around the grid point. There is no
  centre arm, no diagonals and no stored `dx`/`dy`; the reference positions
  record the stencil. The arms decouple the gradient step (`aux_separation_m`)
  from the seed resolution.

`from_axes`, `deformation_gradient` and `grid_image` are the per-stencil seam,
and they are the only abstract members: `from_axes` lays the stencil down,
`deformation_gradient` differences $\nabla F$ across it, and `grid_image`
collapses the advected positions onto the diagnostic grid. Everything else is
shared, concrete base-class behaviour — emit and ingest on `SeedGrid`, and on
`FlowMap` the whole diagnostic chain: the Cauchy–Green tensor
$C = (\nabla F)^\top \nabla F$, its eigen-decomposition
$C\,\xi_i = \lambda_i\,\xi_i$, the FTLE
$\Lambda = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$, plus `image` and
`hyperbolic_lcs`.

Signatures for all of this are in [`docs/api.md`](api.md); on the public
surface every adjacent lon/lat pair is keyword-only.

## Design decisions

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
centre of. `_grid` names the point without reference to the auxiliary arms, and
that name appears on every plot axis and in every repr.

For the neighbour stencil `lon_grid` equals `lon_0`, and the value is stored
twice rather than aliased. That is the same call already made for the explicit
auxiliary arms: store the value, do not make a consumer reconstruct it from a
convention. It costs one duplicated coordinate, and a dataset read off disk is
then self-describing.

The accessors are concrete one-liners on the two base classes, not abstract
properties overridden four times. Once the coordinate carries one name on both
stencils, `return self.ds["lon_grid"]` branches on nothing, so four identical
overrides would be dead duplication. `grid_image` *is* abstract and overridden,
because there the type genuinely carries information: the neighbour flow map
passes its advected positions through, the auxiliary one takes the centroid of
its four arms (with `skipna=False`, so one lost arm makes the grid point NaN,
matching what the deformation gradient does).

Diagnostics are labelled by `lon_grid`/`lat_grid` and *only* by that pair. The
release positions are dropped in `FlowMap._positions`, because on an
auxiliary flow map the surviving `lon_0` is one stencil arm — a point about $s$
metres from the grid point the diagnostic describes. Keeping `lon_0`/`lat_0`
alongside the canonical pair would leave a second, wrong label available on the
diagnostics.

`FlowMap.image` interpolates along axes relabelled `lon_grid`/`lat_grid`, so its
result would naturally come back carrying the caller's arbitrary reference points
under the grid-point name. It renames them to `lon_0`/`lat_0` before returning:
they are the $x_0$ that were mapped rather than diagnostic grid points, and the
canonical pair exists so that one name never covers two quantities.

### Why the advected positions are `lon`/`lat`, not `lon_1`/`lat_1`

The reference positions carry a `_0` suffix and the times are `t0` and `t1`, so
`lon_1`/`lat_1` would be the symmetric choice for the arrival positions. They are
plain `lon`/`lat` instead, for two reasons.

Every product downstream of the flow map returns the position of something at
the time it is asked about, and returns it as `lon`/`lat`: `grid_image`,
`FlowMap.image`, `ftle_ridge_seeds` and `shrink_lines` all do. Naming the flow
map's own arrival positions `lon_1` would make `flowmap.ds` disagree with
`flowmap.image()` about what an arrival position is called, and the suffix would
have to be dropped again at the first function that has no `t1` to refer to.

The suffix also means something narrower than "the second time". `_0` marks the
*reference* position, the $x_0$ a diagnostic is differenced from and reported
against; the unsuffixed name is the current position. That is the distinction
Haller writes as $x_0$ and $F(x_0)$ rather than $x_0$ and $x_1$, and it survives
into a chained evolution where the output of one `image` call becomes the `x_0`
of the next.

### Why `_positions` hands out degrees, not metres

`FlowMap._positions` returns the reference and advected positions as
`(lon_0, lat_0, lon, lat)` in *degrees*; the subclass gradients convert to metres
themselves, through `_separation_m`. The earlier version converted first and
handed the stencil out already in metres, which only works if there is one frame
for the whole grid to convert into. There is not: a separation becomes metres
only once two points are named, because it is taken in the local east/north frame
of *that pair* — the longitude difference wrapped to $[-180, 180]$ and scaled by
the cosine of the pair's mid-latitude, the latitude difference by the Earth
radius alone. The advected pair therefore gets its own cosine rather than the
reference pair's, which is exactly the rescaling that makes $\nabla F$ report
stretching for a rigid meridional translation.

The same argument removed the shared projection everywhere else: there is no
`lat_ref`, no standard parallel, and no function that converts a position (as
opposed to a pair) to metres. `_wrap_lon` applies to differences only and
`_circular_mean_lon` anchors on the first element along its dim, so stored
longitudes keep whatever convention they arrived in. Nothing normalises them
onto a branch of the package's choosing.

`tensorlines._step_lonlat_by_meters` applies the same rule on the integration
side. It is written as the *exact inverse* of `_separation_m` — the same
mid-latitude cosine, solved for the longitude increment — rather than as the
direct great-circle problem, so that measuring the step it took reproduces the
direction it was given. A great-circle step is the more accurate arc, but it
curves away from the direction field the ODE integrates, and that error
accumulates linearly over the hundreds of steps in a line
([`numerics.md`](numerics.md#stepping-a-tensor-line) has the measurement).
The increment is added to the incoming longitude, so a traced line stays on its
seed's branch across the antimeridian.

`FlowMap.image` is the one place the package has to put positions from different
sources on a common branch: it interpolates the advected longitude field, and
arithmetic on a field that tears from 179.9 to -179.9 between adjacent grid
points reads the tear as a 40 000 km jump. Rather than normalising anything, it
re-anchors each advected longitude on the branch of its own grid point,
`lon_grid + _wrap_lon(lon - lon_grid)`, which is a difference operation again
and is correct for any displacement short of 180 degrees.

### Why lon/lat pairs are keyword-only

Every public entry point that takes an adjacent lon/lat pair takes it
keyword-only: `from_axes`, `pset_to_flowmap`, `image`, and `shrink_lines`' seed
pair. Two same-typed adjacent arguments can be transposed silently: the call
does not raise, it returns a plausible-looking field for the wrong location.
Keyword-only turns a transposition into a `TypeError` at the call site, at the
cost of a longer call.

The rule covers same-typed neighbours only:
`shrink_lines(flowmap, ...)` keeps `flowmap` positional, since it is the only
argument of its type and there is nothing to transpose it with. The cost is that
the star-unpacking idiom is unavailable —
`pset_to_flowmap(*seed.to_parcels_pset(), ...)` does not work, and the 2-tuple
`to_parcels_pset` returns is unpacked into named locals first, which costs one
line per call site. `ftle_ridge_seeds` pays that cost in the return type
instead: it hands back an `xr.Dataset` whose two fields are already named, so
the seeds reach the next call as
`shrink_lines(fm, seed_lon=seeds["lon"], seed_lat=seeds["lat"])`, with no tuple
to unpack.

### Why the metadata is attached at construction

`name`, `long_name` and `units` are stamped on the coordinates once, in
`from_axes` and `pset_to_flowmap`, and ride through the diagnostics untouched,
rather than being re-applied defensively in each operator. The consequence is
that a `FlowMap` built by hand from an unlabelled dataset stays unlabelled —
which is consistent with `FlowMap.__init__` already accepting, say, a
non-axis-aligned grid that `NeighborFlowMap` cannot correctly difference.

For the reader who verifies the science by displaying datasets and plotting
fields, the `long_name` is the plot title and the `units` is the colorbar
caption. The same reasoning keeps any two quantities from sharing a `name`:
`deformation_gradient()` and `cauchy_green()` are built by the same helper but
come back as `deformation_gradient` and `cauchy_green`, because two distinct
quantities under one name collide silently when they are merged into a
`Dataset`.

## Walkthrough: a typical session

Defining a seed, generating a particle set, advecting it with Parcels
(external), ingesting the result into a flow map, and estimating the FTLE. Every
step but the advection is a method call on a `SeedGrid` or a `FlowMap`:

1. `NeighborSeedGrid.from_axes(lon=lon, lat=lat)` turns two 1-D axes into a seed
   whose `.ds` carries `lon_grid`/`lat_grid` and `lon_0`/`lat_0` on `(i, j)` —
   coordinates only, and carrying no time.
2. `seed.to_parcels_pset()` flattens the release positions into a 2-tuple of
   1-D arrays `(lon_0, lat_0)`.
3. **Parcels, external and not driven by this package**, builds a `ParticleSet`
   from those arrays and executes an advection from $t_0$ to $t_1$, returning
   advected positions in the same flat order.
4. `seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)` folds them back onto
   the seed's grid, yielding a `NeighborFlowMap` whose `.ds` adds the advected
   `lon`/`lat` as data variables and the scalar coords `t0` and signed `T`.
5. `deformation_gradient()` (Haller Eq. 9) differences those positions against
   the reference ones, each pair in its own local east/north frame, giving
   $\nabla F$ on `(i, j, row, col)`, labelled by `lon_grid`/`lat_grid`.
6. `cauchy_green()` (Eq. 6) forms $C = (\nabla F)^\top \nabla F$, `cg_eigen()`
   (Eq. 7) returns its eigenvalues in ascending order with orthonormal
   eigenvectors, and `ftle()` (§4.1) reduces that to
   $\Lambda(i, j) = \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ in 1/s.

Those last four steps are the concrete base-class chain that a single
`fm.ftle()` call invokes under the hood; they are spelled out here to show where
each Haller quantity enters. `fm.to_seed()` drops the advected positions, `t0`
and `T` to recover a seed grid for re-release.

The `Auxiliary*` pair follows the identical workflow; the only differences are
that the particle set is additionally stacked over the four-arm `displacement`
dim, and `deformation_gradient` differences across that per-point stencil rather
than against neighbouring grid points. Backward integration (attracting LCS) is
selected purely by passing `t1` before `t0` at ingest (so $T = t_1 - t_0$ is
negative); no separate direction flag exists, and a zero window ($t_1 = t_0$) is
rejected with `ValueError`.

## Walkthrough: extracting LCS as shrink lines

Downstream of the FTLE, the geometric layer
([`src/lcs_parcels/tensorlines.py`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/src/lcs_parcels/tensorlines.py)) turns the
strain field into LCS **curves**. The extraction is implemented as two free
functions that consume a `FlowMap`'s xarray outputs rather than as methods on
`FlowMap`, which stays a gridded-diagnostics object. That keeps the one new
external dependency (`scipy`, for grid interpolation) at the boundary:

- `ftle_ridge_seeds(ftle)` — finds seed points at the FTLE ridge tops (windowed
  local maxima above a magnitude floor), returning an `xr.Dataset` of
  `lon`/`lat` on a `seed` dim whose attributes record the floor that was applied
  and what `window_m` became on this grid;
- `shrink_lines(flowmap, seed_lon=..., seed_lat=...)` — integrates the $\xi_1$
  tensor lines ($\dot r = \xi_1(r)$, Haller Table 1) through those seeds,
  returning an `xr.Dataset` of polylines on `(line, point)`.

Repelling versus attracting is set by which flow map is passed: forward gives
repelling LCS, backward gives attracting LCS — the forward–backward duality of
Haller & Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
`FlowMap.hyperbolic_lcs()` runs the whole chain:

```python
lcs = forward.hyperbolic_lcs()   # repelling; backward.hyperbolic_lcs() for attracting
```

The method is named for the family of LCS it extracts. Elliptic LCS are a
separate extraction with a separate parameter set, so the generic name `lcs()`
would have to be undone when that extraction lands (GitHub issue #8).

`hyperbolic_lcs()` is a convenience wrapper and introduces no new type. It
evaluates the FTLE once and hands that field to the ridge finder, so a caller
who only wants the curves need not arrange the three steps. Ridge-finding still
takes a *field* rather than a `FlowMap`: passing the flow map would hide a
recomputation, and a caller who wants to smooth or mask the FTLE before picking
ridges must be able to. The three functions underneath therefore stay
independently callable.

`hyperbolic_lcs()` returns the curves *and* the FTLE field they were seeded from
in one `Dataset`. The first plot anyone makes is the curves over that field, and
returning the curves alone would force a second eigendecomposition of the whole
grid to recover a field the call already had; the `(line, point)` and `(i, j)`
dims coexist without conflict. The seeds' own attributes ride along on it as
well, minus their `long_name`, which describes the seed points rather than the
curves: a caller who never sees the intermediate `Dataset` can still read
`lcs.attrs["ftle_threshold"]` and `lcs.attrs["min_seed_separation_m"]` off the
result.

Its parameters all default to `None` and only those the caller set are
forwarded, so `ftle_ridge_seeds` and `shrink_lines` remain the single owners of
their defaults. That holds for the mutually exclusive pair too: passing both
`quantile` and `ftle_min` here forwards both, and the `ValueError` is raised in
`ftle_ridge_seeds`, which is where the rule lives. There is no direction
argument — the flow map already carries $\mathrm{sign}(T)$, and the returned
dataset announces repelling or attracting in its own `long_name` attributes.
What those parameters mean and why they are stated in the units they are is in
[`docs/numerics.md`](numerics.md).

`hyperbolic_lcs()` being a method while the extraction lives in `tensorlines`
means `grids` would import `tensorlines`, which already imports the separation
helpers from `grids`. The two names are therefore imported inside
`hyperbolic_lcs()` rather than at module level. The alternative — moving those
helpers into a third module — would touch every import in the package to buy
back two lines, so the deferred import stands until something else needs that
module to exist.

### Why `window_m` was not redefined as the seed separation

`window_m` is the *side* of the window a candidate must be the maximum over, so
the window reaches `window_m / 2` to either side of its own grid point and the
closest two seeds can be is about half the value passed in. That is easy to
misread, and the obvious fix is to redefine the knob as the minimum seed
separation.

That redefinition was rejected because the knob is already in use. Redefining it
doubles the window of every existing call, and the call still runs and still
returns seeds — different ones. Breaking changes are the norm here, but this one
would not announce itself: a renamed argument raises `TypeError` and a changed
return type raises at the next line, while a redefined float raises nothing.

The meaning therefore stands, and the consequence is reported instead.
`_window_geometry` returns `min_seed_separation_m` alongside the cell counts and
the median grid spacings, and every one of those keys lands in the returned
dataset's `attrs`. The value is computed rather than approximated: a window of
`cells` reaches `(cells - 1) // 2` cells to either side, so the nearest point
that can also be a windowed maximum is one cell beyond that, and the reported
figure is the smaller of the two per-dimension distances. It comes out near
`window_m / 2` on a regular grid, but the reported number is the one this grid
actually produces.

The `UserWarning` for a `window_m` spanning fewer than three cells in either
dimension is the same concern. A one-cell window makes every point its own
maximum, so `ftle >= peak` is satisfied everywhere and the local-maximum test
stops selecting; what comes back is every point above the magnitude floor. Since
the call does not fail and the output is a plausible-looking seed set, the
package warns rather than raising.

### Why an absolute FTLE floor exists at all

Tuning parameters here are meant to be scale-free, and an absolute FTLE floor is
not: a stretching rate that marks a ridge in a fast flow marks nothing in a slow
one, and the same number retunes with the region, the window, and the season.
So `quantile` remains the default and `ftle_min` has to be asked for by name.

It exists because some analyses need exactly the property the quantile lacks. A
quantile floor is defined by the field it is applied to, so two runs — a
three-day window against a ten-day one, one region against its neighbour — are
each thresholded at their own top decile, and the seed counts come out
comparable however the underlying strain differed. A run whose subject is that
difference needs one threshold held fixed across all of it, which a quantile
cannot express. The two selectors are therefore mutually exclusive, and passing
both raises a `ValueError` rather than following a precedence rule: "quantile
0.9 and 1e-6 1/s" has no reading a caller is likely to have meant.

## Reprs

`SeedGrid` and `FlowMap` carry terse one-line reprs, defined on the base classes and
reading the `lon_grid`/`lat_grid` accessors, so neither concrete class overrides
anything and a future stencil gets a correct repr for free:

```text
<NeighborSeedGrid 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00>
<NeighborFlowMap 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00, t0 2020-01-01T00:00:00, T +7.0 days>
```

They are summaries rather than dataset dumps: `.ds` remains how the dataset is
displayed, and a repr that does not look like an xarray repr signals that these
objects wrap a dataset rather than subclass one.

The repr carries the signed $T$ and no direction word. Repelling versus
attracting is a property of the *diagnostic*, not of the flow map, so the flow
map states its window and the diagnostic (`hyperbolic_lcs`) states which family
it extracted; printing both would have been redundant.
