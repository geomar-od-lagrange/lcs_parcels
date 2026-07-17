<!--
Implementation plan for Q1 of plans/lcs-time-evolution.md: evolving an already
extracted hyperbolic LCS as a material curve, purely from captured flow maps.
Concretizes the design agreed in review; a fresh session implements from here.
Q2 (sliding-window tracking) and Q3 (OECS) are deferred.
-->

# Q1 — evolving an LCS as a material curve (implementation plan)

Status: **planned, ready to implement** — Q1 only. Q2 and Q3 from
[`plans/lcs-time-evolution.md`](lcs-time-evolution.md) are explicitly deferred.
Notation follows Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## What Q1 does

An extracted hyperbolic LCS is a **material** curve at $t_0$ (a set of $x_0$
points, exactly the `lon`/`lat` that `shrink_lines` returns). Its position at
another time is $\mathcal{M}(t) = F_{t_0}^{t}(\mathcal{M}(t_0))$ (Haller Eq. 5).
We watch one *fixed* LCS move — we do **not** re-diagnose it (that is Q2).

## Design decisions (agreed in review)

1. **Reuse the flow map; do not start a new particle run.** The flow map
   $F_{t_0}^{t}$ *is* the advected-position field `(lon(x_0), lat(x_0))` that a
   `FlowMap` already stores. Evolving the LCS is interpolating that field at the
   LCS's $x_0$ points — no second integrator, and self-consistent with the map
   that diagnosed the LCS. (A fresh Parcels advection of the curve would be a
   second, inconsistent discretization.)
2. **Evolve each LCS in its coherent direction.** A repelling LCS is attracting
   in backward time; an attracting LCS is attracting in forward time. Evolving in
   the coherent direction makes perturbations *decay* (well-conditioned); the
   opposite direction makes them grow exponentially (the curve filaments and the
   numerics blow up — intrinsic, not a solver artifact). So:
   - **attracting** LCS (diagnosed from the *backward* flow) → evolve **forward**;
   - **repelling** LCS (diagnosed from the *forward* flow) → evolve **backward**.
3. **Diagnose once at the max-$|T|$ end, then evolve via intermediate maps.** The
   LCS is sharpest at the longest window, so diagnose there; walk the *same*
   material curve through the intermediate flow maps.

Caveat to document, not fix: the evolved curve's fidelity is bounded by the
flow-map **grid** resolution (sub-grid folding is not resolved). In the coherent
direction there is little folding, so this is acceptable; it is the reason the
flow-map route is smoother than re-advecting the curve, not a loss of accuracy.

## The primitive: image of $x_0$ under a flow map

A single small helper — the only genuinely new package piece.

**Recommendation:** a `FlowMap` method (it *applies* the map the object stores,
alongside `cauchy_green`/`ftle`; and via `xarray.interp` it needs no direct
`scipy` import in `grids.py`). Free-function `flow_map_image(flowmap, lon0, lat0)`
is the acceptable alternative if we prefer to keep `FlowMap` diagnostics-only, as
with `shrink_lines`.

```python
FlowMap.image(lon0, lat0) -> xr.Dataset   # lon/lat on the input dims
```

- `lon0`, `lat0`: xarray `DataArray`s of $x_0$ points sharing arbitrary dims —
  typically the `(line, point)` grid `shrink_lines` returns, or a single along-LCS
  `param` dim. **Vectorized and dim-preserving.**
- Implementation: reindex the advected fields onto rectilinear 1-D `lon_0`/`lat_0`
  axes (`lon_0` varies along `i`, `lat_0` along `j`, as in `NeighborFlowMap`),
  then `ds[["lon", "lat"]].interp(lon_0=lon0, lat_0=lat0,
  kwargs=dict(bounds_error=False, fill_value=np.nan))`. `xarray.interp` does the
  vectorization and carries the indexer dims onto the output for free.
- NaN handling falls out: a land/edge cell (NaN advected position) propagates to
  NaN through linear interpolation, and NaN-padded input points (the tails of
  `shrink_lines` curves) map to NaN — so the evolved curve terminates exactly
  where the flow map is undefined, matching the diagnostics.
- Returns an `xr.Dataset` with `lon`/`lat` on the input dims — the *same*
  structure as a `shrink_lines` curve, so an evolved LCS is drop-in plottable and
  could itself be re-fed.

### Tests (`tests/test_evolution.py`)

Use `conftest.advected_flowmap(AuxiliarySeed, …, M, …)` (constant $M$ ⇒ the
advected-position field is linear, so interpolation is exact):

- image at a grid node equals that node's stored `ds.lon`/`ds.lat`;
- image at an interior off-node point equals the analytic linear value;
- dim preservation: pass a `(param,)` and a `(line, point)` `DataArray` → output
  carries the same dims;
- an off-grid point yields `NaN`; a NaN input point yields `NaN`.

## Capturing intermediate flow maps

Run each grid advection (the two we already run for the forward/backward FTLE) in
$\Delta$-chunks; after each chunk, snapshot `pset.x`/`pset.y` and ingest a
`FlowMap` via `seed.pset_to_flowmap(..., t0=t0, t1=t0 ± kΔ)`. Successive
`execute` calls continue from the current positions, so chunk $k$ gives exactly
$F_{t_0}^{t_0 \pm kΔ}$ (flow-map composition). Result: a list of `FlowMap`s per
direction.

- Only the **max-$|T|$** `FlowMap` feeds diagnosis (`ftle`, `ftle_ridge_seeds`,
  `shrink_lines`); the intermediates are used *only* as position maps via
  `.image(...)` (no gradients needed on them).
- This adds no integration cost beyond the two advections already run — just
  extra snapshots.

## Evolution workflow (the notebook)

```
forward maps  {F^{+kΔ}}  (t0 → t0+T, chunked)   # also diagnoses repelling at +T
backward maps {F^{-kΔ}}  (t0 → t0-T, chunked)   # also diagnoses attracting at -T

repelling  = shrink_lines(fwd_maps[-1],  *ftle_ridge_seeds(fwd_maps[-1].ftle()))
attracting = shrink_lines(bwd_maps[-1],  *ftle_ridge_seeds(bwd_maps[-1].ftle()))

# evolve each in its coherent direction, via the OTHER run's intermediates:
attr_evo = concat([m.image(attracting.lon, attracting.lat) for m in fwd_maps], "lead")
rep_evo  = concat([m.image(repelling.lon,  repelling.lat)  for m in bwd_maps], "lead")
```

Each run is thus reused twice: the forward run diagnoses repelling *and* evolves
attracting; the backward run diagnoses attracting *and* evolves repelling. The
`concat` over a `lead` dim yields an evolution cube `lon`/`lat` on
`(lead, line, point)`; snapshots are `.isel(lead=k)`.

**Notebook** `examples/cabo_verde_lcs_evolution.py` (jupytext-managed, executed,
offline from the bundled currents): the two chunked runs, the two diagnoses, the
two evolutions, and a snapshot figure (attracting at $t_0, +\tfrac{T}{2}, +T$;
repelling at $t_0, -\tfrac{T}{2}, -T$). Keep it Q1-focused; no OECS, no tracking.

## Validation

Before trusting the flow-map route, compare it once against a direct Parcels
advection of the same LCS vertices (the throwaway prototype) in the coherent
direction — they should agree to within the grid-interpolation error. Report the
actual max/median separation (a number, not a claim). This can live in the
notebook as a short check or in a scratch script referenced from the PR.

## Deferred

- **Q2** — sliding-window re-extraction + cross-frame tracking (birth/death/merge).
- **Q3** — instantaneous OECS from the rate-of-strain tensor.

Both are surveyed in [`plans/lcs-time-evolution.md`](lcs-time-evolution.md); Q3's
"tensor lines of an arbitrary symmetric tensor field" refactor is *not* needed
for Q1 and should not be pulled in early.

## References

- Haller, G. (2015). *Lagrangian Coherent Structures.* Annu. Rev. Fluid Mech.
  47:137–162.
  [doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
- Haller, G. & Sapsis, T. (2011). *Lagrangian coherent structures and the
  smallest finite-time Lyapunov exponent.* Chaos 21:023115.
  [doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
