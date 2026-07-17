<!--
Standalone design note on how timing information is carried in the LCS
diagnostics data model. Feeds into the API in plans/lcs-api-design.md.
-->

# Timing design

> **Superseded by [`plans/seed-flowmap-design.md`](seed-flowmap-design.md) for
> the class structure and the input side of the timing model.** The single
> `ParticleGrid` (`from_axes(..., t0=...)` owning `t0`, ingest taking only `t1`)
> described below has been replaced by a time-free `Seed` and a `FlowMap`: the
> seed carries no time, and **both** `t0` and `t1` enter at ingest
> (`Seed.pset_to_flowmap(lon, lat, *, t0, t1)`). The stored side — `(t0, T)` with
> signed `T = t1 - t0`, scalar at the boundary and a dimension only in the
> assembled cube — is unchanged in substance, but a release series is now an
> **external loop** over scalar `t0` assembled with `xr.concat`, not an extra
> broadcast dimension on the seed. The prose below has been corrected for that
> split but is otherwise a historical record.

Status: decided.

How a flow-map sample carries time. Follows the notation of
[plans/lcs-api-design.md](lcs-api-design.md) and Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## Decision

There are two distinct representations and they are deliberately different.

**Input.** The seed is **time-free**: `from_axes(lon, lat)` records no time.
Ingest is given **both** the release time `t0` and the **end time** `t1`
(`datetime64`): `Seed.pset_to_flowmap(lon, lat, *, t0, t1)`. The caller never
passes `T` directly — ingest derives the signed `T = t1 - t0` from the two
timestamps. (Both ends entering at ingest replaces the earlier model where the
seed owned `t0` and ingest took only `t1`; the seed is now a reusable spatial
template that every release stamps with its own `(t0, t1)`.)

**Stored.** On the ingested grid we keep `t0` and the derived, **signed**
integration window:

- `t0` — the release time (from the seed), as **`datetime64`**.
- `T = t1 - t0` — the integration window, as **`timedelta64`**, **signed**
  (negative `T` is backward integration -> attracting LCS; positive ->
  repelling).

`t1` is the *input*, not part of the stored model: it is consumed at ingest to
derive `T` and then dropped. We never evaluate or store the flow map at an
absolute end time, only over an interval.

This supersedes the earlier "store `t0` and `t1`, derive `T`" sketch *and* the
intermediate "pass `(t0, T)` at ingest" form.

## Why input `t1`, stored `T`

- **Input `(t0, t1)`, not `T`.** A release happens at `t0` and the advection
  output is naturally timestamped at the model end time `t1`. Both ends are
  already concrete; `T` is a redundant input ingest can derive. Passing the two
  timestamps and deriving `T = t1 - t0` removes a hand-computed, error-prone
  argument. Both ends enter at ingest because the seed is time-free; the caller
  must pass the same `t0` it released the particles at (see the "agree at both
  ends" contract in [`plans/seed-flowmap-design.md`](seed-flowmap-design.md)).
- **Stored `(t0, T)`, not `(t0, t1)`.** The diagnostics depend on the flow map
  *over an interval* acting for a duration; FTLE
  $= \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ needs $T$ directly, and
  $\operatorname{sign}(T)$ is the only source of forward/backward direction (no
  separate flag). `(t0, T)` is also the compact, fully-used layout for
  ensembles: every (release, window) pair is a real experiment, whereas
  `(t0, t1)` as array dimensions wastes roughly half the rectangle ($t_1$ on the
  wrong side of $t_0$) and cuts constant-window slices across the diagonal.

## Footprint

Footprint is governed by the **data** array, not the time labels. FTLE / eigen
fields have dims `(i, j, t0, T)`, i.e. $N_i N_j N_0 N_T$ values. The time coordinates
are 1-D side arrays of length $N_0$ and $N_T$ — smaller than the data by the full
$N_i N_j$ factor, i.e. negligible. So carrying `t0` and `T` as coordinates costs
effectively nothing; the dimension choice (`(t0, T)`, not `(t0, t1)`) is what
keeps the data array minimal.

## Placement in the xarray model

- `t0` and `T` are **coordinates** on the `FlowMap`'s `.ds`, not attrs (attrs are
  dropped by most xarray operations; coordinates propagate onto derived fields
  like `ftle`). Both are added at ingest, derived from the `(t0, t1)` arguments;
  the seed carries neither.
- **Single run:** `t0` and `T` are scalar coordinates on the `FlowMap` — zero
  overhead.
- **Ensemble:** they become **dimension coordinates** `t0` (size $N_0$) and `T`
  (size $N_T$) only in the *assembled cube*. Because the seed is time-free, a
  release series is an **external loop** over scalar `t0`: each release is emitted,
  advected independently, and ingested via `pset_to_flowmap(..., t0=, t1=)` to a
  scalar-`(t0, T)` `FlowMap`. The per-run flow maps are then assembled with
  `xr.concat`/`combine_by_coords`, which promotes each scalar `t0`/`T` into the
  `(t0, T)` axes at the right slice (self-aligning, no manual broadcasting). FTLE
  then has dims `(i, j, t0)` for a release series, `(i, j, t0, T)` for releases by
  windows.
- The clean rectangular `T` axis assumes every release uses the **same set of
  windows**. Ragged windows (windows differ per release) are the exception; they
  fall back to per-sample scalar times rather than a shared `T` dimension.

## API consequence

`from_axes` records no time; the ingest factory takes both `t0` and `t1` and
derives `T`:

```python
Seed.from_axes(lon, lat) -> Seed
#   time-free — no t0 recorded

Seed.pset_to_flowmap(lon, lat, *, t0, t1) -> FlowMap
#   t0, t1: datetime64 (scalar)
#   stores t0 and T = t1 - t0  (signed timedelta64) on the FlowMap
```

`ftle()` reads `t0`/`T` off the coordinates and divides by `|T|`. Converting `T`
(`timedelta64`) to seconds for that division goes through a single helper; note
ocean/model data may use non-standard calendars (360-day, noleap), so the
conversion must be calendar-aware (`datetime64`/`timedelta64` for standard
calendars, cftime where needed).

## Out of scope

- Storing `t1` on the grid, or evaluating the flow map at an absolute `t1` as a
  diagnostic (`t1` enters only as the ingest timestamp from which `T` is
  derived).
- Full intermediate-time trajectories (only the interval endpoints matter for
  pointwise FTLE); an `FTLE`-vs-window sweep is just multiple `T` values.
