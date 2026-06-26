<!--
Standalone design note on how timing information is carried in the LCS
diagnostics data model. Feeds into the API in plans/lcs-api-design.md.
-->

# Timing design

Status: decided.

How a flow-map sample carries time. Follows the notation of
[plans/lcs-api-design.md](lcs-api-design.md) and Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## Decision

There are two distinct representations and they are deliberately different.

**Input.** The seed grid *owns* its release time `t0` (`datetime64`), recorded at
construction by `from_axes(..., t0=...)`. Ingest is then given only the **end
time** `t1` (`datetime64`): `from_parcels_pset_lon_lat(seed, lon, lat, t1=...)`.
The caller never passes `T` — the seed already knows `t0`, so making the caller
also supply `T = t1 - t0` would force them to hand back something the seed
already holds.

**Stored.** On the ingested grid we keep `t0` and the derived, **signed**
integration window:

- `t0` — the release time (from the seed), as **`datetime64`**.
- `T = t1 - t0` — the integration window, as **`timedelta64`**, **signed**
  (negative `T` is backward integration → attracting LCS; positive → repelling).

`t1` is the *input*, not part of the stored model: it is consumed at ingest to
derive `T` and then dropped. We never evaluate or store the flow map at an
absolute end time, only over an interval.

This supersedes the earlier "store `t0` and `t1`, derive `T`" sketch *and* the
intermediate "pass `(t0, T)` at ingest" form.

## Why input `t1`, stored `T`

- **Input `t1`, not `T`.** The seed grid is released at `t0` and the advection
  output is naturally timestamped at the model end time `t1`. Both ends are
  already concrete; `T` is redundant input the seed can compute itself. Passing
  `t1` (and deriving `T = t1 - seed.t0`) removes a hand-computed, error-prone
  argument and keeps the seed the single owner of `t0`.
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
fields have dims `(i, j, t0, T)` → $N_i N_j N_0 N_T$ values. The time coordinates
are 1-D side arrays of length $N_0$ and $N_T$ — smaller than the data by the full
$N_i N_j$ factor, i.e. negligible. So carrying `t0` and `T` as coordinates costs
effectively nothing; the dimension choice (`(t0, T)`, not `(t0, t1)`) is what
keeps the data array minimal.

## Placement in the xarray model

- `t0` and `T` are **coordinates** on `.ds`, not attrs (attrs are dropped by most
  xarray operations; coordinates propagate onto derived fields like `ftle`).
  `t0` is recorded at seeding; `T` is added at ingest.
- **Single run:** `t0` and `T` are scalar coordinates — zero overhead.
- **Ensemble:** they promote naturally to **dimension coordinates** `t0` (size
  $N_0$) and `T` (size $N_T$). The seed may already carry `t0` as a dimension
  (a release series); ingest supplies `t1` (scalar or array, broadcast against
  the seed's `t0`), and `T = t1 - t0` lands on the `(t0, T)` axes. FTLE then has
  dims `(i, j, t0)` for a release series, `(i, j, t0, T)` for releases × windows.
  This is plain xarray broadcasting — no special machinery.
- The clean rectangular `T` axis assumes every release uses the **same set of
  windows**. Ragged windows (windows differ per release) are the exception; they
  fall back to per-sample scalar times rather than a shared `T` dimension.

## API consequence

`from_axes` records `t0`; the ingest factory takes `t1` and derives `T`:

```python
from_axes(lon, lat, *, t0) -> ParticleGrid
#   t0: datetime64 (scalar or array) — recorded on .ds

from_parcels_pset_lon_lat(seed, lon, lat, *, t1) -> ParticleGrid
#   t1: datetime64 (scalar or array)
#   stores t0 (from seed) and T = t1 - seed.t0  (signed timedelta64)
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
