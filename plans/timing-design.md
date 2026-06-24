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

Store **two** times on the ingested grid:

- `t0` — the release time, as **`datetime64`**.
- `T` — the integration window, as **`timedelta64`**, **signed** (negative `T`
  is backward integration → attracting LCS; positive → repelling).

`t1` is **not** stored and **not** a concept we work with: we never evaluate the
flow map at an absolute end time, only over an interval. Anything that would need
`t1` can derive `t0 + T`, but the API does not expose it.

This supersedes the earlier "store `t0` and `t1`, derive `T`" sketch.

## Why these two, not `(t0, t1)`

- The diagnostics depend on the flow map *over an interval* $F$ acting for a
  duration; FTLE $= \tfrac{1}{|T|}\log\sqrt{\lambda_{\max}}$ needs $T$ directly,
  and $\operatorname{sign}(T)$ is the only source of forward/backward direction
  (the particle set owns this; no separate flag).
- `(t0, T)` is the compact, fully-used layout for ensembles: every
  (release, window) pair is a real experiment. `(t0, t1)` as array dimensions
  would be wasteful — roughly half the rectangle ($t_1$ on the wrong side of
  $t_0$) is unused, and constant-window slices cut across the diagonal.

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
- **Single run:** `t0` and `T` are scalar coordinates — zero overhead.
- **Ensemble:** they promote naturally to **dimension coordinates** `t0` (size
  $N_0$) and `T` (size $N_T$). FTLE then has dims `(i, j, t0)` for a release
  series, `(i, j, t0, T)` for releases × windows. This is plain xarray
  broadcasting — no special machinery (as anticipated in the main plan).
- The clean rectangular `T` axis assumes every release uses the **same set of
  windows**. Ragged windows (windows differ per release) are the exception; they
  fall back to per-sample scalar times rather than a shared `T` dimension.

## API consequence

The ingest factory records `t0` and `T` rather than a bare float:

```python
from_parcels_pset_lon_lat(seed, lon, lat, *, t0, T) -> ParticleGrid
#   t0: datetime64 (scalar or array)
#   T:  timedelta64, signed (scalar or array)
```

`ftle()` reads `t0`/`T` off the coordinates and divides by `|T|`. Converting `T`
(`timedelta64`) to seconds for that division goes through a single helper; note
ocean/model data may use non-standard calendars (360-day, noleap), so the
conversion must be calendar-aware (`datetime64`/`timedelta64` for standard
calendars, cftime where needed).

## Out of scope

- Evaluating or storing the flow map at an absolute `t1`.
- Full intermediate-time trajectories (only the interval endpoints matter for
  pointwise FTLE); an `FTLE`-vs-window sweep is just multiple `T` values.
