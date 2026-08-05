# Changelog

Versions are CalVer, `YYYY.M.D.N`. The API breaks whenever the design improves,
so each entry says what broke and what to write instead. There are no
deprecation shims: the old form is deleted and every call site updated.

## 2026.8.5.1

First release on PyPI. `pip install lcs_parcels`.

Everything below changed since the last tag, `v2026.08.04.1`. Nothing outside
this repository can have depended on it yet, but the entries are written as they
will be from now on.

### Breaking

- **The seed classes are renamed.** `Seed` is now `SeedGrid`, `NeighborSeed` is
  `NeighborSeedGrid`, and `AuxiliarySeed` is `AuxiliarySeedGrid`. The name says
  what the object is, a spatial grid of release positions. Write
  `NeighborSeedGrid.from_axes(lon=lon, lat=lat)` where you wrote
  `NeighborSeed.from_axes(...)`.
- **`FlowMap.image` takes `lon_0`/`lat_0`, not `lon0`/`lat0`.** They are
  keyword-only, so the old spelling raises `TypeError` rather than silently
  doing the wrong thing. The names now match the coords the method returns.
  Write `flowmap.image(lon_0=curve["lon"], lat_0=curve["lat"])`.
- **Two message strings are shorter.** `ftle_ridge_seeds`' `ValueError` is now
  `"give either quantile or ftle_min, not both"`, and its narrow-window
  `UserWarning` ends `"Consider wider window_m or finer grid."`. Code matching
  on the text breaks; code catching the exception or warning class does not.

- **`ftle_ridge_seeds` returns an `xr.Dataset`, not a `(lon, lat)` tuple.**
  `lon` and `lat` are on a `seed` dim, and the dataset's `attrs` carry the
  ridge-selection metadata. Write
  `seeds = ftle_ridge_seeds(ftle)` and then
  `shrink_lines(fm, seed_lon=seeds["lon"], seed_lat=seeds["lat"])`
  where you wrote `seed_lon, seed_lat = ftle_ridge_seeds(ftle)`.
- **`ftle_ridge_seeds(quantile=...)` no longer defaults in the signature.**
  `quantile` and the new `ftle_min` both default to `None`, and passing neither
  still selects `quantile=0.90`. Passing both raises `ValueError`. Existing calls
  that pass `quantile` explicitly are unaffected.
- **`deformation_gradient` returns different numbers.** Separations are now
  measured in each pair's own local east/north frame rather than in one
  equirectangular frame with a standard parallel at the seed centroid. Nothing
  in the call changes; the values are corrected. A rigid meridional translation
  used to report an FTLE of exactly zero and now reports the stretching it
  causes.
- **`AuxiliarySeedGrid.from_axes` raises `ValueError` near a pole**, where
  `aux_separation_m` would span 90 degrees of longitude or more, closer to a
  pole than about `0.64 * aux_separation_m`. It previously returned an arm on
  the far side of the pole and a wrong gradient.
- **`requires-python` is now `>=3.12`**, up from an untested `>=3.11`.

### Added

- `ftle_min`, an absolute FTLE floor for ridge selection, beside the default
  `quantile`. Use it when several windows or regions need one common threshold,
  which a quantile cannot express.
- `min_seed_separation_m` and the rest of the window geometry on the
  `ftle_ridge_seeds` result, and on what `hyperbolic_lcs` returns. A 30 km
  `window_m` puts seeds about 15 km apart, and that number is now reported
  rather than left to be inferred.
- A `UserWarning` when `window_m` spans fewer than three grid cells, where the
  local-maximum test stops selecting anything.

### Fixed

- `FlowMap.image` no longer tears at the antimeridian. An advection that returns
  positions wrapped to `[-180, 180)`, which is what Parcels and CMEMS do, used
  to give a linear interpolant a 360-degree jump between adjacent grid points.
- Longitude differences and means wrap, so a domain crossing the antimeridian
  works. Longitudes are never normalised on ingest.
- The tensor-line step follows the direction field instead of drifting off it.
  The previous step left a heading-invariant field at a rate that accumulated
  linearly in the step count, 3.4 km off a parallel at 70 N over an 800 km line
  at a 20 km step.

## Earlier tags

`v2026.07.17.1` and `v2026.08.04.1` predate PyPI and predate the unpadded CalVer
rule. They are kept as they are.
