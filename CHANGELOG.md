# Changelog

Versions are CalVer, `YYYY.M.D.N`. The API breaks whenever the design improves,
so each entry says what broke and what to write instead. There are no
deprecation shims: the old form is deleted and every call site updated.

## Unreleased

Changes on `main` since the last tag. This heading becomes the version at
release time.

### Breaking

- **`ftle_ridge_seeds` picks its seeds over a spherical neighbourhood, not a
  rolling window.** A grid point is now a seed when it is the FTLE maximum over
  every grid point within `window_m / 2` of it along the great circle. Nothing in
  the call changes and nothing raises; the seed set differs. On a uniform
  mid-latitude grid it differs by a handful of peaks either way. On a domain
  spanning 20 N to 75 N the old rule resolved `window_m` against one *median*
  cell size and degenerated to a single cell north of 50 N, returning 9.3 times
  as many seeds there as the new rule; below 40 N the two agree exactly. If you
  have tuned `window_m` against a wide-latitude domain, re-check the seed count.
- **`window_m` is a diameter, and `min_seed_separation_m` is exactly half of
  it.** The knob keeps its name, its units and its default, and still means "how
  far apart do I want my seeds, times two". The reported separation no longer
  depends on the grid: it was 16871.8 m for a 30 km window on the Cabo Verde
  example, and is now 15000.0 m.
- **`ftle_ridge_seeds` no longer returns `window_cells_i`, `window_cells_j`,
  `grid_spacing_i_m` or `grid_spacing_j_m`** in `attrs`, on its own result or on
  what `hyperbolic_lcs` returns. They described a cell count that no longer
  exists. Reading one now raises `KeyError`; `window_m` and
  `min_seed_separation_m` are still there.
- **The narrow-window `UserWarning` fires on a different condition.** It used to
  mean "fewer than three grid cells in one dimension" and now means "most grid
  points have no neighbour inside the radius at all". A
  `pytest.warns(UserWarning)` around a small `window_m` still passes; one
  matching the old message text does not.

### Added

- **`UnstructuredAuxiliarySeedGrid` and `UnstructuredAuxiliaryFlowMap`**, the
  auxiliary four-arm stencil around grid points that need not lie on a mesh:
  a flat list, a swath along a ship track, a cluster where the resolution is
  wanted, or a curvilinear mesh. The stencil makes the deformation gradient
  well-defined at a grid point on its own, so nothing in the diagnostic chain
  needed the layout.

  ```python
  seed = UnstructuredAuxiliarySeedGrid.from_points(lon=track_lon, lat=track_lat)
  ```

  An `xr.DataArray` keeps its own dims; a plain 1-D array lands on a
  `grid_point` dim. `FlowMap.image`, `shrink_lines` and `hyperbolic_lcs` all run
  on the result, reading the field between grid points off their triangulation
  instead of along axes. `from_axes` is inherited and still gives a rectilinear
  point set (Closes #20).

- A DOI badge, and the same concept DOI in `CITATION.cff`. It is the
  all-versions DOI, so it resolves to the newest archived release rather than
  pinning to one, and it carries no version of its own.

## 2026.8.8.1

The flow-map repr, a repository link on the documentation site, and citation
metadata. No signature changed, so no call has to be rewritten. A doctest or a
test that compares `repr(flowmap)` against a stored one-line string does not
match any more.

### Changed

- `FlowMap.__repr__` is two lines rather than one. The single line ran to 101
  columns and scrolled sideways in a Markdown code block on GitHub and on PyPI.
  The grid and its extent stay on the first line, `t0` and `T` move to a second
  hanging under the first field, and no field was dropped or rounded. The
  `SeedGrid` repr already fitted and is unchanged.
- The README states the FTLE metadata point without the sentence about
  keyword-only lon/lat pairs, which the example above it already shows.

### Added

- Every page of the documentation site carries a link to the repository, as a
  GitHub button and plain "Source on GitHub" and "Issues" links in the
  alabaster sidebar. The README names the repository too, which is what a
  reader arriving on PyPI sees.
- `CITATION.cff`, which GitHub reads for its "Cite this repository" widget and
  Zenodo reads for the author list of an archived release.

## 2026.8.6.1

Links, badges and notebook output. Nothing in the API changed, so no call has
to be rewritten.

### Changed

- The README's example links point at the rendered notebooks on the
  documentation site rather than at the GitHub blobs. The README is included
  into the site's home page, so those links used to send a reader off the site
  to read a notebook the site itself publishes.
- Links into the documentation use the general URL,
  `https://lcs-parcels.readthedocs.io/`, with deep links behind Read the Docs'
  `/page/` prefix. The versioned form pinned them to `latest`.
- The PyPI badge, its link target, the `pip install` line and the `pypi`
  environment URL in `publish.yml` all spell the canonical `lcs-parcels`. The
  distribution name is unchanged and both spellings resolve, so installing
  works exactly as before.
- The CI badge is pinned to `main`. A red run on a branch or a fork PR no
  longer shows as a failing badge on the front page and on PyPI.
- Parcels is linked at `https://parcels-code.org/`, and `get_data`'s Copernicus
  Marine help link no longer names a retired collection id. Both were
  redirects.

### Added

- `Documentation` in `[project.urls]`, which is what puts the documentation
  site in the PyPI sidebar.
- `CHANGELOG.md` is a page on the documentation site, included rather than
  copied.

### Fixed

- The committed notebooks no longer carry an absolute path from the machine
  that executed them. `ipywidgets` is in the `examples` environment, so
  `copernicusmarine`'s tqdm stops printing an IProgress warning naming the
  environment it ran in.
- The Parcels import cell is tagged `remove-output`. The rendered pages carried
  an ipykernel scratch path inside the alpha-version warning, and about 55 kB
  per notebook of holoviews/bokeh bootstrap whose inlined HTML injected
  `cdn.bokeh.org` script tags into the reader's browser for a library no
  notebook uses.
- `get_data` reports the size of the file on disk whether or not it downloaded
  anything. The download cell used to be silent when the file was already
  there.

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
