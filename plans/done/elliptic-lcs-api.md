# Plan: elliptic LCS in the package

Bring coherent-vortex detection by closed shear lines (Haller & Beron-Vera
2013, [doi:10.1017/jfm.2013.391](https://doi.org/10.1017/jfm.2013.391)) into
`lcs_parcels`, alongside the hyperbolic layer. The explorative notebooks on
`explore/shear-lines` are the evidence base, not the design: their tracer,
return map and sweep are replaced, not wrapped.

This plan absorbs `plans/lambda-range-and-selection.md`. Its measurement items
reappear below as M2 and M4, and that file moves to `plans/done/` when this
plan is accepted.

## Decisions taken

From the review on 2026-10-05:

- **Scope is elliptic LCS only.** LAVD needs vorticity along whole
  trajectories, which a single flow map does not carry, and is out of scope.
- **The result is a plain `xr.Dataset`**, shaped like `hyperbolic_lcs()`:
  curves on `(eddy, point)`, one row per boundary, per-eddy variables, run
  parameters in `attrs`. No wrapper class.
- **The entry points mirror the hyperbolic layer.** Module functions for each
  step, each runnable by hand, plus a one-call `FlowMap.elliptic_lcs()`.
- **$\lambda$ stays visible.** The scan is an explicit keyword with a
  documented default. The search returns every closed orbit with the
  $\lambda$ it closed at, and a separate selection step picks the boundary.
  The user can display the nested family and reason about the choice.
- **Polarity comes from the polar rotation angle** of $\nabla F$ (Farazmand &
  Haller 2016,
  [doi:10.1016/j.physd.2015.09.007](https://doi.org/10.1016/j.physd.2015.09.007)),
  a new `FlowMap` diagnostic.
- **The centre rule is measured, then chosen.** Both $\lambda_2/\lambda_1$
  minima and polar-rotation extrema are implemented. M1 fixes the default.
- **Both Cape Cauldron notebooks are rewritten on the API.** The hand-rolled
  machinery is deleted from them.
- **Tests use analytic flow maps and synthetic $C$ fields**, built in the test
  with SciPy and no Parcels. The repository ships no test data and gains none.

## Public surface

As planned. See the Outcome section for what changed.

```text
FlowMap.polar_rotation() -> xr.DataArray
elliptic_centres(field, *, extremum, window_m=100_000.0, edge_m=None) -> xr.Dataset
lambda_range(*, lambda_max=1.5, step=0.03) -> np.ndarray
closed_shear_lines(flowmap, *, centre_lon, centre_lat, lambdas=None,
                   max_radius_m=150_000.0, launch_spacing_m=None,
                   step_m=None, closure_tol_m=None) -> xr.Dataset
outermost_shear_lines(orbits, *, flowmap) -> xr.Dataset
FlowMap.elliptic_lcs(**same keywords, optional) -> xr.Dataset
```

Every default above is provisional until the measurement named against it
has run. The defaults in metres are physical lengths, and the `None` defaults
derive from the flow map's grid spacing, so each one means the same at another
resolution.

### `FlowMap.polar_rotation()`

The rotation angle $\theta$ of $R$ in $\nabla F = R U$, on `(i, j)`, in
radians, counter-clockwise positive in the local east/north frame. In two
dimensions it is
$\theta = \operatorname{atan2}(F_{yx} - F_{xy},\ F_{xx} + F_{yy})$, so it
needs no decomposition. A NaN in $\nabla F$ propagates. Returned with `name`
`polar_rotation`, `units` `rad`.

### `elliptic_centres(field, *, extremum, ...)`

Windowed extrema of a field, with a margin `edge_m` (default `window_m / 2`)
kept clear of the domain edge and of NaN cells. `extremum` is `"min"` or
`"max"` and has no default, so the caller states which one a field needs. It
takes a field rather than a flow map, as `ftle_ridge_seeds` does, so the
caller builds the indicator in the open:

```python
eigen = flowmap.cg_eigen()
anisotropy = eigen["lambda"].isel(eig=1) / eigen["lambda"].isel(eig=0)
centres = elliptic_centres(anisotropy, extremum="min")

theta = flowmap.polar_rotation()
centres = elliptic_centres(abs(theta - theta.median()), extremum="max")
```

It shares `_window_geometry` with `ftle_ridge_seeds`, and returns `lon`/`lat`
on a `centre` dim with the same window attributes.

### `lambda_range(*, lambda_max, step)`

The default scan as a public function, so the default is a call the user can
read and vary. It returns $\lambda$ log-symmetric about 1 on
$[1/\Lambda, \Lambda]$, with `step` the spacing in $\ln \lambda$. Convergence
is as common as divergence, so the range brackets 1 symmetrically in the log.

### `closed_shear_lines(flowmap, *, centre_lon, centre_lat, ...)`

For each centre, $\lambda$ in `lambdas` (default `lambda_range()`), and sign
$\pm$, it traces $\eta^\pm_\lambda$ from launch points on two sections. The
sections run due east and due west from the centre, over `max_radius_m`, at
`launch_spacing_m`. The search finds the zeros of $P(s) - s$ on the return
map, refines each by bisection, and keeps an orbit when it:

- returns within `closure_tol_m` of its launch point,
- winds once around its centre, with a winding number of exactly $\pm 1$,
- does not intersect itself.

The winding-number and simple-curve tests replace the notebook's
circumference-ratio filter, which had no principled bound.

It returns **every** orbit that passes, not one per centre. Variables:

| Variable | Dims | Units |
|---|---|---|
| `lon`, `lat` | `(orbit, point)`, closed, NaN-padded | degrees |
| `centre` | `(orbit,)`, index into the given centres | |
| `centre_lon`, `centre_lat` | `(orbit,)` | degrees |
| `lam` | `(orbit,)` | `1` |
| `sign` | `(orbit,)`, branch of $\eta^\pm_\lambda$ | `1` |
| `area_m2` | `(orbit,)`, enclosed area | m2 |
| `radius_m` | `(orbit,)`, $\sqrt{A/\pi}$ | m |
| `residual_m` | `(orbit,)`, closure miss | m |

`radius_m` is the equivalent-area radius, which is the definition GLED uses, so
a boundary compares with an atlas radius directly. The `attrs` record the scan
and its outcome, including the number of (centre, $\lambda$, sign, section)
combinations that never returned.

### `outermost_shear_lines(orbits, *, flowmap)`

It keeps the largest-area orbit per centre. When one centre's boundary
encloses another centre's, it keeps only the outer one. It adds `polarity`,
the sign of the mean `polar_rotation` inside the boundary relative to its
domain median, as $+1$ counter-clockwise or $-1$ clockwise. The result is the
`(eddy, point)` dataset. Converting to cyclonic or anticyclonic needs the
hemisphere, so that conversion stays with the caller and is shown in the
examples.

### `FlowMap.elliptic_lcs()`

It chains centres, search and selection. Only the keywords passed are
forwarded, as in `hyperbolic_lcs()`. It carries the centre field it used and
every step's `attrs` on the result.

## Implementation

1. **Generalise the tracer.** `_trace_half_line` takes the tangent rule as an
   argument instead of calling `_shrink_line_tangent` directly. `shrink_lines`
   output stays bit-identical, and the existing tests pin that.
2. **Add the $\eta^\pm_\lambda$ tangent.** $\xi_2$ is $\xi_1$ turned
   90 degrees counter-clockwise, which ties the two eigenvector signs together
   so the $\pm$ branches cannot swap. The tangent is NaN unless
   $\lambda_1 < \lambda^2 < \lambda_2$.
3. **Vectorise the search across all lines at once**: centres, $\lambda$,
   signs, sections and launch points in one array per step.
   - Tracks are not stored. Each line carries a small state: left the section
     yet, crossed it, crossing position, accumulated winding angle.
   - Only the accepted orbits are re-traced to keep their geometry.
   - Centres are processed in chunks to bound memory.
   - The sweep in `cape_cauldron_gled` takes about 25 minutes per resolution
     in Python loops, and M5 measures what vectorising buys.
4. **Selection, nesting and polarity.**
5. **`polar_rotation`, `elliptic_centres`, `lambda_range`**, and the one-call
   method.
6. **Measurements M1 to M5**, then fix the defaults and record each one next
   to its default.
7. **Rewrite the examples.**
   - `cape_cauldron_vortices` becomes the short API example on the hourly
     model field.
   - `cape_cauldron_gled` keeps the GLED comparison at 1/8 and 1/4 degree on
     the API, and adds sweep B as `closed_shear_lines` from the GLED centres.
   - Rank in AGENTS.md: `cabo_verde_ftle` < `cape_cauldron_vortices` <
     `cape_cauldron_gled`.
8. **Document the layer.**
   - `docs/api.md`: a section "Elliptic LCS: closed shear lines" and rows in
     the metadata table.
   - `docs/notation.md`: $\eta^\pm_\lambda$, $\lambda$, $\theta$.
   - `docs/numerics.md`: the well-definedness condition, the return map, the
     winding test, and why the defaults derive from the grid spacing.
   - `docs/architecture.md`: why the search returns all orbits and selection
     is separate, why centres take a field, and why polarity comes from
     $\nabla F$ and not from $C$.
   - Also the README scope line and `docs/changelog.md`.

The order is TDD, with a Red commit of failing tests before each Green.

## Tests

- **Synthetic $C$, after `_uniform_tensor_interp` in `tests/test_tensorlines.py`.**
  - $\eta^\pm_\lambda$ is a unit vector.
  - The two branches are mirror images about $\xi_1$.
  - $\eta^\pm_\lambda$ is NaN outside $\lambda_1 < \lambda^2 < \lambda_2$.
  - At $\lambda^2 \to \lambda_1$ it reduces to $\xi_1$.
- **Analytic axisymmetric vortex.** The angular velocity is
  $\Omega(r) = \Omega_0 e^{-r^2/R^2}$, integrated with `solve_ivp` into a
  `NeighborFlowMap`. Circles about the centre are material, and their tangent
  elements are not stretched.
  - Every orbit found is a circle about the centre within `closure_tol_m`.
  - Every orbit closes at $\lambda = 1$ within one `step`.
  - The orbits stop where $\lambda_1 < \lambda^2 < \lambda_2$ fails, which in
    the far field is where $C$ is isotropic to round-off. The test fixes
    whether the $\eta$ tangent needs an anisotropy guard as `shrink_lines`
    has.
- **The same vortex in a weak uniform strain.** The outermost orbit lies
  inside the separatrix of the steady streamfunction. It is bounded, and
  stronger strain moves it inward.
- **`polar_rotation`.**
  - $\theta = \Omega T$ for solid-body rotation.
  - $\theta$ flips sign with the rotation sense.
  - $\theta$ is zero for pure strain.
- **Keyword-only.** A swapped centre lon/lat raises `TypeError`.
- **Metadata.** Every returned variable carries `name`, `long_name` and
  `units`.

## Measurements

Each measurement reports a number that fixes a default, and the number is
recorded next to that default.

- **M1, centre rule.** On the GLED case at 1/8 and 1/4 degree, compare
  $\lambda_2/\lambda_1$ minima with polar-rotation extrema. The comparison
  counts GLED eddies matched against the ceiling that the GLED centres
  themselves reach, false-positive centres, and the cost in centres searched.
- **M2, $\lambda$ range and step.**
  - Find whether $\Lambda = 1.5$ brackets every closure in the CMEMS and the
    two geostrophic runs. The 1/4 degree run closed at the old scan floor of
    0.80.
  - Measure how stable the outermost area is per centre across step 0.02,
    0.03 and 0.05.
- **M3, launch spacing and closure tolerance** against grid spacing. Between
  the two runs, the outermost radius at one centre moved from 42.8 to 9.6 km,
  so measure how much of that is the search and how much is the field.
- **M4, truncation against divergence.** This is item 4 of the $\lambda$
  plan. On the geostrophic field, compare $\det \nabla F$ with the stencil arm
  halved. That says whether $\Lambda$ is absorbing truncation error.
- **M5, runtime** of the vectorised search on the GLED case, against the
  current 25 minutes.

## Out of scope

- LAVD and any trajectory ingest.
- Curvilinear grids. The layer interpolates on `lon_grid`/`lat_grid` axes
  like `shrink_lines`.
- Tracking eddies across windows. `FlowMap.image()` already evolves a
  boundary, since a boundary is a material curve.

## Outcome

Implemented on branch `elliptic-lcs`. Where the result departs from the plan
above:

- **`lambda_range` is `stretch_range`, and the data variable is `stretch`.**
  `cg_eigen()` already returns the eigenvalues as `lambda`, and two quantities
  never share a name. The keywords follow: `stretches`, `stretch_max`.
- **There is no self-intersection test.** Two integral curves of a line field
  cannot cross where the field is defined, and the search stops at the first
  return. The winding test stays.
- **The search also accepts direct hits.** On the circles of the analytic
  vortex every launch closes, so $P(s) - s$ never changes sign. A launch that
  returns within `closure_tol_m` is an orbit without bisection.
- **Polarity is `rotation_sense`**, $+1$ counter-clockwise, and the boundary
  centroid is returned as `centroid_lon`/`centroid_lat`. The candidate centre
  can lie off-centre in a merged vortex.
- **The rotation comes from a separate short flow map.** $\theta$ is defined
  modulo $2\pi$. Read off the 30-day flow map, its sign matched GLED's
  polarity for 4 or 5 of 7 to 9 matched eddies. Over 1 or 2 days it matched
  23 of 23 at the GLED centres. `outermost_shear_lines` takes an optional
  `rotation` field instead of a `flowmap`, and `elliptic_lcs()` returns no
  rotation sense.
- **No anisotropy guard on $\eta^\pm_\lambda$.** The analytic vortex closes
  circles out to 58 km without one.

The measurements are recorded in `docs/numerics.md`, "The measurements that
fixed the defaults":

- **M1:** $\lambda_2/\lambda_1$ minima match 8 of 23 GLED eddies at 1/4
  degree, against 5 for polar-rotation extrema and a ceiling of 9 from the
  GLED centres. `elliptic_lcs()` uses the eigenvalue ratio.
- **M2:** every GLED-matched closure lies inside $[1/1.5, 1.5]$. Steps of
  0.02, 0.03 and 0.05 match the same eddies. The defaults stay.
- **M3:** a launch spacing of one grid cell and a tolerance of half a cell
  stay. Halving the spacing adds one 8 km boundary.
- **M4:** halving the stencil arm moves the median $\lvert\det\nabla F\rvert$
  from 3.38 to 1.46 to 1.14, so the bulk departure from area preservation is
  truncation.
- **M5:** 11 s over 250 centres, against about 25 minutes for the
  exploratory loops.
