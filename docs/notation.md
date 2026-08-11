# Notation reference

This document defines the symbols and conventions used throughout
`lcs_parcels`. Naming follows Haller (2015), *Lagrangian Coherent Structures*,
Annu. Rev. Fluid Mech. 47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
Each symbol is listed with its Haller equation number and with the name it
carries in the code. Math is written in LaTeX; equation numbers refer to
Haller (2015).

## Symbol table

| Symbol | Meaning | Haller Eq. | Code name |
|---|---|---|---|
| $v(x, t)$ | velocity field, with position $x = (x^1, x^2)$ in 2D | 2 | (input, external) |
| $x_0$ | initial (reference) particle *release* position; for the `Neighbor*` classes the grid point itself, for the auxiliary classes the explicit stencil arms; carried by both the seed and the flow map | 3 | `lon_0`, `lat_0` (the grid dims, plus `displacement` on the auxiliary classes) |
| $x_{\mathrm{grid}}$ | diagnostic grid point, where every diagnostic is reported; carried explicitly by every pair and by both families |  | `lon_grid`, `lat_grid` |
| $F_{t_0}^{t_1}(x_0) = x(t_1; t_0, x_0)$ | flow map: initial position $\to$ position at time $t_1$; stored as the advected positions on a `FlowMap` (its only data vars) | 3 | `lon`, `lat`, sharing the dims of `lon_0`/`lat_0` |
| $F_{t_0}^{t_1}(x_{\mathrm{grid}})$ | flow map image of the diagnostic grid points: the advected positions reduced onto the grid dims | 3 | `grid_image` |
| $\nabla F_{t_0}^{t_1}(x_0)$ | deformation gradient (the gradient of the flow map; $2\times 2$ in 2D) | 4, 9 | `deformation_gradient`, `gradF` |
| $C(x_0) = \left(\nabla F_{t_0}^{t_1}\right)^\top \nabla F_{t_0}^{t_1}$ | right Cauchy–Green strain tensor ($2\times 2$, symmetric positive-definite) | 6 | `cauchy_green`, `C` |
| $C\,\xi_i = \lambda_i\,\xi_i,\ \ 0 < \lambda_1 \le \lambda_2,\ \ \xi_1 \perp \xi_2$ | eigen-decomposition of $C$ | 7 | `cg_eigen` |
| $\lambda_1, \lambda_2$ | eigenvalues of $C$ (ordered $0 < \lambda_1 \le \lambda_2$); $\lambda_{\max} = \lambda_2$ | 7 | `lambda` (coord `eig`) |
| $\xi_1, \xi_2$ | orthonormal eigenvectors of $C$ | 7 | `xi` (coords `comp`, `eig`) |
| $\Lambda_{t_0}^{t_1}(x_0) = \dfrac{1}{\lvert t_1 - t_0\rvert}\,\log\sqrt{\lambda_{\max}}$ | finite-time Lyapunov exponent (FTLE); uses the largest eigenvalue, and $\lvert T\rvert$ so that a backward map ($T < 0$) gives a positive exponent | §4.1 | `ftle` |
| $t_0$ | release time; supplied at ingest (`pset_to_flowmap`) and stored as a scalar coord on the `FlowMap` | 3 | `t0` (flow map coord) |
| $t_1$ | integration end time; supplied at ingest, consumed to derive $T$, not stored (recoverable as $t_0 + T$) | 3 | `t1` (input) |
| $T = t_1 - t_0$ | integration window, signed; derived at ingest from $t_0$ and the end time $t_1$, stored as a scalar coord on the `FlowMap`; its sign sets the integration direction | 3 | `T` (flow map coord) |
| $dx = R\cos\phi\,d\lambda,\ \ dy = R\,d\phi$ | metric of the sphere of radius $R$: the local east/north frame in which every separation is measured ($\lambda$ longitude, $\phi$ latitude; lon/lat $\to$ metres). A *finite* separation between two named points takes the cosine at that pair's mid-latitude, $\bar\phi = \tfrac{1}{2}(\phi_a + \phi_b)$ |  | (internal metric, `_separation_m`) |
| $\Delta\lambda \in [-180^\circ, 180^\circ]$ | longitude *difference*, wrapped; stored longitudes are never wrapped |  | `_wrap_lon` |
| $\dot r = \xi_1(r)$ | shrink line: tensor line tangent to $\xi_1$; a repelling LCS (forward flow) or, by forward–backward duality, attracting LCS (backward flow) | Table 1 ($n = 2$) | `shrink_lines`, `ftle_ridge_seeds` |
| $w$ | ridge-seed window: the diameter, in metres, of the spherical neighbourhood a seed must be the FTLE maximum over; two seed points are therefore at least $w/2$ apart |  | `window_m` |
| $\Lambda \ge \Lambda_{\min}$ | FTLE magnitude floor a seed must also clear, set either as a quantile of the field at hand or as an absolute value in the field's units (1/s) |  | `quantile`, `ftle_min` |
| $\lambda_2 / \lambda_1 \ge a_{\min}$ | anisotropy floor: the ratio of the Cauchy–Green eigenvalues below which $\xi_1$ is not a well-defined direction (dimensionless) |  | `min_anisotropy` |
| $E_\lambda(x_0)$ | generalized Green–Lagrange strain tensor (deferred) | 8 |  |
| $\eta^\pm(x_0)$ | shear vector field; stretch/shear lines (deferred) | 10, 11, Table 1 |  |

## Conventions in detail

### Three coordinate pairs

Everything positional in the package is one of exactly three lon/lat pairs, all
in degrees. They are defined here and nowhere else.

- **`lon_grid` / `lat_grid`** are the diagnostic grid points
  $x_{\mathrm{grid}}$: the locations at which $\nabla F$, $C$, its eigenpairs,
  and the FTLE are reported. A diagnostic is also plotted or selected against
  this coordinate. Every pair and both families (`SeedGrid` and `FlowMap`)
  carry it explicitly, reachable as the properties `obj.lon_grid` /
  `obj.lat_grid`. Their dims are the grid dims: `(i, j)` on `Neighbor*` and
  `Auxiliary*`, and on `UnstructuredAuxiliary*` whatever dims the caller's own
  arrays carried, or `grid_point` if those were plain 1-D arrays.
- **`lon_0` / `lat_0`** are the reference release positions $x_0$, where
  actual particles are put into the water. Their dims are the grid dims for
  the `Neighbor*` classes, and the grid dims plus `displacement`, one per
  stencil arm, for the auxiliary classes.
- **`lon` / `lat`** are the advected positions $F_{t_0}^{t_1}(x_0)$, data
  variables on a `FlowMap`, sharing the dims of `lon_0`/`lat_0` so that
  $\nabla F = \partial(\mathrm{lon}, \mathrm{lat}) / \partial(\mathrm{lon}_0,
  \mathrm{lat}_0)$ is well-defined.

For the `Neighbor*` classes the release point *is* the grid point, so `lon_grid`
equals `lon_0` and `lat_grid` equals `lat_0`. The value is stored under both
names rather than aliased, so a consumer can read `lon_grid` without knowing
which stencil produced the dataset, and the dataset stays self-sufficient. The
auxiliary arms are stored explicitly for the same reason.

The diagnostics carry `lon_grid`/`lat_grid` only and drop `lon_0`/`lat_0`. On
the auxiliary classes, the differenced position is one arm. Attaching it to a
per-grid-point result would label a quantity by a point about $s$ metres away
from the point it describes.

### Flow map vs. deformation gradient

These are two distinct objects and the code keeps the names apart:

- $F_{t_0}^{t_1}(x_0)$ (Eq. 3) is the flow map itself, a 2-component vector
  field giving the final position of a particle released at $x_0$. Its
  components are stored as the advected positions `lon` / `lat`, the only data
  variables on a `FlowMap` and the particle's actual position, as in Parcels.
  They sit alongside the reference positions $x_0$ = `lon_0` / `lat_0`,
  coordinates carried by both the seed and the flow map. A `SeedGrid` has no
  advected positions at all; they enter only at ingest. The symbol `F` /
  `flow_map` denotes the map as a whole.
- $\nabla F_{t_0}^{t_1}(x_0)$ (Eq. 4) is the gradient of that map, a
  $2\times 2$ matrix at each $x_0$, named `deformation_gradient` / `gradF`.

`F` names the map and `gradF` its gradient; the $2\times 2$ object is never
called `F`.

### Computing the deformation gradient (Eq. 9)

$\nabla F$ is estimated by finite differences of final positions with respect to
initial positions. Each column of $\nabla F$ is a centred difference of the
*advected* positions (the numerator, measured from the ingested Parcels
outputs) divided by the controlled *initial* separation (the denominator).
Both are separations in metres, each taken in the local frame of its own pair of
points (see [the local east-north frame](#the-local-east-north-frame) below).
The metric converts lon/lat separations to metres and never supplies the
advected displacement (Haller's Eq. 9 stencil). Two stencil strategies are
modelled as separate classes, and the auxiliary one comes in two layouts:

- **Neighbour stencil** (`NeighborFlowMap`): the stencil is the neighbouring
  grid points $(i\pm 1, j\pm 1)$. The stencil adds no extra dimensions. The
  diagnostic resolution and the gradient step are the same grid, which has to
  be rectilinear for the neighbours to be defined.
- **Auxiliary stencil** (`AuxiliaryFlowMap` and `UnstructuredAuxiliaryFlowMap`):
  each grid point carries a fixed four-arm stencil on a single `displacement`
  dim, `displacement = ['east', 'north', 'west', 'south']`. Each arm sits at
  $\pm s$ metres from the diagnostic grid point (`aux_separation_m`), in that
  point's own local east/north frame. The arm spans are therefore $2s$ at
  every latitude, per Haller Eq. 9. The arms are stored *explicitly* as the
  reference release positions `lon_0(i, j, displacement)` /
  `lat_0(i, j, displacement)`. The dataset is therefore self-sufficient, since
  no metric convention is needed to recover where particles started.
  $\nabla F$ is the plain $\partial(\text{lon}, \text{lat}) /
  \partial(\text{lon}_0, \text{lat}_0)$ differenced over `displacement`. The
  diagnostic grid points `lon_grid` / `lat_grid` are the arm centres and are
  kept separately. The stencil omits the diagonal corners and the centre arm,
  which would duplicate the grid position. The four-arm stencil decouples
  the gradient step from the diagnostic resolution. A grid point is also
  never differenced against another one, so the gradient step is decoupled
  from the layout of the grid points as well. `AuxiliaryFlowMap` keeps the
  grid points on `(i, j)`, and `UnstructuredAuxiliaryFlowMap` takes them as
  they come.

Cells with a missing stencil point (e.g., a lost particle arriving as NaN)
yield a NaN $\nabla F$. That NaN propagates through $C$, the eigen-analysis,
and the FTLE without special-casing.

### Cauchy–Green tensor and its eigen-decomposition (Eqs. 6–7)

$C = (\nabla F)^\top \nabla F$ (Eq. 6) is symmetric positive-definite, so its
eigenvalues are real and positive and its eigenvectors are orthonormal. The
convention is the eigenvalue ordering $0 < \lambda_1 \le \lambda_2$ with
$\xi_1 \perp \xi_2$ (Eq. 7). The eigen step is a vectorized call to
`np.linalg.eigh` over the `(row, col)` core dims.

### FTLE and the largest eigenvalue (§4.1)

$$\Lambda_{t_0}^{t_1}(x_0) = \frac{1}{|t_1 - t_0|}\,\log\sqrt{\lambda_{\max}}
= \frac{1}{|T|}\,\log\sqrt{\lambda_2}.$$

Note it is the largest eigenvalue $\lambda_{\max} = \lambda_2$ (maximum
stretching) that enters the FTLE, not the smallest. The integration time
enters as $|T|$, not as $T$. The sign of $T = t_1 - t_0$ encodes forward or
backward integration, and this package supports backward maps, which is how
attracting LCS are produced. Writing $1/(t_1 - t_0)$ would therefore flip the
sign of every backward FTLE. With $|T|$, forward and backward maps of the
same window give exponents of the same sign.

$\Lambda$ is returned in SI units, `1/s`. Convert it to per day for display in
the plotting code; the package does not convert.

### Integration time $T$

$T = t_1 - t_0$ (Eq. 3), signed. A `SeedGrid` owns no $t_0$.
Both ends of the window enter at ingest, through
`pset_to_flowmap(*, lon, lat, t0, t1)`. It takes the release time $t_0$ and
the end time $t_1$, and derives $T = t_1 - t_0$. It then stores $t_0$ and $T$
as scalar coords on the `FlowMap` ($t_1$ is recoverable as $t_0 + T$). The
package does not choose the integration direction. $\mathrm{sign}(T)$ follows
from $t_1$ relative to $t_0$. A zero window
($t_1 = t_0$) is rejected with `ValueError`, since the FTLE's $1/|T|$ would
divide by zero. A release series (sweep $t_0$ or $t_1$) is an external loop over
scalar-`(t0, T)` flow maps, assembled with `xr.concat` / `combine_by_coords`
into the $(i, j, t_0, T)$ cube. See
[`plans/timing-design.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/plans/done/timing-design.md).

### The local east-north frame

Haller's math is Cartesian and the grid is lon/lat. The package uses no shared
projection and no standard parallel. Distances are metres on a sphere of radius
$R$ = `EARTH_RADIUS_M` = 6371 km, whose metric is

$$dx = R\cos\phi\,d\lambda, \qquad dy = R\,d\phi$$

with $\lambda$ longitude and $\phi$ latitude in radians. This is the local
east/north frame at the point where it is evaluated.

A *finite* separation is between two named points, so the cosine is taken at
their mid-latitude. For points $a$ and $b$ (`_separation_m`), with the
degrees-to-radians factor $\pi/180$ written out:

$$\Delta x = R\cos\!\left(\tfrac{\phi_a + \phi_b}{2}\right)
\,\mathrm{wrap}(\lambda_b - \lambda_a)\,\tfrac{\pi}{180},
\qquad
\Delta y = R\,(\phi_b - \phi_a)\,\tfrac{\pi}{180},$$

where $\mathrm{wrap}$ (`_wrap_lon`) maps a longitude *difference* into
$[-180^\circ, 180^\circ]$ by subtracting the nearest multiple of $360^\circ$.
Every pair therefore gets its own frame. The reference
separation is measured at the mid-latitude of the reference points, and the
advected separation at the mid-latitude of the advected ones. $\nabla F$, the
ratio of the two, is then the Jacobian of the map read between the tangent
frame at $x_0$ and the tangent frame at $F(x_0)$. On a sphere no chart has a
constant tangent Jacobian, so both ends cannot be measured in one shared
frame. A rigid meridional translation changes the metres per degree of
longitude, so it comes back with $F_{xx} \ne 1$ rather than as a null
deformation.

Every flow map differences through the same `_separation_m`, and both auxiliary
seed grids place their arms by inverting the same relation at each grid point's
own latitude. An arm span is therefore $2s$ metres wherever the grid point
sits. No
separation carries a domain-size limit, and none treats the antimeridian as a
special case. Reading a field *between* grid
points is a separate matter. On `Neighbor*` and `Auxiliary*` it goes along the
`lon_grid` axis, so that axis must be monotonic. A domain crossing the
antimeridian is therefore seeded on `170, 175, 180, 185` rather than on
`170, 175, 180, -175`. On a wrapped axis `shrink_lines`, `hyperbolic_lcs`, and
`FlowMap.image` all raise `ValueError` out of SciPy. `UnstructuredAuxiliary*` has no
axis, and triangulates its grid points in degrees instead, so they must still
sit on one longitude branch. Traced tensor lines are bound by neither and may
cross the antimeridian. The mid-latitude
cosine is a midpoint rule, so the accuracy of a separation is set by the
separation itself. The error series
is in [the local east-north frame](numerics.md#the-local-east-north-frame),
together with the numerical checks.

The poles are excluded. Going the other way, from metres to degrees, is what
`AuxiliarySeedGrid.from_axes` does to place an arm. A fixed eastward offset
needs
$\Delta\lambda = \Delta x / (R\cos\phi)$, which grows without bound as
$\cos\phi \to 0$. Past $180^\circ$ it aliases through the wrap into an arm on
the far side of the pole. That arm is a wrong gradient rather than a NaN, so
both auxiliary constructors raise `ValueError` once the offset reaches
$90^\circ$ of longitude. `_step_lonlat_by_meters` divides by the same cosine and
does not fold latitude at $\pm 90^\circ$. A line stepped past the pole continues
to latitudes outside $[-90^\circ, 90^\circ]$ instead of folding over the pole.

Longitudes are stored in whatever convention the caller supplies; nothing is
normalised on ingest, and `lon_grid`, `lon_0`, and `lon` come back on the branch
they went in on. Only *differences* and *means* are wrapped. A mean is taken on
the circle by `_circular_mean_lon`, which anchors on the first element along the
averaging dim and averages the wrapped offsets from it. The result therefore
keeps the anchor's branch. `AuxiliaryFlowMap.grid_image` uses
`_circular_mean_lon` for the longitude of the
four-arm centroid.

Stepping *along* a direction, rather than differencing between two points,
inverts `_separation_m`. `tensorlines._step_lonlat_by_meters` advances
$(\lambda, \phi)$ by a local east/north vector $d$ of length `step_m` as

$$\Delta\phi = \frac{d_{\text{north}}}{R}\tfrac{180}{\pi},
\qquad
\Delta\lambda = \frac{d_{\text{east}}}
{R\,\cos\!\left(\phi + \tfrac{1}{2}\Delta\phi\right)}\tfrac{180}{\pi},$$

the same mid-latitude cosine solved for $\Delta\lambda$, so that measuring the
step with `_separation_m` returns the vector that was asked for. It is not the
great-circle endpoint of that length and bearing; why the inverse is the right
operation for a tensor line is in
[stepping a tensor line](numerics.md#stepping-a-tensor-line). The longitude
increment is added to the incoming longitude, so a traced line crossing the
antimeridian stays on its seed's branch.

### Geometric LCS layer (tensor lines)

Hyperbolic LCS are extracted as shrink lines, tensor lines tangent to
$\xi_1$ solving $\dot r = \xi_1(r)$ (Haller Table 1, $n = 2$). `shrink_lines`,
in
[`src/lcs_parcels/tensorlines.py`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/src/lcs_parcels/tensorlines.py),
integrates the shrink lines, with `ftle_ridge_seeds` for the seed
points. Repelling LCS
are the shrink lines of the forward flow map; attracting LCS those of the
backward flow map (forward–backward duality, Haller & Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)).
`FlowMap.hyperbolic_lcs()` runs the FTLE, the seeding, and the tensor lines in
one call.

The layer's tuning parameters are stated in the units of the thing itself, so
a call means the same thing at any grid resolution and over any window. The
ridge-seed window $w$ (`window_m`), the tensor-line arc step `step_m`, and the
length cap `line_length_m` are metres. The well-definedness guard
`min_anisotropy` is the dimensionless eigenvalue ratio $a_{\min}$. A line
terminates where

$$\frac{\lambda_2}{\lambda_1} < a_{\min}.$$

`line_length_m` is a cap on the traced arc, not the achieved length. A line
that terminates early is shorter. The returned block is NaN-filled past
termination, so every row has equal length.

$w$ is the *diameter* of the ridge-seed neighbourhood, which reaches $w/2$ from
its own grid point. Two seed points are therefore at least $w/2$ apart, not
$w$.
`window_m` is therefore not itself a seed separation, though the separation it implies is
exactly half of it. `ftle_ridge_seeds` reports that half as the
`min_seed_separation_m` attribute of the dataset it returns. It bounds strict
maxima, since a plateau of exactly equal values makes every one of its points a
neighbourhood maximum.

$\Lambda_{\min}$ is the magnitude floor the FTLE at a seed must also clear.
`quantile` states it as a quantile of the field at hand and is the default at
0.90. `ftle_min` states it as an absolute value in the field's units, 1/s for
`FlowMap.ftle()`. The two are mutually exclusive, and the resolved value is
reported as the `ftle_threshold` attribute whichever set it.

$a_{\min}$ carries no $T$, no grid scale, and no stretching rate. It is the
relative gap between the eigenvalues of $C$, which sets how sensitive $\xi_1$ is
to a perturbation of $C$. That sensitivity determines whether $\xi_1$ is a
direction or numerical
noise. The default is $a_{\min} = 1.15$. It is a well-definedness guard, not an LCS
selector; the selection is made by $\Lambda_{\min}$.

The following remain deferred:

- $E_\lambda(x_0)$, the generalized Green–Lagrange strain tensor (Eq. 8).
- $\eta^\pm(x_0)$, the shear vector field, with stretch and shear (elliptic) lines
  (Eqs. 10–11, Table 1).

## Array and dimension conventions

Tensors and vectors are stored as single `DataArray`s with component
dimensions, not as scalar variables (`F11, F12, …`):

| Object | Code name | Dims | Component coords |
|---|---|---|---|
| diagnostic grid points $x_{\mathrm{grid}}$ (coords; seed + flow map, every pair) | `lon_grid`, `lat_grid` | the grid dims |  |
| reference release positions $x_0$ (coords; seed + flow map) | `lon_0`, `lat_0` | the grid dims, plus `displacement` on the auxiliary classes |  |
| advected flow map $F_{t_0}^{t_1}(x_0)$ (data vars; flow map only) | `lon`, `lat` | same dims as `lon_0`/`lat_0` |  |
| flow map image of the grid points (flow map only) | `grid_image` (`lon`, `lat`) | the grid dims |  |
| release time $t_0$ / signed window $T$ (scalar coords; flow map only) | `t0`, `T` | scalar |  |
| auxiliary-grid stencil axis |  | `(displacement,)` | `displacement = ['east','north','west','south']` |
| deformation gradient $\nabla F$ | `gradF` | the grid dims plus `row, col` (set, order not contractual) | `row, col = ['x', 'y']` |
| Cauchy–Green $C$ | `C` | the grid dims plus `row, col` (set, order not contractual) | `row, col = ['x', 'y']` |
| eigenvalues $\lambda_i$ | `lambda` | the grid dims plus `eig` |  |
| eigenvectors $\xi_i$ | `xi` | the grid dims plus `comp, eig` | `comp = ['x', 'y']` |
| FTLE $\Lambda$ | `ftle` | the grid dims |  |
| shrink-line polylines $\dot r = \xi_1$ | `lon`, `lat` (in the `shrink_lines` dataset) | `(line, point)` |  |

The grid dims are the dims the diagnostic grid points carry. On
`Neighbor*` and `Auxiliary*` they are `i, j`; on `UnstructuredAuxiliary*` they
are the dims of the arrays handed to `from_points`, or `grid_point` if those
were plain 1-D arrays.
The `comp` coordinate labels vector/tensor
components `['x', 'y']`; `row`/`col` (dimension coordinates valued `['x', 'y']`)
index the two axes of a $2\times 2$ tensor; `eig` indexes the two eigenpairs.
The dims listed above are a *set*, not a memory layout. The package is
label-based throughout, so the axis order a given call happens to return is not
part of the contract and must never be relied on. Select with `.sel` / `.isel`
and named dims; call `.transpose(...)` yourself if you need a specific layout
(as the interpolators do before handing an array to SciPy). For the auxiliary
classes the reference release positions `lon_0` / `lat_0` *are* the explicit
stencil arms and (on a flow map) the advected arms `lon` / `lat` share those
dims. The diagnostic grid points `lon_grid` / `lat_grid` are kept separately.
The `displacement` dim is differenced away by `deformation_gradient`, so
$\nabla F$ and everything downstream are back on the grid dims, labelled by
`lon_grid`/`lat_grid`. A single
`FlowMap` carries scalar `t0`/`T`; assembling a release series promotes them to
extra $t_0$ / $T$ axes that broadcast on top of these.

Storing tensors with component dims keeps the eigen step compact. xarray has no
native eigendecomposition. It does not wrap `np.linalg`, and `np.linalg.eigh(C)`
would drop the dims and coords and requires the matrix axes to be last.
`cg_eigen` therefore declares the core dims and re-wraps the result:
`xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row', 'col']], ...)`.

## References

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

Haller, G. & Sapsis, T. (2011). *Lagrangian coherent structures and the smallest
finite-time Lyapunov exponent.* Chaos, 21, 023115.
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
