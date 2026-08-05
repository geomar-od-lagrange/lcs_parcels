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
| $x_0$ | initial (reference) particle *release* position; for the `Neighbor*` classes the grid point itself, for the `Auxiliary*` classes the explicit stencil arms; carried by both the seed and the flow map | 3 | `lon_0`, `lat_0` (`(i, j)`; `(i, j, displacement)` for the `Auxiliary*` classes) |
| $x_{\mathrm{grid}}$ | diagnostic grid point, where every diagnostic is reported; carried explicitly by both stencils and by both families | — | `lon_grid(i, j)`, `lat_grid(i, j)` |
| $F_{t_0}^{t_1}(x_0) = x(t_1; t_0, x_0)$ | flow **map**: initial position $\to$ position at time $t_1$; stored as the advected positions on a `FlowMap` (its only data vars) | 3 | `lon`, `lat`, sharing the dims of `lon_0`/`lat_0`: `(i, j)`; `(i, j, displacement)` for the `Auxiliary*` classes |
| $F_{t_0}^{t_1}(x_{\mathrm{grid}})$ | flow map image of the diagnostic grid points: the advected positions reduced onto `(i, j)` | 3 | `grid_image` |
| $\nabla F_{t_0}^{t_1}(x_0)$ | deformation gradient (the **gradient** of the flow map; $2\times 2$ in 2D) | 4, 9 | `deformation_gradient`, `gradF` |
| $C(x_0) = \left(\nabla F_{t_0}^{t_1}\right)^\top \nabla F_{t_0}^{t_1}$ | right Cauchy–Green strain tensor ($2\times 2$, symmetric positive-definite) | 6 | `cauchy_green`, `C` |
| $C\,\xi_i = \lambda_i\,\xi_i,\ \ 0 < \lambda_1 \le \lambda_2,\ \ \xi_1 \perp \xi_2$ | eigen-decomposition of $C$ | 7 | `cg_eigen` |
| $\lambda_1, \lambda_2$ | eigenvalues of $C$ (ordered $0 < \lambda_1 \le \lambda_2$); $\lambda_{\max} = \lambda_2$ | 7 | `lambda` (coord `eig`) |
| $\xi_1, \xi_2$ | orthonormal eigenvectors of $C$ | 7 | `xi` (coords `comp`, `eig`) |
| $\Lambda_{t_0}^{t_1}(x_0) = \dfrac{1}{\lvert t_1 - t_0\rvert}\,\log\sqrt{\lambda_{\max}}$ | finite-time Lyapunov exponent (FTLE); uses the **largest** eigenvalue, and $\lvert T\rvert$ so that a backward map ($T < 0$) gives a positive exponent | §4.1 | `ftle` |
| $t_0$ | release time; supplied at ingest (`pset_to_flowmap`) and stored as a scalar coord on the `FlowMap` | 3 | `t0` (flow map coord) |
| $t_1$ | integration end time; supplied at ingest, consumed to derive $T$, not stored (recoverable as $t_0 + T$) | 3 | `t1` (input) |
| $T = t_1 - t_0$ | integration window, **signed**; derived at ingest from $t_0$ and the end time $t_1$, stored as a scalar coord on the `FlowMap`; its sign sets the integration direction | 3 | `T` (flow map coord) |
| $dx = R\cos\phi\,d\lambda,\ \ dy = R\,d\phi$ | metric of the sphere of radius $R$: the local east/north frame in which every separation is measured ($\lambda$ longitude, $\phi$ latitude; lon/lat $\to$ metres). A *finite* separation between two named points takes the cosine at that pair's mid-latitude, $\bar\phi = \tfrac{1}{2}(\phi_a + \phi_b)$ | — | (internal metric, `_separation_m`) |
| $\Delta\lambda \in [-180^\circ, 180^\circ]$ | longitude *difference*, wrapped; stored longitudes are never wrapped | — | `_wrap_lon` |
| $\dot r = \xi_1(r)$ | shrink line: tensor line tangent to $\xi_1$; a repelling LCS (forward flow) or, by forward–backward duality, attracting LCS (backward flow) | Table 1 ($n = 2$) | `shrink_lines`, `ftle_ridge_seeds` |
| $w$ | ridge-seed window: the **side**, in metres, of the square window a seed must be the FTLE maximum over; two seed points can therefore be about $w/2$ apart | — | `window_m` |
| $\Lambda \ge \Lambda_{\min}$ | FTLE magnitude floor a seed must also clear, set either as a quantile of the field at hand or as an absolute value in the field's units (1/s) | — | `quantile`, `ftle_min` |
| $\lambda_2 / \lambda_1 \ge a_{\min}$ | anisotropy floor: the ratio of the Cauchy–Green eigenvalues below which $\xi_1$ is not a well-defined direction (dimensionless) | — | `min_anisotropy` |
| $E_\lambda(x_0)$ | generalized Green–Lagrange strain tensor (**deferred**) | 8 | — |
| $\eta^\pm(x_0)$ | shear vector field; stretch/shear lines (**deferred**) | 10, 11, Table 1 | — |

## Conventions in detail

### Three coordinate pairs

Everything positional in the package is one of exactly three lon/lat pairs, all
in degrees. They are defined here and nowhere else.

- **`lon_grid` / `lat_grid`**, dims `(i, j)` — the **diagnostic grid points**
  $x_{\mathrm{grid}}$: the locations at which $\nabla F$, $C$, its eigenpairs
  and the FTLE are reported, and the coordinate a diagnostic is plotted or
  selected against. Carried explicitly by both stencils and by both families
  (`Seed` and `FlowMap`), and reachable as the properties `obj.lon_grid` /
  `obj.lat_grid`.
- **`lon_0` / `lat_0`** — the **reference release positions** $x_0$: where
  actual particles are put into the water. `(i, j)` for the `Neighbor*` classes,
  `(i, j, displacement)` for the `Auxiliary*` classes, one per stencil arm.
- **`lon` / `lat`** — the **advected positions** $F_{t_0}^{t_1}(x_0)$, data
  variables on a `FlowMap`, sharing the dims of `lon_0`/`lat_0` so that
  $\nabla F = \partial(\mathrm{lon}, \mathrm{lat}) / \partial(\mathrm{lon}_0,
  \mathrm{lat}_0)$ is well-defined.

For the `Neighbor*` classes the release point *is* the grid point, so `lon_grid`
equals `lon_0` and `lat_grid` equals `lat_0`. The value is stored under both
names rather than aliased, so a consumer can read `lon_grid` without knowing
which stencil produced the dataset, and the dataset stays self-sufficient. The
auxiliary arms are stored explicitly for the same reason.

The diagnostics carry `lon_grid`/`lat_grid` only. `lon_0`/`lat_0` are dropped
from them, because on the `Auxiliary*` classes the differenced position is one
arm, and attaching it to an `(i, j)` result would label a quantity by a point
about $s$ metres away from the point it describes.

### Flow map vs. deformation gradient

These are two distinct objects and the code keeps the names apart:

- $F_{t_0}^{t_1}(x_0)$ (Eq. 3) is the flow **map** itself — a 2-component vector
  field giving the final position of a particle released at $x_0$. Its components
  are stored as the advected positions `lon` / `lat` (the only data variables on
  a `FlowMap`; the particle's actual position, as in Parcels), alongside the
  reference positions $x_0$ = `lon_0` / `lat_0` (coordinates, carried by both the
  seed and the flow map). A time-free `Seed` has no advected positions at all;
  they enter only at ingest. The symbol `F` / `flow_map` denotes the map as a
  whole.
- $\nabla F_{t_0}^{t_1}(x_0)$ (Eq. 4) is the **gradient** of that map — a
  $2\times 2$ matrix at each $x_0$. Code name: `deformation_gradient` / `gradF`.

`F` names the map and `gradF` its gradient; the $2\times 2$ object is never
called `F`.

### Computing the deformation gradient (Eq. 9)

$\nabla F$ is estimated by finite differences of final positions with respect to
initial positions. Each column of $\nabla F$ is a centred difference of the
*advected* positions (the **numerator**, measured from the ingested Parcels
outputs) divided by the controlled *initial* separation (the **denominator**).
Both are separations in metres, each taken in the local frame of its own pair of
points (see [the local east-north frame](#the-local-east-north-frame) below).
The metric converts lon/lat separations to metres and never supplies the
advected displacement (Haller's Eq. 9 stencil). Two stencil strategies are
modelled as separate classes:

- **Neighbour stencil** (`NeighborFlowMap`): the stencil is the neighbouring
  grid points $(i\pm 1, j\pm 1)$. No extra dimensions; the diagnostic resolution
  and the gradient step are the same grid.
- **Auxiliary stencil** (`AuxiliaryFlowMap`): each grid point carries a fixed four-arm
  stencil on a single `displacement` dim
  (`displacement = ['east', 'north', 'west', 'south']`), placed at $\pm s$ metres
  about the diagnostic grid point (`aux_separation_m`) in that point's own local
  east/north frame, so the arm spans are $2s$ at every latitude, per Haller
  Eq. 9. The arms are stored
  *explicitly* as the reference release positions
  `lon_0(i, j, displacement)` / `lat_0(i, j, displacement)` (so the dataset is
  self-sufficient — no metric convention is needed to recover where particles
  started), and $\nabla F$ is the plain $\partial(\text{lon}, \text{lat}) /
  \partial(\text{lon}_0, \text{lat}_0)$ differenced over `displacement`. The
  diagnostic grid points `lon_grid(i, j)` / `lat_grid(i, j)` are the arm centres
  and are kept separately. No centre arm (it would duplicate the grid position)
  and no diagonal corners. This decouples the gradient step from the diagnostic
  resolution.

Cells with a missing stencil point (e.g., a lost particle arriving as NaN) yield
a NaN $\nabla F$, and that NaN propagates through $C$, the eigen-analysis, and
the FTLE without special-casing.

### Cauchy–Green tensor and its eigen-decomposition (Eqs. 6–7)

$C = (\nabla F)^\top \nabla F$ (Eq. 6) is symmetric positive-definite, so its
eigenvalues are real and positive and its eigenvectors are orthonormal. The
convention is the eigenvalue ordering $0 < \lambda_1 \le \lambda_2$ with
$\xi_1 \perp \xi_2$ (Eq. 7). The eigen step is a vectorized call to
`np.linalg.eigh` over the `(row, col)` core dims.

### FTLE uses the largest eigenvalue (§4.1)

$$\Lambda_{t_0}^{t_1}(x_0) = \frac{1}{|t_1 - t_0|}\,\log\sqrt{\lambda_{\max}}
= \frac{1}{|T|}\,\log\sqrt{\lambda_2}.$$

Note it is the **largest** eigenvalue $\lambda_{\max} = \lambda_2$ (maximum
stretching) that enters the FTLE, not the smallest. The integration time enters
as $|T|$, not as $T$: the sign of $T = t_1 - t_0$ encodes forward or backward
integration, and this package supports backward maps (they are how attracting
LCS are produced), so writing $1/(t_1 - t_0)$ would flip the sign of every
backward FTLE. With $|T|$, forward and backward maps of the same window give
exponents of the same sign.

$\Lambda$ is returned in SI units, `1/s`. Convert it to per day for display in
the plotting code; the package does not convert.

### Integration time $T$

$T = t_1 - t_0$ (Eq. 3), **signed**. The `Seed` is time-free: it owns no $t_0$.
Both ends of the window enter at ingest — `pset_to_flowmap(*, lon, lat, t0, t1)`
takes the release time $t_0$ and the end time $t_1$, derives $T = t_1 - t_0$,
and stores $t_0$ and $T$ as scalar coords on the `FlowMap` ($t_1$ is recoverable
as $t_0 + T$). The package does not choose the integration direction:
$\mathrm{sign}(T)$ follows from $t_1$ relative to $t_0$. A zero window
($t_1 = t_0$) is rejected with `ValueError`, since the FTLE's $1/|T|$ would
divide by zero. A release series (sweep $t_0$ or $t_1$) is an external loop over
scalar-`(t0, T)` flow maps, assembled with `xr.concat` / `combine_by_coords`
into the $(i, j, t_0, T)$ cube. See
[`plans/timing-design.md`](../plans/timing-design.md).

### The local east-north frame

Haller's math is Cartesian and the grid is lon/lat. The package uses no shared
projection and no standard parallel. Distances are metres on a sphere of radius
$R$ = `EARTH_RADIUS_M` = 6371 km, whose metric is

$$dx = R\cos\phi\,d\lambda, \qquad dy = R\,d\phi$$

with $\lambda$ longitude and $\phi$ latitude in radians — the **local east/north
frame** at the point where it is evaluated.

A *finite* separation is between two named points, so the cosine is taken at
their **mid-latitude**. For points $a$ and $b$ (`_separation_m`), with the
degrees-to-radians factor $\pi/180$ written out:

$$\Delta x = R\cos\!\left(\tfrac{\phi_a + \phi_b}{2}\right)
\,\mathrm{wrap}(\lambda_b - \lambda_a)\,\tfrac{\pi}{180},
\qquad
\Delta y = R\,(\phi_b - \phi_a)\,\tfrac{\pi}{180},$$

where $\mathrm{wrap}$ (`_wrap_lon`) maps a longitude *difference* into
$[-180^\circ, 180^\circ]$ by subtracting the nearest multiple of $360^\circ$.
Every pair therefore gets its own frame: the reference
separation is measured at the mid-latitude of the reference points, the advected
separation at the mid-latitude of the advected ones, and $\nabla F$ is the ratio
of the two. $\nabla F$ is then the Jacobian of the map read between the tangent
frame at $x_0$ and the tangent frame at $F(x_0)$. On a sphere no chart has a
constant tangent Jacobian, so there is no single frame in which both ends could
be measured. A rigid meridional translation changes the metres per degree of
longitude, so it comes back with $F_{xx} \ne 1$ rather than as a null
deformation.

Both `NeighborFlowMap` and `AuxiliaryFlowMap` difference through the same
`_separation_m`, and `AuxiliarySeed` places its arms by inverting the same
relation at each grid point's own latitude, so an arm span is $2s$ metres
wherever the grid point sits. No separation carries a domain-size limit, and
none treats the antimeridian as a special case. The `lon_grid` **axis** is a
separate matter: `FlowMap.image` and `shrink_lines`
interpolate along it, so it must be monotonic, and a domain crossing the
antimeridian is seeded on `170, 175, 180, 185` rather than on
`170, 175, 180, -175`. On a wrapped axis `shrink_lines` and `hyperbolic_lcs`
raise `ValueError` out of SciPy; `FlowMap.image` does not raise, and reads the
axis as if it were sorted, so a query in the wrapped half comes back `NaN` or
interpolated between the wrong two grid points. Traced tensor lines are not
bound by the monotonic axis and may cross the antimeridian. The mid-latitude
cosine is a midpoint rule, so the accuracy of a separation is set by the
separation itself; the error series
is in [the local east-north frame](numerics.md#the-local-east-north-frame),
together with the numerical checks.

The poles are excluded. Going the other way, from metres to degrees, is what
`AuxiliarySeed.from_axes` does to place an arm: a fixed eastward offset needs
$\Delta\lambda = \Delta x / (R\cos\phi)$, which grows without bound as
$\cos\phi \to 0$. Past $180^\circ$ it aliases through the wrap into an arm on
the far side of the pole, which is a wrong gradient rather than a NaN, so
`AuxiliarySeed.from_axes` raises `ValueError` once the offset reaches
$90^\circ$ of longitude. `_step_lonlat_by_meters` divides by the same cosine and
does not fold latitude at $\pm 90^\circ$: a line stepped past the pole continues
to latitudes outside $[-90^\circ, 90^\circ]$ instead of folding over the pole.

Longitudes are stored in whatever convention the caller supplies; nothing is
normalised on ingest, and `lon_grid`, `lon_0` and `lon` come back on the branch
they went in on. Only *differences* and *means* are wrapped. A mean is taken on
the circle by `_circular_mean_lon`, which anchors on the first element along the
averaging dim and averages the wrapped offsets from it, so the result keeps the
anchor's branch; `AuxiliaryFlowMap.grid_image` uses it for the longitude of the
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

Hyperbolic LCS are extracted as **shrink lines** — tensor lines tangent to
$\xi_1$, solving $\dot r = \xi_1(r)$ (Haller Table 1, $n = 2$) — in
[`src/lcs_parcels/tensorlines.py`](../src/lcs_parcels/tensorlines.py)
(`shrink_lines`, with `ftle_ridge_seeds` for the seed points). Repelling LCS
are the shrink lines of the forward flow map; attracting LCS those of the
backward flow map (forward–backward duality, Haller & Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)).
`FlowMap.hyperbolic_lcs()` runs the FTLE, the seeding and the tensor lines in
one call.

The layer's tuning parameters are stated in the units of the thing itself, so
that a call means the same thing at any grid resolution and over any window: the
ridge-seed window $w$ (`window_m`), the tensor-line arc step `step_m` and the
length cap `line_length_m` are metres, and the well-definedness guard
`min_anisotropy` is the dimensionless eigenvalue ratio $a_{\min}$. A line
terminates where

$$\frac{\lambda_2}{\lambda_1} < a_{\min}.$$

`line_length_m` is a cap on the traced arc, not the achieved length: a line that
terminates early is shorter, and the returned block is NaN-filled past
termination so every row has equal length.

$w$ is the *side* of the ridge-seed window, which reaches $w/2$ to either side
of its own grid point, so two seed points can be about $w/2$ apart, not $w$.
`window_m` never denotes a seed separation. `ftle_ridge_seeds` reports the
separation its window implies on the grid it was given as the
`min_seed_separation_m` attribute of the dataset it returns, alongside the odd
per-dimension cell counts $w$ was rounded to and the
median grid spacings that rounding used. The reported separation is taken off
the grid's *smallest* cell rather than its median one, so it is a floor; it
bounds strict maxima, since a plateau of exactly equal values makes every one of
its cells a windowed maximum.

$\Lambda_{\min}$ is the magnitude floor the FTLE at a seed must also clear.
`quantile` states it as a quantile of the field at hand and is the default at
0.90; `ftle_min` states it as an absolute value in the field's units, 1/s for
`FlowMap.ftle()`. The two are mutually exclusive, and the resolved value is
reported as the `ftle_threshold` attribute whichever set it.

$a_{\min}$ carries no $T$, no grid scale and no stretching rate: it is the
relative gap between the eigenvalues of $C$, which sets how sensitive $\xi_1$ is
to a perturbation of $C$, and so whether $\xi_1$ is a direction or numerical
noise. Default $a_{\min} = 1.15$. It is a well-definedness guard, not an LCS
selector; the selection is made by $\Lambda_{\min}$.

The following remain deferred:

- $E_\lambda(x_0)$ — generalized Green–Lagrange strain tensor (Eq. 8).
- $\eta^\pm(x_0)$ — shear vector field; stretch and shear (elliptic) lines
  (Eqs. 10–11, Table 1).

## Array and dimension conventions

Tensors and vectors are stored as **single** `DataArray`s with component
dimensions, not as scalar variables (`F11, F12, …`):

| Object | Code name | Dims | Component coords |
|---|---|---|---|
| diagnostic grid points $x_{\mathrm{grid}}$ (coords; seed + flow map, both stencils) | `lon_grid`, `lat_grid` | `(i, j)` | — |
| reference release positions $x_0$ (coords; seed + flow map) | `lon_0`, `lat_0` | `(i, j)`; `(i, j, displacement)` for the `Auxiliary*` classes | — |
| advected flow map $F_{t_0}^{t_1}(x_0)$ (data vars; flow map only) | `lon`, `lat` | same dims as `lon_0`/`lat_0` | — |
| flow map image of the grid points (flow map only) | `grid_image` (`lon`, `lat`) | `(i, j)` | — |
| release time $t_0$ / signed window $T$ (scalar coords; flow map only) | `t0`, `T` | scalar | — |
| auxiliary-grid stencil axis | — | `(displacement,)` | `displacement = ['east','north','west','south']` |
| deformation gradient $\nabla F$ | `gradF` | `i, j, row, col` (set, order not contractual) | `row, col = ['x', 'y']` |
| Cauchy–Green $C$ | `C` | `i, j, row, col` (set, order not contractual) | `row, col = ['x', 'y']` |
| eigenvalues $\lambda_i$ | `lambda` | `(i, j, eig)` | — |
| eigenvectors $\xi_i$ | `xi` | `(i, j, comp, eig)` | `comp = ['x', 'y']` |
| FTLE $\Lambda$ | `ftle` | `(i, j)` | — |
| shrink-line polylines $\dot r = \xi_1$ | `lon`, `lat` (in the `shrink_lines` dataset) | `(line, point)` | — |

Logical grid dims are `i, j`. The `comp` coordinate labels vector/tensor
components `['x', 'y']`; `row`/`col` (dimension coordinates valued `['x', 'y']`)
index the two axes of a $2\times 2$ tensor; `eig` indexes the two eigenpairs.
The dims listed above are a *set*, not a memory layout: the package is
label-based throughout, so the axis order a given call happens to return is not
part of the contract and must never be relied on. Select with `.sel` / `.isel`
and named dims; call `.transpose(...)` yourself if you need a specific layout
(as `shrink_lines` does before handing the tensor to SciPy). For
the `Auxiliary*` classes the reference release positions
`lon_0(i, j, displacement)` / `lat_0(i, j, displacement)` *are* the explicit
stencil arms and (on a flow map) the advected arms `lon(i, j, displacement)` /
`lat(i, j, displacement)` share those dims; the diagnostic grid points
`lon_grid(i, j)` / `lat_grid(i, j)` are kept separately. The `displacement` dim
is differenced away by `deformation_gradient`, so $\nabla F$ and everything
downstream are back on `(i, j)`, labelled by `lon_grid`/`lat_grid`. A single
`FlowMap` carries scalar `t0`/`T`; assembling a release series promotes them to
extra $t_0$ / $T$ axes that broadcast on top of these.

Storing tensors with component dims keeps the eigen step compact. xarray has no
native eigendecomposition: it does not wrap `np.linalg`, and `np.linalg.eigh(C)`
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
