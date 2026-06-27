# Notation reference

This document is the single source of truth for the symbols and conventions used
throughout `lcs_parcels`. It follows Haller (2015), *Lagrangian Coherent
Structures*, Annu. Rev. Fluid Mech. 47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
Each symbol is tied to its Haller equation number and to the name it carries in
the code. Math is written in LaTeX; equation numbers refer to Haller (2015).

## Symbol table

| Symbol | Meaning | Haller Eq. | Code name |
|---|---|---|---|
| $v(x, t)$ | velocity field, with position $x = (x^1, x^2)$ in 2D | 2 | (input, external) |
| $x_0$ | initial (reference) particle *release* position at time $t_0$; for `NeighborGrid` the grid point itself, for `AuxiliaryGrid` the explicit stencil arms | 3 | `lon_0`, `lat_0` (`(i, j)`; `(i, j, displacement)` for `AuxiliaryGrid`) |
| — | grid-point centres where diagnostics are reported (`AuxiliaryGrid` only) | — | `lon_c(i, j)`, `lat_c(i, j)` |
| $F_{t_0}^{t}(x_0) = x(t; t_0, x_0)$ | flow **map**: initial position $\to$ position at time $t$; stored as the advected positions $x(t_1)$ | 3 | `lon(i, j)`, `lat(i, j)` |
| $\nabla F_{t_0}^{t_1}(x_0)$ | deformation gradient (the **gradient** of the flow map; $2\times 2$ in 2D) | 4, 9 | `deformation_gradient`, `gradF` |
| $C(x_0) = \left(\nabla F_{t_0}^{t_1}\right)^\top \nabla F_{t_0}^{t_1}$ | right Cauchy–Green strain tensor ($2\times 2$, symmetric positive-definite) | 6 | `cauchy_green`, `C` |
| $C\,\xi_i = \lambda_i\,\xi_i,\ \ 0 < \lambda_1 \le \lambda_2,\ \ \xi_1 \perp \xi_2$ | eigen-decomposition of $C$ | 7 | `cg_eigen` |
| $\lambda_1, \lambda_2$ | eigenvalues of $C$ (ordered $0 < \lambda_1 \le \lambda_2$); $\lambda_{\max} = \lambda_2$ | 7 | `lambda` (coord `eig`) |
| $\xi_1, \xi_2$ | orthonormal eigenvectors of $C$ | 7 | `xi` (coords `comp`, `eig`) |
| $\Lambda_{t_0}^{t_1}(x_0) = \dfrac{1}{t_1 - t_0}\,\log\sqrt{\lambda_{\max}}$ | finite-time Lyapunov exponent (FTLE); uses the **largest** eigenvalue | §4.1 | `ftle` |
| $t_0$ | release time of the seed grid; recorded by `from_axes` | 3 | `t0` |
| $t_1$ | integration end time; supplied at ingest, consumed to derive $T$, not stored | 3 | `t1` (input) |
| $T = t_1 - t_0$ | integration window, **signed**; derived at ingest from the seed's $t_0$ and the end time $t_1$, its sign sets the integration direction | 3 | `T` |
| $dx = R\cos\phi_{\mathrm{ref}}\,d\lambda,\ \ dy = R\,d\phi$ | local-tangent meters convention, anchored at one grid reference latitude $\phi_{\mathrm{ref}}$ ($\lambda$ longitude, $\phi$ latitude; lon/lat $\to$ meters) | — | (internal metric) |
| $E_\lambda(x_0)$ | generalized Green–Lagrange strain tensor (**deferred**) | 8 | — |
| $\eta^\pm(x_0)$ | shear vector field; shrink/stretch/shear lines (**deferred**) | 10, 11, Table 1 | — |

## Notes on the subtle points

### Flow map vs. deformation gradient

These are two distinct objects and the code keeps the names apart:

- $F_{t_0}^{t}(x_0)$ (Eq. 3) is the flow **map** itself — a 2-component vector
  field giving the final position of a particle released at $x_0$. Its components
  are stored as the advected positions `lon` / `lat` (data variables; the
  particle's actual position, as in Parcels), alongside the reference positions
  $x_0$ = `lon_0` / `lat_0` (coordinates); a seed grid sets them equal
  ($F_{t_0}^{t_0}(x_0) = x_0$). The symbol `F` / `flow_map` denotes the map as a
  whole (reserved to contrast with the gradient `gradF`).
- $\nabla F_{t_0}^{t_1}(x_0)$ (Eq. 4) is the **gradient** of that map — a
  $2\times 2$ matrix at each $x_0$. Code name: `deformation_gradient` / `gradF`.

Earlier sketches labeled the $2\times 2$ object simply "F"; that is really
$\nabla F$. We reserve `F` for the map and `gradF` for its gradient.

### Computing the deformation gradient (Eq. 9)

$\nabla F$ is estimated by finite differences of final positions with respect to
initial positions. Each column of $\nabla F$ is a centered difference of the
*advected* positions (the **numerator**, measured from the ingested Parcels
outputs) divided by the controlled *initial* separation in meters (the
**denominator**, from the metric below). The metric only converts lon/lat
separations to meters; it never supplies the advected displacement (Haller's
Eq. 9 stencil). Two stencil strategies are modeled as separate classes, both
first-class:

- **Neighbor differencing** (`NeighborGrid`): the stencil is the neighboring grid
  points $(i\pm 1, j\pm 1)$. No extra dimensions; the diagnostic resolution and
  the gradient step are the same grid.
- **Auxiliary grid** (`AuxiliaryGrid`): each grid point carries a fixed four-arm
  stencil on a single `displacement` dim
  (`displacement = ['east', 'north', 'west', 'south']`), placed at $\pm s$ meters
  about the centre (`aux_separation_m`), per Haller Eq. 9. The arms are stored
  *explicitly* as the reference release positions
  `lon_0(i, j, displacement)` / `lat_0(i, j, displacement)` (so the dataset is
  self-sufficient — no metric convention is needed to recover where particles
  started), and $\nabla F$ is the plain $\partial(\text{lon}, \text{lat}) /
  \partial(\text{lon}_0, \text{lat}_0)$ differenced over `displacement`. The
  grid-point centres `lon_c(i, j)` / `lat_c(i, j)` (where diagnostics are
  reported) are kept separately. No center arm (it would duplicate the grid
  position) and no diagonal corners. This decouples the gradient step from the
  diagnostic resolution.

Cells with a missing stencil point (e.g. a lost particle arriving as NaN) yield
a NaN $\nabla F$, and that NaN propagates through $C$, the eigen-analysis, and
the FTLE without special-casing.

### Cauchy–Green tensor and its eigen-decomposition (Eqs. 6–7)

$C = (\nabla F)^\top \nabla F$ (Eq. 6) is symmetric positive-definite, so its
eigenvalues are real and positive and its eigenvectors are orthonormal. The
convention is the eigenvalue ordering $0 < \lambda_1 \le \lambda_2$ with
$\xi_1 \perp \xi_2$ (Eq. 7). The eigen step is a vectorized call to
`np.linalg.eigh` over the `(row, col)` core dims.

### FTLE uses the largest eigenvalue (§4.1)

$$\Lambda_{t_0}^{t_1}(x_0) = \frac{1}{t_1 - t_0}\,\log\sqrt{\lambda_{\max}}
= \frac{1}{|T|}\,\log\sqrt{\lambda_2}.$$

Note it is the **largest** eigenvalue $\lambda_{\max} = \lambda_2$ (maximum
stretching) that enters the FTLE, not the smallest. The integration time enters
as $|T|$; the sign of $T = t_1 - t_0$ encodes forward vs. backward integration,
and this diagnostic uses only $|T|$.

### Integration time $T$

$T = t_1 - t_0$ (Eq. 3), **signed**. The seed grid owns its release time $t_0$
(recorded by `from_axes`); the particle set is emitted, advected externally, and
re-ingested with only the end time $t_1$, from which $T$ is derived and stored.
This package never chooses the direction — $\operatorname{sign}(T)$ follows from
$t_1$ relative to $t_0$. Multiple release times $t_0$ and multiple integration
windows $T$ are simply extra broadcast dimensions handled by xarray. See
[`plans/timing-design.md`](../plans/timing-design.md).

### Local-tangent meters convention

Haller's math is Cartesian, but the grid is lon/lat. Positions are converted to a
**single** local tangent frame in meters before differencing, anchored at the
grid centroid — the one reference point
$\lambda_{\mathrm{ref}} = \overline{\lambda_0}$, $\phi_{\mathrm{ref}} = \overline{\phi_0}$
(the means of `lon_0`/`lat_0`):

$$X = R\cos\phi_{\mathrm{ref}}\,(\lambda - \lambda_{\mathrm{ref}})\,\tfrac{\pi}{180},
\qquad Y = R\,(\phi - \phi_{\mathrm{ref}})\,\tfrac{\pi}{180},$$

with $R$ the Earth radius, $\phi$ latitude, $\lambda$ longitude. The cosine
factor uses that **one** grid reference latitude $\phi_{\mathrm{ref}}$ for every
point, not a per-point $\cos\phi$: a single shared frame is what makes the
off-diagonal $\nabla F$ terms exact — a per-point cosine would corrupt them by a
cosine ratio. In the tiny-separation regime this flat-tangent approximation is
adequate; it is a convention, not a correctness blocker. Both `NeighborGrid` and
`AuxiliaryGrid` share the identical metric code, so positions are read back in the
same frame the seed was emitted in.

### Deferred symbols

These belong to the geometric LCS layer (tensor-line integration) deferred to a
later module and are listed only for completeness:

- $E_\lambda(x_0)$ — generalized Green–Lagrange strain tensor (Eq. 8).
- $\eta^\pm(x_0)$ — shear vector field; shrink, stretch, and shear lines
  (Eqs. 10–11, Table 1).

## Array and dimension conventions

Tensors and vectors are stored as **single** `DataArray`s with component
dimensions, not as scalar variables (`F11, F12, …`):

| Object | Code name | Dims | Component coords |
|---|---|---|---|
| reference release positions $x_0$ (coords) | `lon_0`, `lat_0` | `(i, j)`; `(i, j, displacement)` for `AuxiliaryGrid` | — |
| advected flow map $F_{t_0}^{t_1}(x_0)$ (data vars) | `lon`, `lat` | same dims as `lon_0`/`lat_0` | — |
| grid-point centres (`AuxiliaryGrid` only, coords) | `lon_c`, `lat_c` | `(i, j)` | — |
| auxiliary-grid stencil axis | — | `(displacement,)` | `displacement = ['east','north','west','south']` |
| deformation gradient $\nabla F$ | `gradF` | `(i, j, row, col)` | `row, col = ['x', 'y']` |
| Cauchy–Green $C$ | `C` | `(i, j, row, col)` | `row, col = ['x', 'y']` |
| eigenvalues $\lambda_i$ | `lambda` | `(i, j, eig)` | — |
| eigenvectors $\xi_i$ | `xi` | `(i, j, comp, eig)` | `comp = ['x', 'y']` |
| FTLE $\Lambda$ | `ftle` | `(i, j)` | — |

Logical grid dims are `i, j`. The `comp` coordinate labels vector/tensor
components `['x', 'y']`; `row`/`col` (dimension coordinates valued `['x', 'y']`)
index the two axes of a $2\times 2$ tensor; `eig` indexes the two eigenpairs. For
`AuxiliaryGrid` the reference release positions
`lon_0(i, j, displacement)` / `lat_0(i, j, displacement)` *are* the explicit
stencil arms and the advected arms `lon(i, j, displacement)` /
`lat(i, j, displacement)` share those dims; the diagnostic centres
`lon_c(i, j)` / `lat_c(i, j)` are kept separately. The `displacement` dim is
differenced away by `deformation_gradient`, so $\nabla F$ and everything
downstream are back on `(i, j)`. Extra release-time / integration-time axes
broadcast on top of these.

Storing tensors with component dims keeps the eigen step compact. Note that
xarray has no native eigendecomposition: it does not wrap `np.linalg`, so
`np.linalg.eigh(C)` would drop the dims/coords and return bare arrays, and it
requires the matrix axes to be last. Two label-preserving options:

- `xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row', 'col']], ...)` —
  declares the core dims and re-wraps the result.
- a closed-form $2\times 2$ symmetric solver in pure xarray arithmetic
  (eigenvalues from trace/determinant via the quadratic formula, then
  eigenvectors). This needs no `apply_ufunc` and stays fully label-native and
  dask-lazy.

The choice is left to the implementation session.

## Reference

Haller, G. (2015). *Lagrangian Coherent Structures.* Annual Review of Fluid
Mechanics, 47, 137–162.
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
