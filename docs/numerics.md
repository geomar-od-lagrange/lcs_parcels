# Numerics

The quantitative half of the design record: the frame separations are measured
in, the longitude arithmetic that goes with it, the units the tuning parameters
are stated in, and the guard that decides when $\xi_1$ is a direction rather
than noise. The companion document
[`docs/architecture.md`](architecture.md) covers the other
half, what the types are and how they compose. Symbols and the definitions of
every quantity used below live in [`docs/notation.md`](notation.md); notation
follows Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## The local east-north frame

There is no projection and no shared frame. A separation is only ever taken
between two named points, and it is taken in the local east/north frame of that
pair. `_separation_m(lon_a=, lat_a=, lon_b=, lat_b=)` returns, for points
$a = (\lambda_a, \phi_a)$ and $b = (\lambda_b, \phi_b)$ in degrees,

$$\Delta x = R \cos\bar\phi \, \mathrm{wrap}(\lambda_b - \lambda_a)
  \tfrac{\pi}{180}, \qquad
\Delta y = R\,(\phi_b - \phi_a)\tfrac{\pi}{180},
\qquad \bar\phi = \tfrac{1}{2}(\phi_a + \phi_b),$$

with $R$ the mean Earth radius and $\mathrm{wrap}$ the longitude-difference wrap
of the next section. The cosine is the pair's own **mid-latitude** cosine. No
other latitude enters, so no point in the domain is measured against a parallel
outside its own pair.

$\nabla F$ is a ratio of two such separations, the advected pair over the
reference pair, and each is taken with its own cosine: the denominator with the
mid-latitude of the two release points, the numerator with the mid-latitude of
the two arrival points. That is what makes the result the tangent-space Jacobian
of the flow map, written in the orthonormal east/north basis at $x_0$ on the
input side and in the orthonormal east/north basis at $F(x_0)$ on the output
side. Both bases are orthonormal, so $C = (\nabla F)^\top \nabla F$, its
eigenvalues and the FTLE are the geometric quantities Haller's equations refer
to, at any latitude and across the antimeridian.

Against the naive degrees-space Jacobian
$\partial(\Lambda, \Phi) / \partial(\lambda, \phi)$, arrival lon/lat
differentiated with respect to release lon/lat, both in degrees, the same
statement reads

$$\nabla F = \mathrm{diag}(\cos\Phi,\, 1)\;
  \frac{\partial(\Lambda, \Phi)}{\partial(\lambda, \phi)}\;
  \mathrm{diag}\!\left(\tfrac{1}{\cos\phi},\, 1\right),$$

$R$ and the degree conversion cancelling. The two diagonal factors are the
metric of the sphere at the two ends: they convert an input in degrees to metres
at the release latitude $\phi$ and an output in metres back from degrees at the
arrival latitude $\Phi$. Both are latitudes of the stencil itself. No third
latitude enters, and so no property of the domain enters.

The rigid meridional translation is the case that makes the difference concrete.
Shift every particle north by a fixed number of degrees: the degrees-space
Jacobian is the identity, but the physical map is not an isometry, since a zonal
separation held at fixed $\Delta\lambda$ contracts by $\cos\Phi / \cos\phi$, and
$\nabla F = \mathrm{diag}(\cos\Phi/\cos\phi,\, 1)$ reports exactly that
contraction. The local frame replaced an earlier equirectangular frame whose
single standard parallel sat at the seed centroid. That earlier frame reported
the identity in this case, and its error grew with meridional excursion, which
restricted the package to regional domains of modest latitude range; GitHub
issue #18.

The remaining approximation is the finite arc. $\Delta y$ carries none of it:
$R\,\Delta\phi$ is the meridional geodesic exactly. $\Delta x$ is the arc along
the parallel through the pair's mid-latitude, evaluating a cosine at the
midpoint of the interval it is applied across. Both of the errors that
introduces are second-order in the **separation of the pair**, not in the size
of the domain, its latitude range, or its position.

For a zonal pair at latitude $\phi$ spanning $\Delta\lambda$ radians the
mid-latitude is the pair's own latitude, so what is left is the parallel arc
against the geodesic joining the two points, longer by a relative

$$\frac{(\Delta\lambda \sin\phi)^2}{24}.$$

The term vanishes on the equator, where the parallel *is* a great circle, and
grows as $\tan\phi$ at fixed separation in metres. Measured against the
great-circle distance, the separation at which it reaches $10^{-6}$ is 54 km at
30 N, 18 km at 60 N and 5.5 km at 80 N; $10^{-4}$ needs 541 km, 180 km and
55 km. For a pair separated in latitude as well, the midpoint value $\cos\bar\phi$
stands in for the interval mean $(\sin\phi_b - \sin\phi_a)/(\phi_b - \phi_a)$ and
is high by $\Delta\phi^2/24$, again with $\Delta\phi$ the pair's own span.

The default 1 km auxiliary arms are far inside every one of those limits away
from the pole; the zonal limit falls to 1 km itself only above about 88 N. A
larger
`aux_separation_m`, or the neighbour stencil, should be read off the series at
the span actually used. The neighbour span is not a tunable and it is not one
grid cell either: `_central_separation_m` differences the $i + 1$ neighbour
against the $i - 1$ one, so the span is **two** cells, and the series is read at
twice the grid spacing. For a neighbour stencil on a coarse grid the
finite-difference truncation of that same two-cell span is the larger term.

The mid-latitude rule is *exact* for a map that is linear in one tangent frame,
so both stencils reproduce the analytic answer of the synthetic test flow with
no metric error left. Worst case over the three regions the suite runs (`reference`,
`antimeridian`, `high_latitude`), the largest absolute deviation of any
$\nabla F$ component from the analytic value is 1.5e-14 for the neighbour
stencil, about 20 ulp on components of order 3, which is round-off, and 3.0e-12
for the auxiliary one. The auxiliary figure is some 4500 ulp, too large for
round-off; it is cancellation in differencing arms a kilometre apart on a sphere
6371 km across.

## Wrapping differences, never positions

`_wrap_lon` wraps a longitude **difference** into $[-180, 180]$, as
`dlon - 360 * round(dlon / 360)`. It is applied to differences and to the
offsets a circular mean is built from, and to nothing else: longitudes are
stored exactly as the caller handed them over, and the advected positions come
back on whatever branch Parcels returned.

Subtracting the nearest multiple of 360 rather than shifting by 180 and taking a
modulo is a precision choice. The shift-and-modulo form
`-((180 - dlon) % 360 - 180)` is not the identity on an argument already inside
the range: swept over $(-180, 180)$ it departs from its input by up to 2.8e-14
degrees. The `round` form returns such an argument bit-for-bit (measured 0.0),
which is what the metre-scale stencil differences need, since every one of them
is a wrap of a quantity already far inside the range.

Normalising stored positions would have to pick a branch, and any branch has a
cut that some domain straddles. A seed grid running 350 to 370 degrees east
would come back torn into two pieces at 0, and its `lon_grid` axis would stop
being monotone, which `FlowMap.image` needs for its interpolation. Wrapping the
difference has no such choice to make: $\mathrm{wrap}(\lambda_b - \lambda_a)$ is
the shorter of the two ways round for any pair less than 180 degrees apart, in
any convention, and every pair the package differences (opposite stencil arms,
adjacent grid points, and their advected images) is far inside that bound.

A mean needs more than a difference, because averaging longitudes across the cut
is not a difference operation. `_circular_mean_lon(lon, dim)` anchors on the
first element along `dim` and averages the wrapped offsets from it,

$$\bar\lambda = \lambda_{\mathrm{anchor}}
  + \overline{\mathrm{wrap}(\lambda - \lambda_{\mathrm{anchor}})},$$

which is the circular mean for a set spanning much less than a hemisphere and
which returns a value on the anchor's branch, so the mean of four auxiliary arms
inherits the convention of the arm positions rather than imposing one.
`AuxiliaryFlowMap.grid_image` is the one caller: the four advected arms it
averages can straddle the antimeridian. `skipna=False`, so a lost arm makes the
grid point NaN, matching the gradient path.

`FlowMap.image` needs the same treatment for a different reason. It interpolates
the advected longitude *field*, and an advection that hands positions back
wrapped to $[-180, 180)$ gives that field a tear: two adjacent grid points read
179.9 and -179.9, and a linear interpolant between them traverses 40 000 km.
Instead of normalising, `image` re-anchors each advected longitude on the branch
of the grid point it came from,

$$\lambda_{\mathrm{grid}} +
  \mathrm{wrap}(\lambda_{\mathrm{advected}} - \lambda_{\mathrm{grid}}),$$

which is a difference operation again and so has no branch to choose. It is
correct whenever the displacement over the window is under 180 degrees. The
interpolated result therefore comes back on the branch `lon_0` was given in.

## Stepping a tensor line

`_step_lonlat_by_meters` advances a shrink line by `step_m` along a direction
that is a local east/north vector at the current point, the same frame $C$ and
its eigenvectors were built in. It is written as the **exact inverse of
`_separation_m`**: the northward component gives $R\,\Delta\phi$, and the
eastward one is divided by $R\cos(\phi + \tfrac{1}{2}\Delta\phi)$, the same
mid-latitude cosine, solved for $\Delta\lambda$. Measuring the step afterwards
with `_separation_m` returns the vector that was asked for. The increment is
added to the incoming longitude, so a line crossing the antimeridian stays on
the branch its seed came in on. That is the rule of the previous section,
applied to a step rather than to a difference.

The step it replaced divided the eastward component by one reference cosine
$\cos\phi_{\mathrm{ref}}$ for the whole field. That is the standard-parallel
error again, but compounded: a tensor line is hundreds of steps long and the
error is systematic, so it does not average out. It bends the traced curve away
from the direction $\xi_1$ that was integrated.

An intermediate version solved the *direct great-circle problem* instead,
angular distance $\lVert d\rVert / R$ and bearing
$\mathrm{atan2}(d_{\mathrm{east}}, d_{\mathrm{north}})$, then the standard
formulae for the endpoint. That gives a more accurate arc, but the integrator is
solving $\dot r = \xi_1(r)$, where a step has to stay on the direction field. A
great-circle arc launched due east turns poleward, leaving a heading-invariant
field at a rate
$(\text{step}/R)^2 \tan\phi / 2$ per step, which accumulates *linearly* in the
step count rather than cancelling. Measured on a due-east field traced 800 km:

| Latitude | step 20 km | 10 km | 5 km |
|---|---|---|---|
| 0 N | 0.0 m | 0.0 m | 0.0 m |
| 45 N | 1255 m | 628 m | 314 m |
| 70 N | 3447 m | 1724 m | 862 m |

Halving the step halves the drift, since the per-step rate falls by four and the
step count doubles. The inverse step measures 0.0 m in every cell of that table.
That table is a due-east field, and the inverse step is exact for any field
aligned with a parallel or a meridian, at any step size and any latitude. On
other headings its $\cos\bar\phi$ is a midpoint rule like the measurement's, so
the error is second order in the step rather than absent: on a 45-degree heading
at 60 N it is 0.04 m at a 25 km step and 2.6 m at 100 km.

Three truncations remain in a traced line, none of them metric. The midpoint
scheme is second-order in `step_m` along the direction field.
`RegularGridInterpolator` reads $C$ between grid points linearly, so the field
the line follows is piecewise-linear in the grid spacing. $\nabla F$ underneath
it carries the finite-difference truncation of its own stencil. The
last two are properties of the tensor field the line is traced through, and
shrinking `step_m` does not reduce them.

## Why the tuning parameters are stated in their own units

The tunable quantities of the tensor-line layer are stated in the units of the
thing itself (`window_m`, `step_m` and `line_length_m` in metres,
`min_anisotropy` as a dimensionless eigenvalue ratio) and converted internally
against the field's own grid spacing, so the same call means the same thing at
any resolution and over any window. Stated the way they are implemented, the
two lengths would each depend on something the caller is not asking about: a
window as a cell count depends on grid resolution, and a line as a step
count depends on the step size.

The magnitude floor on ridge selection comes in both forms. `quantile`, the
default, is relative to the field it is handed, so it means the same thing on
any field. `ftle_min` is an absolute value in the field's own units, and it does
not mean the same thing on any field: a rate that marks a ridge in a fast flow
marks nothing in a slow one. It exists because that is the property a run
comparing windows or regions needs, one threshold across all of them, which a
quantile cannot express. The reasoning behind offering both is in
[`architecture.md`](architecture.md#why-an-absolute-ftle-floor-exists-at-all).

Both lengths are budgets rather than achieved quantities, and both invite the
same misreading. `line_length_m` bounds the traced arc: the integrator spends at
most `line_length_m / (2 * step_m)` steps per direction and a line that
terminates earlier is shorter, with the returned block NaN-filled past
termination so every row has equal length. `window_m` is the *side* of the
window a seed must be the maximum over, which reaches only
`window_m / 2` to either side of its own grid point, so two seeds can sit about
`window_m / 2` apart.

The seed spacing is reported rather than left to that `window_m / 2` estimate,
as `min_seed_separation_m` on the returned dataset. The window is a count of
*cells*, so the distance it corresponds to is the cell size times the count, and
the cell size is not one number: the reported value takes the **smallest** cell
on the grid, since that is where two seeds get closest. Taking the median instead
overstates the floor by 11% over a 30-degree band and by a factor of 3 over 75
degrees, both measured. The bound holds for strict local maxima; selection is
`ftle >= rolling max`, so every cell of a plateau of exactly equal values ties
and adjacent cells can all be seeds.

## Why the well-definedness guard is an eigenvalue ratio

`min_anisotropy` floors $\lambda_2 / \lambda_1$. It is the third form the guard
has taken. The two rejected forms are recorded here because each was defended on
grounds the replacement also has to answer.

A raw $\lambda_2$ floor was rejected on the argument that it "silently retunes
as the window changes". That is true: $\lambda_2$ grows exponentially in $|T|$,
so a fixed floor tightens as the window lengthens, and a guard on
well-definedness becomes a selector. A stretching-rate floor,
`ftle_min_per_day`, cured that by dividing out $|T|$, but it retunes across flow
regimes instead, which is the direction that bites in practice. The 0.005/day
default was calibrated to the mesoscale ocean at a 7-day window.

The comparison below is a paper exercise across three flow regimes, and it rests
on one premise: each regime is integrated over a window $|T|$ *comparable to its
own* $1/\mathrm{FTLE}$, the window being chosen so the flow has time to stretch
by roughly one e-fold. That fixes the three $(\mathrm{FTLE}, |T|)$ pairs:

| Regime | FTLE | $\lvert T\rvert$ |
|---|---|---|
| fast laboratory flow | 6 /day | 6 h |
| mesoscale ocean (the calibration point) | 0.15 /day | 7 d |
| slow large-scale flow | 0.011 /day | 180 d |

Measured against each flow's own FTLE signal, the *fixed rate* 0.005/day is
$0.005/6 = 0.08\%$ for the laboratory flow and $0.005/0.011 = 45.5\%$ for the
slow one, where it would eat nearly half the field. The *ratio* form converts to
a rate through the window: a floor $a_{\min}$ on $\lambda_2/\lambda_1$ is, for
incompressible flow ($\lambda_1\lambda_2 = 1$), a $\lambda_2$ floor of
$\sqrt{a_{\min}}$ and hence an equivalent rate
$\tfrac{1}{|T|}\log\sqrt{\sqrt{a_{\min}}} = \tfrac{1}{4|T|}\log a_{\min}$. With
$a_{\min} = 1.15$ that is 0.0050/day at $|T| = 7$ d (the calibration identity),
0.140/day at 6 h and $1.9\times 10^{-4}$/day at 180 d. As fractions of the two
outer flows' FTLE, that is 2.3% and 1.8%. The first argument named the right failure
mode, but its fix was too narrow: dividing out $|T|$ makes a floor scale-free in
the window and in nothing else.

The ratio retunes with neither the window nor the flow regime, and it is also
the physically correct quantity. The sensitivity of an eigenvector of $C$
to a perturbation of $C$ scales as the inverse of the *relative* gap between the
eigenvalues, so $\lambda_2 / \lambda_1$ decides whether $\xi_1$ is a direction
or numerical noise, and no stretching rate can stand in for it.
Measured: at the default 1.15 a 1% error in $C$ swings $\xi_1$ by about 2
degrees; at a ratio of 1.05 by 6 degrees; by a ratio of 4 it has flattened out
at about 0.25 degrees.

The default 1.15 is the old `ftle_min_per_day=0.005` behaviour carried over
essentially exactly: at the 7-day window of the examples that rate corresponded
to a $\lambda_2$ floor of 1.0725, which for incompressible flow
($\lambda_1 \lambda_2 = 1$) is a ratio of 1.15. The change of quantity therefore
has no effect on the tuning at the calibration point. Its effects appear away
from that point.

Away from the calibration point the two floors differ measurably, because the
ocean surface is not incompressible. Measured on the Cabo Verde example (5-day
window, points taken 34 km clear of any coast), the flow map's areal factor
$\det \nabla F$ has a median of 1.08 forward and 0.97 backward, nearly
area-preserving in the bulk, but ranges from 0.63 to 2.4 forward and from 0.045
to 14 backward. The implied divergence reaches 0.08–0.13 /day at the 99th
percentile, against a median FTLE of about 0.13 /day, so it is the same order as
the signal.

Where $\det \nabla F$ departs from 1, a $\lambda_2$ floor and a
$\lambda_2/\lambda_1$ floor stop being interchangeable, and the divergence is
one-sided. On the backward flow the old $\lambda_2$ floor terminates 0.97% of
grid points against the ratio floor's 0.135%, and the 78 points that only it
terminates have a median $\det \nabla F$ of 0.75 with a median ratio of 1.6: strongly
convergent, with $\xi_1$ well defined. A magnitude floor cannot distinguish
"nothing is stretching here" from "everything is contracting here", so it
preferentially terminates shrink lines inside convergence zones, where
attracting LCS are found. On the forward flow of that same 5-day case, where the
median areal factor is above 1, the two guards agreed almost exactly, which is
why the bias did not surface during forward-only development. The agreement is a
property of the window measured rather than of forward flow in general: at other
windows the two termination rates differ, in both directions.

## Why pruning is a run length in a tube

`ftle_ridge_seeds` puts several seeds on any ridge longer or wider than
`window_m`, and `shrink_lines` traces each of them into (nearly) the same
curve. `prune_shrink_lines` has to drop those near-duplicates while keeping two
lines that only run together over part of their length. The measurements below
were taken on the Cabo Verde example, forward map, `window_m` 30 km, with 53
seeds and 38 traceable lines.

### Score: the line integral, not the mean

Every member of a bundle carries the same FTLE at a given arc length, so the
line integral $\int \mathrm{FTLE}\,\mathrm{d}s$ ranks the bundle by length and
the longest trace wins. A line is never dropped in favour of one of its own
sub-segments, since the superset has the larger integral wherever the FTLE is
non-negative. Without a field the integral reduces to the arc length, which
is the same ranking within a bundle. The mean used by Farazmand & Haller (2012)
cannot separate bundle members at all, both being the same. Measured as the directed Hausdorff
distance from each of the 38 lines to the nearest stronger one (the max over its
points of the distance to that line):

| range | lines |
|---|---|
| 0–10 km | 11 |
| 10–17.5 km | 3 |
| 17.5–22.5 km | 0 |
| 22.5–100 km | 18 |
| over 100 km | 5 |

The cluster below 10 km is RK2 drift plus a few cells across the ridge, and the
empty band at 17.5–22.5 km separates it cleanly from the tail of lines that run
apart. The same statistic with the mean distance in place of the max shows no
such band (13 of 37 below 5 km, then a smear), so a scheme built on the mean
distance would have no threshold to pick.

### Coverage: an angle on the unit sphere

A point of a candidate line is covered once some point of an already-kept line
lies within the tube radius `window_m / 2` of it. The test is on the unit
sphere, comparing the dot product of the two position vectors against
$\cos(\text{radius} / R)$, so no projection and no standard parallel enter
(consistent with [the local east-north frame](#the-local-east-north-frame)).
The arc length of a line is the sum of its `_separation_m` steps, the same
measurement the integrator inverts, so the length of a traced line is its step
count times `step_m`.

Coverage is computed for every pair of lines before the walk, one broadcast
reduction per line, and the walk is then a boolean reduction over the kept
lines. That costs $L^2 P^2$ dot products for $L$ lines of $P$ points, well
under a second for the example. A spatial tree would bring it to
$L P \log(L P)$ and is the change to make if the line count grows by an order
of magnitude (GitHub issue #11).

### Rule: whole lines, and the sweep that fixed the constants

Lines are walked from the strongest score down. A line is dropped once it has
at least one covered point *and* the arc length of its uncovered segments (its
*new length*) is below `window_m`, and a line sharing nothing with a stronger
line is always kept, whatever its length. Trimming the covered stretch, as LCS Tool
(Onu, Huhn & Haller 2015,
[doi:10.1016/j.jocs.2014.05.002](https://doi.org/10.1016/j.jocs.2014.05.002))
does, was rejected because it would break a line into segments and break the
`(line, point)` layout along with it. Two curves that run together and then
separate would come back as fragments rather than as the two whole lines they
are.

Neither the tube radius nor the minimum new length is a free parameter, since
both are read off `window_m`, the resolution already declared when the seeds
were picked. Sweeping both by hand shows the kept count is flat around that
pair:

| tube radius | min new length | kept |
|---|---|---|
| 10 km | 20 km | 24 |
| 15 km | 15 km | 23 |
| 15 km | 30 km | 22 |
| 20 km | 40 km | 18 |
| 15 km | (max-distance rule) | 25 |

At `(15 km, 30 km)`, the pair `(window_m / 2, window_m)` for the 30 km run, the
new-length rule drops three lines beyond a rule that only checked the
max-distance: two with no new length at all and one 525 km line with 27 km of
new length. Requiring a covered point before a line can be dropped restores one
6 km stub that shares nothing with any other line, bringing the count at
`(15 km, 30 km)` to 23 rather than 22.

The bundles that remain at those settings are curves that run together for
50–150 km and then separate by more than 30 km (a triple at 26.5 W 16.3 N, a fan
at 24.8 W 16 N, a pair along 22 W). Each is two distinct structures rather than
one curve traced twice, and the rule keeps them.

## Closed shear lines

`closed_shear_lines` traces $\eta^\pm_\lambda$ with the RK2 step shrink lines
use, on the same interpolated $C$. Its tangent diagonalises $C$ in closed form
and takes $\xi_2$ as $\xi_1$ turned 90 degrees counter-clockwise. An
eigensolver may flip either eigenvector, and flipping $\xi_2$ alone turns
$\eta^+$ into $\eta^-$. Tying the two signs together leaves one overall sign,
which the running heading fixes, so a line stays on its branch.

The tangent carries no anisotropy guard like `min_anisotropy`. It is NaN only
where $\lambda_1 < \lambda^2 < \lambda_2$ fails. In an incompressible flow
$\lambda_1 \lambda_2 = 1$, so $\lambda = 1$ lies strictly between the
eigenvalues wherever $C$ is not isotropic. On the analytic Gaussian vortex of
`tests/test_elliptic.py` every orbit wider than the two-cell stencil is a
circle about the centre with a radius scatter of at most 0.3%, out to 58 km, so
no guard was needed there.

### The return map

Each centre gets two sections, due east and due west, so an orbit whose eastern
arc runs through a region where $\eta^\pm_\lambda$ is undefined is still found
from the west. A line launched at arc length $s$ along a section *returns* at
its first crossing of that section after more than half a turn about the
centre. Its return $P(s)$ is the arc length of that crossing, and a zero of
$P(s) - s$ is a closed orbit.

The search keeps these zeros:

- **A launch that already returns within `closure_tol_m`.** On the circles of
  an axisymmetric vortex every launch closes, so $P(s) - s$ sits near zero
  along the whole section without changing sign, and a sign-change search
  alone finds nothing. On the analytic vortex it returns one orbit instead of
  one per launch.
- **A sign change of $P(s) - s$ between neighbouring launches**, neither of
  them a hit. Bisection narrows it to a quarter of `closure_tol_m`. A bracket
  whose midpoint does not return straddles a gap in the return map rather than
  a zero, and is dropped. The final orbit must still close within
  `closure_tol_m`, which removes brackets that converged onto a jump.

The winding about the centre replaces a circumference test, which kept a loop
when its length was within a factor of 2 of $2\pi r$ and had no principled
bound. A crossing before half a turn is a line grazing back across
the section, and it ends the line without a return. A line still out after one
and a half turns ends without one too, which also bounds the work per line.
There is no self-intersection test, because two integral curves of a line
field cannot cross where the field is defined, and the search stops at the
first return.

### Why the defaults derive from the grid spacing

`launch_spacing_m`, `step_m` and `closure_tol_m` default to the grid spacing,
half of it, and half of it, so they mean the same at another resolution.
`max_radius_m` is the largest eddy radius looked for, a physical length like
`window_m`. On the 1/4 degree geostrophic case below, launched from the 23 GLED
centres, the number of GLED eddies with a boundary is:

| launch spacing | `closure_tol_m` | GLED eddies matched |
|---|---|---|
| 0.5 grid | 0.25 grid | 9 |
| 0.5 grid | 0.5 grid | 10 |
| 0.5 grid | 1 grid | 10 |
| 1 grid | 0.25 grid | 7 |
| **1 grid** | **0.5 grid** | **9** |
| 1 grid | 1 grid | 9 |
| 2 grid | 0.5 grid | 8 |

Halving the launch spacing adds one boundary 8 km in radius. Doubling the
tolerance enlarges one boundary from 28 to 37 km, by accepting a loop that
closes less well. A quarter-grid tolerance loses 2 boundaries.

### The measurements that fixed the defaults

The measurements below are dated records. The DUACS field and the GLED records
they use were fetched by an example since removed. The case is DUACS
geostrophic velocity from 2018-06-01 over 30 days, seeded at 1/25 degree over
2 E to 22 E, 44 S to 28 S, at the native 1/8 degree and block-averaged to 1/4
degree. GLED v1.0 (Liu & Abernathey 2023,
[doi:10.5194/essd-15-1765-2023](https://doi.org/10.5194/essd-15-1765-2023))
lists 23 coherent eddies for that window. A GLED eddy counts as matched when an
eddy centroid lies within its radius.

**Centre rule.** Windowed minima of $\lambda_2 / \lambda_1$ against windowed
maxima of $\lvert \theta - \operatorname{median} \theta \rvert$, both over
100 km, with the GLED centres as the ceiling:

| field | centres from | centres | GLED matched | boundaries outside GLED |
|---|---|---|---|---|
| 1/4 degree | $\lambda_2 / \lambda_1$ minima | 250 | 8 | 3 |
| 1/4 degree | $\theta$ extrema | 234 | 5 | 2 |
| 1/4 degree | GLED centres | 23 | 9 | 0 |
| 1/8 degree | $\lambda_2 / \lambda_1$ minima | 226 | 7 | 4 |
| 1/8 degree | $\theta$ extrema | 238 | 3 | 0 |
| 1/8 degree | GLED centres | 23 | 9 | 0 |

The eigenvalue ratio reaches 8 of the 9 eddies the ceiling reaches at 1/4
degree, against 5 for the rotation angle, so `elliptic_lcs()` uses it.

**Stretching range.** Scanned to $\Lambda = 2$ at a step of 0.03, every orbit
found from the GLED centres closes inside $[1/1.5, 1.5]$. The extremes are
0.677 at 1/4 degree and 1.310 at 1/8 degree. One boundary per field closes at
1.522, and both are boundaries from eigenvalue-ratio centres that no GLED eddy
matches. On hourly CMEMS model currents from 2025-06-01 over 30 days the orbits
close between 1.162 and 1.350. So the default $\Lambda = 1.5$ brackets every
matched boundary.

**Stretching step.** From the GLED centres at 1/4 degree, steps of 0.02, 0.03
and 0.05 match the same 9 eddies. The outermost radius of each agrees within
1 km across those steps, except one at 53, 48 and 47 km.

**Truncation against divergence.** Halving the stencil arm moves a
truncation-driven $\det \nabla F$ toward 1 and leaves a physical one where it
is. On the 1/8 degree geostrophic field:

| stencil | median $\lvert \det \nabla F \rvert$ | 95th percentile |
|---|---|---|
| neighbour, 1/25 degree | 3.38 | 667 |
| auxiliary, 2 km arm, 1/10 degree | 1.46 | 1317 |
| auxiliary, 1 km arm, 1/10 degree | 1.14 | 1966 |

The median approaches 1 as the arm shrinks, so in the bulk of the field the
departure from area preservation is truncation, and widening $\Lambda$ to
absorb it would fit noise. The upper tail grows instead, in the filaments where
a smaller arm resolves more stretching.

**Runtime.** The search traces every centre, section, stretching factor,
branch and launch in one array per step, and drops a line from the array once
it ends. Over the 250 eigenvalue-ratio centres at 1/4 degree it takes 11 s.
Looping over centres and stretching factors in Python took about 25 minutes on
the same case.

**Rotation sense.** The sign of $\theta$ at the 23 GLED centres, on the 1/4
degree field, against GLED's polarity, by the length of the window $\theta$ is
taken over:

| window | median $\lvert\theta\rvert$ | sign matches GLED |
|---|---|---|
| 1 day | 0.52 rad | 23 of 23 |
| 2 days | 1.04 rad | 23 of 23 |
| 4 days | 1.66 rad | 19 of 23 |
| 8 days | 1.44 rad | 14 of 23 |
| 30 days | 1.90 rad | 9 of 23 |

$\theta$ is defined modulo $2\pi$, and past half a turn its sign flips. The
example takes the rotation from a 1-day flow map.

**Removing the local dilation, measured and not adopted.** Divergence that
varies around a loop stretches its elements unevenly, so searching
$C / \lvert\det\nabla F\rvert$ instead of $C$ makes such a loop uniformly
stretched again. On an analytic vortex followed by a conformal map it recovers
the circles the plain search loses. On the 2018-06-01 window it changed no count
of matched GLED eddies by more than one, on DUACS, on GLORYS12 currents and on
GLORYS12 geostrophic currents, at 1/25 and 1/50 degree and over 15 and 30 days.
So the search takes $C$ only. A band of $\pm 10\%$ on the stretching instead of
an exact $\lambda$ changed the counts by at most four on DUACS and at most one
on GLORYS12.

**Separatrix.** For the analytic vortex in a uniform strain $s$, the steady
flow has saddles where $\Omega(r) = s$, at radius $r_*$, and its closed
streamlines lie within $r_*$ of the centre. The outermost boundary falls just
inside, which `tests/test_elliptic.py` checks:

| strain $s$ (1/s) | $r_*$ | outermost `radius_m` |
|---|---|---|
| $2.3 \times 10^{-7}$ | 60.7 km | 59.0 km |
| $4.6 \times 10^{-7}$ | 50.7 km | 47.0 km |
