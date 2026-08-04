# Numerics: why the numbers come out right

The quantitative half of the design record: the frame positions are measured in
and the error that frame carries, the units the tuning parameters are stated in,
and the guard that decides when $\xi_1$ is a direction rather than noise. The
companion document [`docs/architecture.md`](architecture.md) covers the other
half — what the types are and how they compose. Symbols and the definitions of
every quantity used below live in [`docs/notation.md`](notation.md); notation
follows Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137–162,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## The equirectangular metres frame

Positions go into metres through an **equirectangular projection with a single
standard parallel** $\phi_{\mathrm{ref}}$ (the mean of `lon_0`/`lat_0`), and
both the reference and the advected positions are read in that one frame. It is
not a tangent plane, and describing it as one obscures where the error lives: a
tangent plane would be a local linearisation whose error grows with distance
from the touch point in *every* direction, whereas here $Y$ is exact by
construction and only the zonal scale is approximated, by being frozen at
$\cos\phi_{\mathrm{ref}}$.

Writing $c_0 = \cos(\text{release latitude})$, $c_1 = \cos(\text{arrival
latitude})$ and $c_{\mathrm{ref}} = \cos\phi_{\mathrm{ref}}$, the true gradient
relates to the one the package computes as

$$\nabla F_{\mathrm{true}}
= \mathrm{diag}\!\left(\tfrac{c_1}{c_{\mathrm{ref}}},\, 1\right)
  \nabla F_{\mathrm{ours}}
  \mathrm{diag}\!\left(\tfrac{c_{\mathrm{ref}}}{c_0},\, 1\right).$$

So $F_{yy}$ is exact; $F_{xx}$ is off by $c_1 / c_0$ — driven by *meridional
excursion* of the particle, with $c_{\mathrm{ref}}$ cancelling entirely — and
the two off-diagonals are off by $c_1 / c_{\mathrm{ref}}$ and
$c_{\mathrm{ref}} / c_0$. The dominant error term is therefore not the size of
the domain in longitude but how far particles travel in latitude relative to the
standard parallel.

That algebra is the whole error structure, and it needs no measurement to state.
What it says is that the error is set by *meridional excursion relative to the
standard parallel* — how far a particle's release and arrival latitudes sit from
$\phi_{\mathrm{ref}}$ — and not by domain width as such; a wide, thin zonal band
is fine, a narrow but meridionally tall one is not.

To put a scale on it, one analytic test flow map at one centre latitude gave a
median FTLE error of 0.65% over a 5-degree domain, 3.0% over 20 degrees and 15%
over 60 degrees. Those three numbers are an illustration of magnitude from a
single case, not a table of error-versus-domain-size: another flow, or the same
domain sizes at another centre latitude, moves them. The package is
consequently valid for regional domains of modest latitude range with no
dateline crossing, and is not currently correct for basin-scale ones. Tracked in
GitHub issue #18, alongside the related dateline/longitude arithmetic in #13;
the fix is exact and cheap — two diagonal rescalings by cosines already carried
in the dataset — so it need not wait for the dateline work.

## Why the tuning parameters are stated in their own units

The tunable quantities of the tensor-line layer are stated in the units of the
thing itself — `window_m`, `step_m` and `line_length_m` in metres,
`min_anisotropy` as a dimensionless eigenvalue ratio — and converted internally
against the field's own grid spacing, so the same call means the same thing at
any resolution and over any horizon. Expressed the natural implementation way
instead, the two lengths would depend on something other than what the caller is
asking for: a neighbourhood as a cell count depends on grid resolution, a line
as a step count depends on the step size.
`quantile` is left alone: making it absolute would change ridge selection from
relative-to-this-field to an absolute threshold, which is a science decision
rather than a units one.

Both lengths are budgets, not achieved quantities, and both invite the same
misreading. `line_length_m` bounds the traced arc: the
integrator spends at most `line_length_m / (2 * step_m)` steps per direction and
a line that terminates earlier is shorter, with the returned block NaN-filled
past termination so every row has equal length. `window_m` is the *side* of the
ridge-seed neighbourhood, which reaches only `window_m / 2` to either side of
its own grid point; two seeds can therefore sit about `window_m / 2` apart. That
is geometry, not measurement: the bound follows from the window's reach and
holds for any field. Anyone reading either budget as "the length you get" or
"the spacing you get" is off by a factor of two.

## Why the degeneracy guard is an eigenvalue ratio

`min_anisotropy` floors $\lambda_2 / \lambda_1$. It is the third form the guard
has taken, and the two rejected ones are recorded here because each was defended
on grounds the replacement also has to answer.

A raw $\lambda_2$ floor was rejected on the argument that it "silently retunes
as the window changes" — true: $\lambda_2$ grows exponentially in $|T|$, so a
fixed floor tightens as the window lengthens and a well-definedness guard drifts
into being a selector. A stretching-rate floor, `ftle_min_per_day`, cured that
by dividing out $|T|$ — but it retunes *worse*, across flow regimes rather than
across windows, which is the direction that actually bites. The 0.005/day
default was calibrated to the mesoscale ocean at a 7-day window.

The comparison below is a paper exercise across three flow regimes, and it rests
on one premise that has to be stated because everything follows from it: each
regime is integrated over a window $|T|$ *comparable to its own* $1/\mathrm{FTLE}$,
which is what one actually does — the window is chosen so the flow has time to
stretch by roughly one e-fold. That fixes the three $(\mathrm{FTLE}, |T|)$ pairs:

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
0.140/day at 6 h and $1.9\times 10^{-4}$/day at 180 d — as fractions of the two
outer flows' FTLE, 2.3% and 1.8%. So the first argument was right about its
target and too narrow: a rate is scale-free in $T$ only.

The ratio retunes with neither, and it is also the physically correct quantity
rather than merely the scale-free one. The sensitivity of an eigenvector of $C$
to a perturbation of $C$ scales as the inverse of the *relative* gap between the
eigenvalues, so $\lambda_2 / \lambda_1$ is exactly what decides whether $\xi_1$
is a direction or numerical noise, and no stretching rate can stand in for it.
Measured: at the default 1.15 a 1% error in $C$ swings $\xi_1$ by about 2
degrees; at a ratio of 1.05 by 6 degrees; by a ratio of 4 it has flattened out
at about 0.25 degrees.

The default 1.15 is the old `ftle_min_per_day=0.005` behaviour carried over
essentially exactly: at the 7-day window of the examples that rate corresponded
to a $\lambda_2$ floor of 1.0725, which for incompressible flow
($\lambda_1 \lambda_2 = 1$) is a ratio of 1.15. The change of quantity is
therefore not a change of tuning at the calibration point — it is a change in
what happens away from it.

And away from it the difference is not academic, because the ocean surface is
not incompressible. Measured on the Cabo Verde example (5-day window, points
taken 34 km clear of any coast), the flow map's areal factor $\det \nabla F$ has
a median of 1.08 forward and 0.97 backward — nearly area-preserving in the bulk —
but ranges from 0.63 to 2.4 forward and from 0.045 to 14 backward. The implied
divergence reaches 0.08–0.13 /day at the 99th percentile, against a median FTLE
of about 0.13 /day: the same order as the signal.

Where $\det \nabla F$ departs from 1, a $\lambda_2$ floor and a
$\lambda_2/\lambda_1$ floor stop being interchangeable, and the divergence is
one-sided. On the backward flow the old $\lambda_2$ floor terminates 0.97% of
grid points against the ratio floor's 0.135%, and the 78 points it kills alone
have a median $\det \nabla F$ of 0.75 with a median ratio of 1.6 — strongly
convergent, and with $\xi_1$ perfectly well defined. A magnitude floor cannot
distinguish "nothing is stretching here" from "everything is contracting here",
so it preferentially terminates shrink lines inside convergence zones — which is
where attracting LCS live. On the forward flow of that same 5-day case, where
the median areal factor is above 1, the two guards agreed almost exactly — but
that agreement is a property of the window measured, not of forward flow in
general: at other windows the two termination rates part company, in both
directions. It is enough to explain why the bias never surfaced during
forward-only development.
