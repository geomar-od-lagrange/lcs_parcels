# Setting the $\lambda$ range from the data

Next round on branch `explore/shear-lines`, following
`plans/done/shear-lines-next-steps.md`. That round settled the stencil, the
centre criterion and the return map. This one is about $\lambda$.

## Status

The committed `examples/cape_cauldron_vortices.py` runs 30 days of CMEMS
surface `uo`/`vo` on a 1/25 degree `NeighborSeedGrid` and closes orbits at one
centre, 13.32 E 35.64 S, with mean radii from 26.9 to 37.7 km at $\lambda$ from
1.20 to 1.28 on the $+$ branch.

The uncommitted `examples/cape_cauldron_gled.py` runs the same search on DUACS
geostrophic `ugos`/`vgos` over the GLED 30-day window starting 2018-06-01, and
scans $\lambda$ from 0.80 to 1.60 in steps of 0.05.

What follows is driven by the CMEMS run. The $|\det \nabla F|$ quantiles are
0.18, 0.69, 1.43, 7.95 and 165 at 15 days, with a 95th percentile of 2192 at
30 days. The orbit count depends on the scan step, 13 at 0.02 and 7
at 0.04, at the same centre.

## What $\lambda$ is

Along a trajectory $\frac{d}{dt}\ln \det \nabla F = \nabla \cdot u$, so a loop
whose interior changes area by $J$ and keeps its shape changes perimeter by
$\sqrt{J}$. The $\lambda$ at which a boundary closes is therefore
$\lambda \approx \sqrt{J}$ over that boundary's interior.

This makes $\lambda$ an output rather than a knob. Each centre carries a nested
family of $\lambda$-loops and the boundary is the outermost member over the
whole family, at whatever $\lambda$ that member sits. Wang, Beron-Vera &
Olascoaga (2016, [doi:10.1002/2015JC011620](https://doi.org/10.1002/2015JC011620))
find $\lambda = 1$ for $T$ from 30 to 210 days on one Agulhas ring and
$\lambda$ from 1.05 to 1.75 for longer $T$, on one dataset and one eddy.

This changes the scan. Convergence is as common as divergence, so the range is
log-symmetric about 1, $\lambda \in [1/\Lambda, \Lambda]$. And $\Lambda$ is a
property of the field rather than a preference, so it is measured rather than
chosen.

## Changes planned for this round

1. **Measure $\Lambda$ before scanning.** Take $\sqrt{|\det \nabla F|}$ in an
   annulus around each candidate centre, not over the whole domain where
   filamentation in the strain-dominated exterior dominates, and set $\Lambda$
   to its 90th percentile. On the CMEMS run the domain-wide 15-day median maps
   to $\sqrt{1.43} = 1.20$, which is where the orbits closed, so the annulus
   statistic is worth checking against the orbits it predicts.

2. **Make the range log-symmetric.** The GLED notebook scans 0.80 to 1.60,
   which reaches $\Lambda = 1.6$ upward and only $1/1.25$ downward. Replace the
   endpoints by $1/\Lambda$ and $\Lambda$ from item 1.

3. **Set the step from radius sensitivity.** The CMEMS run gives 26.9 to
   37.7 km over $\lambda$ from 1.20 to 1.28, so $dr/d\lambda$ is about 135 km
   per unit $\lambda$. Resolving the boundary to the 1/25 degree seed spacing
   needs a step near 0.033, and the jump from 7 orbits to 13 when the step
   halved from 0.04 to 0.02 is the same boundary found twice. Scan at 0.03 and
   dedupe by mean radius.

4. **Separate truncation from divergence.** Truncation error in $\nabla F$
   inflates $\det \nabla F$ the same way physical divergence does, and widening
   $\Lambda$ to absorb it fits noise. Halving the stencil arm moves a
   truncation-driven $\det \nabla F$ toward 1 and leaves a physical one where it
   is. Run that pair on the geostrophic field, where $\det \nabla F$ should
   already sit near 1, and record both distributions.

5. **Report the $\lambda$ of each boundary.** Its distance from 1 says how far
   the loop is from reassuming its arc length, which Wang et al. read as the
   ring losing coherence. Carry it on the returned orbit rather than leaving it
   implicit in the sweep.

## Expected width of the range

For a fluctuating divergence of rms $\delta_{\rm rms}$ and Lagrangian
decorrelation time $\tau_c$, the accumulated $\int \nabla \cdot u \, dt$ is a
random walk, so

$$\ln \Lambda \approx \tfrac{1}{2} \delta_{\rm rms} \sqrt{\tau_c T}.$$

The range therefore widens as $\sqrt{T}$ rather than linearly, and doubling the
horizon costs a factor $\sqrt{2}$ in $\ln \Lambda$. Measure $\delta_{\rm rms}$
and $\tau_c$ on the field in hand rather than assuming them.

Resolution enters through $\delta_{\rm rms}$, with a break at the submesoscale.
Under a $k^{-3}$ kinetic energy spectrum the gradient variance
$\int k^2 E(k)\,dk$ grows logarithmically with $k_{\max}$, so refining a
mesoscale-resolving grid barely moves $\Lambda$. Under $k^{-2}$ it grows like
$k_{\max}$, so $\delta_{\rm rms} \sim \Delta x^{-1/2}$ and the divergent
fraction of the flow grows with it.

At $\delta_{\rm rms}$ near 1 per day and $T = 30$ days this puts $\Lambda$ near
4, which is not a range a sweep can cover at a step of 0.03. A
submesoscale-resolving two-dimensional surface slice has no single $\lambda$
per loop, because its interior gains and loses area locally. That bounds where
the method applies rather than where the implementation falls short.

## Package API

When the sweep moves from the notebook into the package, $\lambda$ is a search
axis and not a scalar keyword with a default. The call takes the range and the
step, and each returned boundary carries the $\lambda$ it closed at as a
coordinate. A range bracketing 1 log-symmetrically means the same thing at any
resolution and any horizon, which a single default $\lambda$ would not.

The Cauchy-Green field is computed once and shared across the whole scan, and
only the tangent field and the integration repeat, so widening the range is
cheap and the step is what bounds the work.

## What to measure

- $\det \nabla F$ on the geostrophic field, whole-domain and per-annulus, at
  the 1/25 degree neighbour stencil and at half that arm.
- $\delta_{\rm rms}$ and $\tau_c$ of the surface divergence on both the CMEMS
  and the geostrophic field.
- Boundary radius against $\lambda$ at the GLED centres, against the radii GLED
  reports for the same eddies.
