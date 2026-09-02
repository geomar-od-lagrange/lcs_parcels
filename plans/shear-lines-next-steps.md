# Shear lines and coherent vortices: next steps

Status of branch `explore/shear-lines` on 2026-09-02 and what to change in the
next iteration of `examples/cape_cauldron_vortices.py`. The notebook implements
the $\eta_\lambda$ shear lines of Haller & Beron-Vera (2013, *J. Fluid Mech.*
731:R4, [doi:10.1017/jfm.2013.391](https://doi.org/10.1017/jfm.2013.391)),
Eq. 14, with a Poincaré-section return map, and finds one Agulhas ring at
13.04 E, 35.64 S over a 15-day window from 2025-06-01.

## What the current run shows

- Closed orbits exist at one centre only, with mean radii from 21 km to 60 km,
  at $\lambda = 1.0$ and $1.1$, all on the $+$ branch.
- The outermost orbit stretches by 1.018 against its $\lambda = 1$; a circle of
  the same radius stretches by 1.21. Both images stay on the grid.
- Arc-length ratios of open $\eta_\lambda$ arcs track $\lambda$ to within
  about two percent over $\lambda \in [0.8, 1.4]$, so the tangent field and
  the stepper are sound.
- The FTLE band around the ring sits near 110 km radius. The detected boundary
  at 60 km is well inside it, so the true material boundary was missed.

## Changes for the next iteration

1. **Stencil.** Switch from `NeighborSeedGrid` to `AuxiliarySeedGrid` with the
   default 1 km arms. The $|\det \nabla F|$ quantiles of the current run are
   0.18, 0.69, 1.43, 7.95 and 165 (5 to 95 percent). No surface flow changes
   area by a factor of 165 in 15 days; the neighbour stencil spans 8.8 km and
   is truncation error wherever strain is high. Every downstream quantity
   (admissibility mask, centre selection, the "not area-preserving" cell) is
   built on that gradient.
2. **Centre selection.** Drop both criteria the notebook tries ($\lambda_2$
   minima, then minima of $(\lambda_1 - 1)^2 + (\lambda_2 - 1)^2$). Haller &
   Beron-Vera place sections through singularities of $C$, where
   $\lambda_1 = \lambda_2$, so seed on windowed minima of $\lambda_2 / \lambda_1$
   with the same 50 km edge and NaN guard. Remove the if-fallback scaffolding
   once one criterion works.
3. **$\lambda$ scan.** Replace the coarse set (0.1 steps outside
   $[0.9, 1.2]$) by a uniform fine scan, e.g. $\lambda \in [0.8, 1.4]$ in
   steps of 0.02. The outermost orbit is only ever found at a sampled
   $\lambda$, so the coarse grid bounds the boundary radius from below.
4. **Return map.** Draw $P(s) - s$ at the ring centre for a few $\lambda$, not
   at the first $\lambda_2$ centre where nothing returns. Consider a second
   section running west, so an orbit whose eastern arc leaves the admissible
   region is still found.
5. **Prose fixes.** Cite Haller (2015) §5.1 for Table 1, as
   `cabo_verde_lcs` does. Remove "a circle is tangent to $\eta_\lambda$
   nowhere". State the track-length observation without inferring which
   termination fired. Make the window statement consistent with the ring size
   the box targets.

## Time horizon

Use $T = 30$ days as the working horizon and report the boundary radius at
15, 30 and 60 days on the same data.

- A ring swirls once in roughly 7 to 12 days. At 15 days a closed shear line
  spans one or two revolutions; at 30 days it has held for three or four, which
  is what a coherent Lagrangian vortex boundary claims. Haller & Beron-Vera used
  90 days on daily geostrophic fields.
- The outermost closed orbit can only shrink as $T$ grows, so the radius as a
  function of $T$ is the result to show rather than any one window.
- With 1 km auxiliary arms, 30 days is resolved in the ring interior and rim.
  At 60 days the rim is under-resolved again, so 60 is a stress test.
- Download 2025-05-31 to 2025-08-01 in one file (about 420 MB hourly, or use
  the daily-mean product `cmems_mod_glo_phy_anfc_0.083deg_P1D-m`), same
  3-degree margin. Rings drift north-west at about 5 km per day and stay inside
  the box; particles that leave become NaN as now.

## Tooling the package would need for a shear-line API

- A tangent-field hook for the RK2 tracer, so `shrink_lines` and an
  $\eta_\lambda$ tracer share one stepper; `_step_lonlat_by_meters` made public.
- A windowed local-minimum seeder beside `ftle_ridge_seeds`, with the edge and
  NaN guard.
- A Poincaré section type with a first-return map that ignores crossings
  outside the segment, closure refinement, and the enclosure and
  circumference-ratio filters.
- A winding diagnostic and arc-length validation through `FlowMap.image()`.

## Side result

On the Cabo Verde field (`examples/cabo_verde_lcs.py` setup, neighbour stencil)
a 2-day forward window has one closed orbit at $\lambda = 1$, about 82 km
across; windows of 3 and 5 days in either direction have none.
