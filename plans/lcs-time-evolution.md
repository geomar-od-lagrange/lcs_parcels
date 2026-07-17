<!--
Exploratory design note: how to diagnose the *time evolution* of hyperbolic
LCS built on top of the shrink-line layer (src/lcs_parcels/tensorlines.py).
Explores options and recommends a first experiment; nothing here is decided.
-->

# Diagnosing LCS time evolution

Status: **exploratory** — options and a recommended first step, not a decision.

We can now extract hyperbolic LCS at a single analysis time (`shrink_lines` on a
forward/backward `FlowMap`; see [`docs/api.md`](../docs/api.md)). This note
explores how to make that *time-resolved*: how an LCS, or the LCS field, changes
as time advances. Notation follows Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

## The question is ambiguous — split it first

"Time evolution of an LCS" means three different things, and the right machinery
differs for each:

- **Q1 — Where does *this* LCS go?** An extracted LCS is a **material** curve, so
  its later positions are fixed by the flow: $\mathcal{M}(t) = F_{t_0}^{t}(\mathcal{M}(t_0))$
  (Haller Eq. 5). This is pure advection — no re-diagnosis. It answers "how does
  this coherent curve move, stretch, and fold."
- **Q2 — How does the LCS *skeleton itself* evolve?** Slide the analysis window
  and re-diagnose; the set of ridges appears, disappears, merges, and splits.
  Answers "how does the coherent-structure field change with analysis time." Needs
  cross-frame **correspondence** (object tracking).
- **Q3 — What is the *instantaneous* structure at each moment?** The short-time
  limit of an LCS is an **Objective Eulerian Coherent Structure** (OECS; Serra &
  Haller 2016, [doi:10.1063/1.4951720](https://doi.org/10.1063/1.4951720)),
  computed from the rate-of-strain tensor at one instant — no window, naturally a
  frame per snapshot. Answers "what does the structure field look like right now,
  continuously in time."

These are complementary, not competing. Below, one approach per question, plus
what each needs from the package.

## Approach A — advect the extracted LCS (answers Q1)

Extract shrink lines at $t_0$, then treat their vertices as a particle set and
advect them with Parcels to get $\mathcal{M}(t)$ for $t > t_0$ (or $t < t_0$ for
an attracting LCS traced backward).

- **Pros.** Exact — it *is* the definition of the material curve's motion. Cheap:
  we advect a few polylines, not a grid. Directly shows transport, filamentation,
  and how a repelling LCS organises nearby tracers.
- **Cons.** Answers only Q1. A repelling LCS repels, so its advected image
  stretches and folds enormously over long $t$; the curve stays the *same
  material* but stops looking like a clean ridge. No birth/death/merge semantics.
- **Package fit / gap.** `shrink_lines` returns `lon`/`lat` on `(line, point)`;
  advecting them just needs to emit an arbitrary point set to Parcels. Our `Seed`
  is grid-based (`from_axes`), so this is a small gap: either a `from_points`
  constructor / lightweight point seed, or advect the vertices directly in the
  example (Parcels-side). The existing `FlowMap` ingest is grid-shaped and is not
  required here — this is a *forward map of a curve*, a different object.

## Approach B — sliding-window re-extraction + tracking (answers Q2)

For release times $t_0^{(k)} = t_0 + k\,\Delta$ (fixed window $T$, step $\Delta$),
compute the forward and backward shrink lines at each — a sequence of LCS
**frames** — then link curves between consecutive frames into tracks.

- **Correspondence options** (increasing sophistication):
  1. **Curve proximity** — modified Hausdorff distance between polylines; match a
     frame-$k$ curve to its nearest frame-$k{+}1$ curve under a threshold.
  2. **Mask overlap** — rasterise each LCS to a thin mask and match by
     intersection-over-union, the workhorse of ocean-eddy tracking.
  3. **Predictive (feature-flow-like)** — advect frame-$k$ **seeds** by $\Delta$
     and match the predictions to frame-$k{+}1$ seeds/curves; overlapping windows
     already share flow history, which stabilises continuity.
- **Pros.** The only approach that yields genuine skeleton evolution — birth,
  death, merge, split — as tracks. Well-trodden in the eddy-tracking literature.
- **Cons.** Correspondence is heuristic; ridges are noisy and curves split/merge
  ambiguously. Cost: each frame is a full advection ($N$ windows $\times$ two
  directions). $T$ vs $\Delta$ is a real knob (short $\Delta$ ⇒ strongly
  overlapping windows ⇒ correlated, easier-to-link frames, but more compute).
- **Package fit / gap.** Frame *generation* is essentially supported: a release
  series is `flowmap.to_seed()` → `pset_to_flowmap(..., t0=…, t1=…)` per window
  (the `(i, j, t_0, T)` cube of [`plans/timing-design.md`](timing-design.md)).
  The *tracking* is new logic — curve distance and track assembly — but can lean
  on libraries: `scipy.spatial` (KD-trees, directed Hausdorff) and
  `scipy.optimize.linear_sum_assignment` for frame-to-frame matching.

## Approach C — instantaneous OECS movie (answers Q3)

OECS are the instantaneous limit of LCS: hyperbolic OECS are tensor lines of the
**rate-of-strain** tensor $S = \tfrac{1}{2}(\nabla v + \nabla v^\top)$ through the
extrema of its eigenvalue field, computed from a single velocity snapshot (Serra
& Haller 2016, [doi:10.1063/1.4951720](https://doi.org/10.1063/1.4951720)).

- **Pros.** Naturally, continuously time-resolved — one field per snapshot, no
  window and *no advection at all* (just $\nabla v$ from the gridded velocity), so
  it is by far the cheapest route to a time-evolution movie and carries no
  tracking heuristics. Strong reuse: hyperbolic OECS are the same tensor-line
  integration we already have, with $S$ in place of $C$ and its smaller/larger
  eigenvalue selecting attracting/repelling.
- **Cons.** Instantaneous, not finite-time: OECS approximate LCS only over short
  horizons, so this answers "evolution of the *instantaneous* structures," not the
  motion of one finite-time LCS. Needs velocity gradients (we currently only
  advect particles; $\nabla v$ would come straight from the CMEMS grid via
  `xarray`, not from Parcels).
- **Package fit / gap.** Suggests generalising the tracer: `shrink_lines`
  currently pulls $C$ from a `FlowMap`; the reusable core is "integrate tensor
  lines of a symmetric 2-tensor field, seeded at eigenvalue extrema." Factoring
  that out lets the *same* integrator serve both finite-time LCS (from $C$) and
  instantaneous OECS (from $S$).

## Adjacent — elliptic (vortex) structures via LAVD

If "LCS" is meant to include **vortices** (elliptic LCS), the time story is
different again: Lagrangian-Averaged Vorticity Deviation (LAVD; Haller et al.
2016, [doi:10.1017/jfm.2016.151](https://doi.org/10.1017/jfm.2016.151)) defines
rotationally coherent vortices as outermost convex LAVD contours over an
averaging interval, and vortex tracking has mature methods. Out of scope for the
hyperbolic shrink-line layer, but the natural companion when the question is about
eddies rather than transport barriers.

## Comparison

| | A: advect LCS | B: window + track | C: OECS movie |
|---|---|---|---|
| Question | Q1 (where it goes) | Q2 (skeleton evolves) | Q3 (instantaneous) |
| Object | one finite-time LCS | finite-time LCS field | instantaneous structures |
| Advection cost | one curve | $N \times$ full grid, both ways | none ($\nabla v$ only) |
| Time resolution | continuous (a chosen curve) | per window step $\Delta$ | per velocity snapshot |
| New machinery | emit point set | tracking module | $\nabla v$ + tensor-field tracer |
| Reuses `shrink_lines` | as input | directly (per frame) | generalised to any $S$/$C$ |
| Maturity / risk | low | medium (matching is fiddly) | low–medium |

## Recommendation (for discussion)

Two tracks, cheapest-and-highest-reuse first:

1. **Start with C (OECS movie).** It gives an immediate, continuous
   time-evolution visualisation with no advection and no tracking heuristics, and
   it forces the useful refactor "tensor-line integration of an arbitrary
   symmetric tensor field" — after which `shrink_lines` (from $C$) and OECS (from
   $S$) share one core. Best effort-to-insight ratio and it de-risks the reusable
   abstraction.
2. **Then B (sliding window + tracking)** for true finite-time LCS evolution,
   beginning with the simplest correspondence (predictive seed matching over
   overlapping windows) before investing in Hausdorff/IoU. Use **A (material
   advection)** as a cheap companion diagnostic — pick one extracted LCS and show
   where it goes — which also exercises the point-emit gap.

A synthetic flow with a known answer should anchor the tracking work: the
periodically forced **double gyre** is the standard LCS benchmark and would let us
validate frame-to-frame linking before trusting it on CMEMS.

## Open questions

- **Window vs step.** How long a window $T$ and step $\Delta$ for B, and how much
  window overlap is worth the extra compute for cleaner tracks?
- **Identity under split/merge.** What does "the same LCS across frames"
  operationally mean when ridges merge or split — track curves, or the underlying
  seeds/ridge points?
- **Where to compute $\nabla v$.** On the native CMEMS grid, or a resampled one;
  and how sensitive OECS are to that choice.
- **Refactor shape.** Is a `tensor_lines(tensor_field, seeds)` core (with
  `shrink_lines` and an `oecs_*` wrapper on top) the right seam, or premature
  generality before C is actually attempted?
- **Validation.** Beyond the double gyre, is there an observational check
  (drifter clusters, altimetry eddies) the group already trusts?

## References

- Haller, G. (2015). *Lagrangian Coherent Structures.* Annu. Rev. Fluid Mech.
  47:137–162.
  [doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).
- Haller, G. & Sapsis, T. (2011). *Lagrangian coherent structures and the
  smallest finite-time Lyapunov exponent.* Chaos 21:023115.
  [doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597).
- Serra, M. & Haller, G. (2016). *Objective Eulerian coherent structures.* Chaos
  26:053110. [doi:10.1063/1.4951720](https://doi.org/10.1063/1.4951720).
- Haller, G., Hadjighasem, A., Farazmand, M. & Huhn, F. (2016). *Defining coherent
  vortices objectively from the vorticity.* J. Fluid Mech. 795:136–173.
  [doi:10.1017/jfm.2016.151](https://doi.org/10.1017/jfm.2016.151).
