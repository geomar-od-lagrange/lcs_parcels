# Acting on the PR #7 review

Plan for resolving the inline review comments on PR #7, folded together with the
open issues. Written 2026-08-03.

The review dropped 20 `TODO` comments across `AGENTS.md`, `README.md`,
`src/`, and the example notebooks. The reviewer flagged each *class* of issue
once, at the first place they hit it, so a repo-wide sweep was run to find the
unflagged instances of each theme. This plan works the themes, not the
individual comments.

## Decisions taken

Settled in review discussion; not to be relitigated while executing.

- **Audience is the root theme.** Three reader classes: the trusting user, the
  inquisitive user who wants to verify the science without becoming a
  developer, and the developer who needs the design *decisions*. The existing
  "notebooks are human-facing" rule is insufficient and becomes a derivation of
  this model.
- **Explanation is bounded on both sides.** Explain the non-obvious *choice*;
  do not re-teach the field. The reader has read the papers.
- **Indirection is a tax in examples, an asset in library code.** In examples,
  prefer explicit and duplicated code over a helper, assuming a reader
  consuming line by line with minimal scrolling. This distinction falls out of
  the audience model; it is not a blanket preference for duplication.
- **Units stay SI.** `ftle()` returns 1/s. Deviate only where a field has a
  clear convention of its own (Sverdrups and the like). Go CF where it is easy
  --- `units`, `long_name` --- and not further; no bounds, intervals, or
  cell methods.
- **Reprs are terse summaries.** Main attributes only. `.ds` remains how a user
  displays the dataset; hinting at that in the README and examples costs one
  line and simultaneously makes clear these are not xarray objects.
- **Ridge-finding stays explicit.** `ftle_ridge_seeds` keeps taking the FTLE
  field, not the `FlowMap`: no hidden recomputation, and a caller can pass a
  smoothed or masked field. User convenience is bought once, in
  `flowmap.lcs(...)`, which computes FTLE a single time and passes it down.
- **No data is committed to the repo.** Distinguish "runs against the CMEMS
  online store on every execution" from "needs a one-time download".
- **Ruff lands early**; the rest of the infrastructure work is deferred.

## Deferred

Filed and out of scope here.

- #10 --- linting. Pulled forward as PR A; see below.
- #11 --- tensor-line marching performance. Amended to cover the *output
  structure* too: lines are a fixed `2 * n_steps + 1` NaN-padded rectangle, and
  the loop never breaks, so a line that dies after three steps still costs the
  full run. Cost and representation are one problem. `shrink_lines`'s output
  structure does not change in this plan.
- #12 --- packaging, docs site, dependabot, badges, coverage beyond statements.
- #13 --- longitude arithmetic is not dateline-aware: the auxiliary centre
  mean, the metric-frame anchor, every diff op, and the tensor-line stepper all
  treat longitude as a plain real line.
- #14 --- Q2, sliding-window LCS re-extraction and frame-to-frame tracking.
- #15 --- Q3, instantaneous OECS from the rate-of-strain tensor, and the
  "tensor lines of an arbitrary symmetric 2-tensor" refactor it forces.
- #8 (elliptic LCS / vortex detection) and #9 (3D) --- explicitly out of scope.

Filing #14 and #15 emptied `plans/lcs-time-evolution.md` of undelivered
content --- Q1 had already shipped --- so it moved to `plans/done/`.

## Sequencing

Ordering constraints that actually bind: the audience model first, since the
prose and example work is judged against it; canonical coords before metadata,
since both rewrite output coords; every API change before the examples are
rewritten, or they get rewritten twice; prose last among the code work, for the
same reason.

### PR A --- the contract

**1. `AGENTS.md`: the audience model.** Add the three reader classes. Rewrite
the existing example and notebook rules as derivations of it. New rules that
fall out:

- explain the non-obvious choice; do not re-teach the field;
- examples prefer explicit, duplicated code over indirection --- library code
  does not;
- every returned array carries `name`, `long_name`, `units`;
- narrow the moved-plan-link exemption so it does not cover `src/` and `tests/`
  (`grids.py:36` and `tests/test_roundtrip.py:15` both point at files now in
  `plans/done/`).

Clarify the vanilla-plot rule, which currently reads as stricter than it is. It
governs the plot you *first write*: reach for `.plot()` and accept its defaults
rather than opening with multi-line styling. Deviating is fine when asked for,
or when the plot turns out to need it. It is an anti-over-styling rule, not a
ban on ever setting a keyword. Worth stating explicitly --- the review sweep
misread it as a rule the LCS examples violate wholesale.

The NumPy/SciPy escape hatch in `tensorlines` does **not** become an AGENTS.md
rule. It gets a comment at the implementation site (step 7) and comes out of
the user-facing module docstring (step 9).

The one rule the code genuinely violates is the ban on runtime introspection;
step 4 resolves it by construction rather than by amending the rule.

**2. Ruff + a CI lint gate.** Mechanical, and deliberately before the rewrites
so the later diffs stay clean and reviewable. Closes #10.

### PR B --- API and internals

**3. Keyword-only arguments across the public surface.** Every public entry
point that takes an adjacent lon/lat pair takes it positionally, which is the
classic silent-swap footgun --- transposed arguments produce a plausible-looking
result somewhere off the coast of nowhere rather than an error:

- `Seed.from_axes(lon, lat)`, on all three classes (`grids.py:176`, `:498`,
  `:545`);
- `Seed.pset_to_flowmap(lon, lat, *, t0, t1)` (`grids.py:220`) --- already
  keyword-only past `t1`, but not for the pair that matters;
- `FlowMap.image(lon0, lat0)` (`grids.py:415`);
- `shrink_lines(flowmap, seed_lon, seed_lat, *, ...)` (`tensorlines.py:67`).

Make the lon/lat pairs keyword-only throughout. Mechanical, and first in this
PR so every later signature change lands on top of the corrected form rather
than needing a second pass.

Add the rule to `AGENTS.md` here rather than in PR A --- it is a code
convention, so it belongs with the change that establishes it: public API takes
same-typed adjacent arguments keyword-only, so a caller cannot silently
transpose them.

**4. Canonical diagnostic-grid coords and per-subclass properties.** Today the
diagnostic grid location is `lon_0`/`lat_0` on `(i, j)` for the neighbour
stencil, but `lon_c`/`lat_c` for the auxiliary one, where `lon_0` lives on
`(i, j, displacement)`. One concept, two names, which is what forces every
downstream consumer to sniff.

The zoo is smaller than it looks. `from_axes(lon, lat)` takes plain axis
arguments, so there is no `grid_lon`/`grid_lat` in the code today --- only three
coordinate pairs, plus a `lon_grid`/`lat_grid` local at `grids.py:450`. Settle
all of it here:

- **`lon_grid`/`lat_grid` on `(i, j)`, both stencils** --- the diagnostic grid
  point location, carried explicitly. This is the canonical coordinate every
  downstream consumer reads, and it replaces `lon_c`/`lat_c` outright. `_c`
  goes: once it exists on both stencils, "centre" is a misnomer for the
  neighbour case, where there are no arms to be the centre of. `_grid` says
  what it is without knowing the auxiliary story --- which matters, because
  this is the name on every plot axis and in every repr.
- **`lon_0`/`lat_0` stay** --- release positions of actual particles, `(i, j)`
  for neighbour and `(i, j, displacement)` for auxiliary. The `_0` suffix is
  not an accident to be renamed away: it is $x_0$ from the notation the reader
  already has, and it is what `docs/notation.md` documents.
- **`lon`/`lat` stay** --- advected positions, same dims as `lon_0`. The
  `lon_0` to `lon` pairing reads as one object before and after the flow, and
  that symmetry is worth keeping intact.

For the neighbour stencil `lon_grid` equals `lon_0`. The redundancy is
deliberate --- the same call already made for the auxiliary schema: store the
value, do not make consumers reconstruct it from a convention.

Replace the `_grid_lonlat` module helper with properties on `Seed`/`FlowMap`
**named exactly as the coordinates they return** --- `lon_grid`, `lat_grid`, so
`flowmap.lon_grid` is `ds["lon_grid"]` with no translation layer --- plus
`grid_image`, the advected positions collapsed over `displacement` for the
auxiliary case and passed through for the neighbour one. What must not survive is
a base-class property that branches internally: that would be the same violation
wearing a better name; the point is that the type carries the information.
Once the coordinate is canonical, `lon_grid`/`lat_grid` no longer branch on
anything, so they are concrete one-liners on the two base classes; only
`grid_image`, which genuinely differs per stencil, is abstract and overridden.

(`grid_image` is the least settled name here. It returns a two-variable Dataset,
not a coordinate, so it does not follow the rule above.)
This removes `if "displacement" in advected.dims:` at `grids.py:452`, which is
verbatim the pattern `AGENTS.md` forbids by name, and the
`"lon_c" in obj.coords` sniff at `grids.py:101-102`.
`ftle_ridge_seeds` then reads the canonical coord off the field it is handed,
with no type to dispatch on and no sniff.

Touches `grids.py`, `tensorlines.py`, the tests, all four examples, and
`docs/notation.md`.

**5. Output metadata.** `name`, `long_name`, `units` on everything returned.
`_assemble_tensor` currently names its result `"tensor"`, so
`deformation_gradient()` and `cauchy_green()` come back indistinguishable ---
fix that. Label the `eig` coord. This is the precondition for vanilla plots in
step 10, and for dropping the hand-written `* 86400.0` that appears in three
examples.

**6. Parameters in physical units.** The reviewer flagged `window=7` as
implicitly sensitive to grid resolution; the same defect runs wider.

- `window` becomes a distance.
- `n_steps` becomes a length in metres --- it is currently a line *length*
  (1500 km) expressed as a step count.
- `lambda_max_min` becomes an FTLE floor in 1/day evaluated against the flow
  map's own `|T|`; as a raw Cauchy-Green eigenvalue it silently tightens or
  loosens as the window changes.
`quantile` stays as it is for now. It is as grid-dependent as `window` was, but
turning it into an absolute FTLE floor changes ridge selection from
relative-to-this-field to absolute --- a science decision, not a units one.
Revisit it in a dedicated pass on tuning parameters.

Signature changes, so this lands before the tests and examples are written
against them.

**7. `tensorlines` internals: lift, rename, explain, test.** `xi1`, `step`, and
`half` become top-level tested functions or get inlined; same for the nested
`central_diff` and `arm_diff` in `grids.py`. `xi1` is the one the review did
not flag and the one most needing a test --- it holds the entire
eigen/orientation/termination logic.

Rename as they are lifted. There is no space pressure here, and terse names are
charging the reader to reconstruct what the code means: `step` does not say
*integration* step, and `half` does not say it traces one direction away from
the seeds. Proposed, to be settled at implementation time:

| now | proposed |
|---|---|
| `xi1` | `shrink_direction` |
| `step` | `step_lonlat_by_meters` |
| `half` | `trace_half_line` |
| `d` | `direction` |
| `bad` | `terminated` |
| `lam`, `vec` | `eigenvalues`, `eigenvectors` |

The same applies inside `grids.py`: `central_diff` and `arm_diff` lift to
module-level `_central_diff(field, dim)` and `_arm_diff(field, positive,
negative)` with their own unit tests, and their locals get spelled out.

Document and check, per the "explain *and* check" comment:

- name RK2;
- the heading flip, `d[np.sum(d * heading, axis=1) < 0] *= -1`;
- the initial-branch pick, which dots against the 45-degree direction, so a
  seed whose $\xi_1$ is near the anti-diagonal flips on numerical noise;
- the two halves picking orientation independently, which permits a kink at the
  shared seed point;
- the degeneracy guard firing at the RK2 *midpoint*, killing steps whose
  endpoints are both fine;
- `n_steps` as NaN-fill rather than break, with a pointer to #11.

There are currently no unit tests for any of this --- `tests/test_tensorlines.py`
covers only end-to-end straight-line behaviour.

**8. Reprs and `flowmap.lcs(...)`.** Terse summary reprs on `Seed` and
`FlowMap`. `lcs()` computes FTLE once and passes it to ridge-finding, so the
convenience lives in one place and the ridge logic is not duplicated.

### PR C --- prose, examples, and CI

**9. Prose hygiene.** Strip process narrative and design rationale from
reader-facing text:

- the four-section design document at the top of `grids.py` (developer text on
  the import path, duplicating `docs/notation.md`);
- the meta-comment in `__init__.py`;
- `docs/notation.md`'s history of our own repo, and its "the choice is left to
  the implementation session" passage for a choice long since shipped;
- the agent-addressed HTML comments opening `docs/api.md` and
  `docs/architecture.md`;
- the moved-plan links from `src/` and `tests/`.

Design rationale relocates to `docs/architecture.md` rather than being deleted.

**10. Examples rewrite.** Explicit and inline throughout: `set_lost_to_nan`
written out in each notebook rather than shared, `advect` and `ftle_per_day`
dissolved into straight-line code. Display the datasets --- the reader is at
home in the xarray/CF world. Show both stencils. Drop the round-trip demos and
the contract framing entirely; the tests cover that, and a reader interested in
the design reads the tests. Daily intervals in the evolution example.
Progressbar on. Vanilla plots, now possible given step 5. Drop the pixi
execution instructions --- that is our dev environment, not the reader's
concern.

**11. Data story and CI gating.** Two examples currently claim to run "offline
(bundled currents)", but `examples/data/` is gitignored, nothing in the repo
tracks or generates `cabo_verde_currents_hourly.nc`, and both examples open it.
A fresh clone fails on both, they can never be CI-gated, and their committed
outputs are unreproducible by anyone else. Resolve under the online-each-run
versus download-once distinction, committing no data. Then extend CI to gate
every example that can be gated --- today it runs one of four, so
"a broken example is treated like a failing test" is unenforced for three.
