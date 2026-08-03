# Acting on the PR #7 review

Plan for resolving the inline review comments on PR #7, folded together with the
open issues. Written 2026-08-03.

The review dropped 20 `TODO` comments across `AGENTS.md`, `README.md`,
`src/`, and the example notebooks. The reviewer flagged each *class* of issue
once, at the first place they hit it, so a repo-wide sweep was run to find the
unflagged instances of each theme. This plan works the themes, not the
individual comments.

A comment is resolved when its `TODO` is deleted. PR #7's comments should drain
visibly across the three PRs below rather than all at once at the end.

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
- #8 (elliptic LCS / vortex detection) and #9 (3D) --- explicitly out of scope.

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
- the sanctioned NumPy/SciPy boundary in `tensorlines`, which currently
  documents its own escape hatch as policy in a user-facing docstring;
- narrow the moved-plan-link exemption so it does not cover `src/` and `tests/`
  (`grids.py:36` and `tests/test_roundtrip.py:15` both point at files now in
  `plans/done/`).

Also resolve, one way or the other, the two rules the code currently violates:
the ban on runtime introspection (see step 3) and "prefer vanilla plots", which
the LCS examples ignore wholesale --- the latter only becomes achievable after
step 4.

**2. Ruff + a CI lint gate.** Mechanical, and deliberately before the rewrites
so the later diffs stay clean and reviewable. Closes #10.

### PR B --- API and internals

**3. Canonical diagnostic-grid coords and per-subclass properties.** Today the
diagnostic grid location is `lon_0`/`lat_0` on `(i, j)` for the neighbour
stencil, but `lon_c`/`lat_c` for the auxiliary one, where `lon_0` lives on
`(i, j, displacement)`. One concept, two names, which is what forces every
downstream consumer to sniff.

- Carry `lon_c`/`lat_c` on `(i, j)` for **both** stencils. For the neighbour
  case it equals `lon_0`. The redundancy is deliberate --- the same call
  already made for the auxiliary schema: store the value, do not make consumers
  reconstruct it from a convention.
- `lon_0` keeps its honest meaning: release positions of actual particles,
  `(i, j)` for neighbour and `(i, j, displacement)` for auxiliary.
- Replace the `_grid_lonlat` module helper with abstract properties on
  `FlowMap` --- `grid_lon`, `grid_lat`, `advected_centre` --- overridden per
  subclass. A base-class property that branches internally would be the same
  violation wearing a better name; the point is that the type carries the
  information.
- This removes `if "displacement" in advected.dims:` at `grids.py:452`, which
  is verbatim the pattern `AGENTS.md` forbids by name, and the
  `"lon_c" in obj.coords` sniff at `grids.py:101-102`.
- `ftle_ridge_seeds` then reads the canonical coord off the field it is handed,
  with no type to dispatch on and no sniff.

Touches `grids.py`, `tensorlines.py`, the tests, all four examples, and
`docs/notation.md`.

**4. Output metadata.** `name`, `long_name`, `units` on everything returned.
`_assemble_tensor` currently names its result `"tensor"`, so
`deformation_gradient()` and `cauchy_green()` come back indistinguishable ---
fix that. Label the `eig` coord. This is the precondition for vanilla plots in
step 9, and for dropping the hand-written `* 86400.0` that appears in three
examples.

**5. Parameters in physical units.** The reviewer flagged `window=7` as
implicitly sensitive to grid resolution; the same defect runs wider.

- `window` becomes a distance.
- `n_steps` becomes a length in metres --- it is currently a line *length*
  (1500 km) expressed as a step count.
- `lambda_max_min` becomes an FTLE floor in 1/day evaluated against the flow
  map's own `|T|`; as a raw Cauchy-Green eigenvalue it silently tightens or
  loosens as the window changes.
- Decide `quantile`'s fate --- as a fraction of *grid points* it is as
  grid-dependent as `window` was.

Signature changes, so this lands before the tests and examples are written
against them.

**6. `tensorlines` internals: lift, explain, test.** `xi1`, `step`, and `half`
become top-level tested functions or get inlined; same for the nested
`central_diff` and `arm_diff` in `grids.py`. `xi1` is the one the review did
not flag and the one most needing a test --- it holds the entire
eigen/orientation/termination logic.

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

**7. Reprs and `flowmap.lcs(...)`.** Terse summary reprs on `Seed` and
`FlowMap`. `lcs()` computes FTLE once and passes it to ridge-finding, so the
convenience lives in one place and the ridge logic is not duplicated.

### PR C --- prose, examples, and CI

**8. Prose hygiene.** Strip process narrative and design rationale from
reader-facing text:

- the four-section design document at the top of `grids.py` (developer text on
  the import path, duplicating `docs/notation.md`);
- the meta-comment in `__init__.py`;
- `docs/notation.md`'s history of our own repo, and its "the choice is left to
  the implementation session" passage for a choice long since shipped;
- the agent-addressed HTML comments opening `docs/api.md` and
  `docs/architecture.md`;
- the recommendation in `plans/lcs-time-evolution.md` that contradicts the
  decision note directly above it;
- the moved-plan links from `src/` and `tests/`.

Design rationale relocates to `docs/architecture.md` rather than being deleted.

**9. Examples rewrite.** Explicit and inline throughout: `set_lost_to_nan`
written out in each notebook rather than shared, `advect` and `ftle_per_day`
dissolved into straight-line code. Display the datasets --- the reader is at
home in the xarray/CF world. Show both stencils. Drop the round-trip demos and
the contract framing entirely; the tests cover that, and a reader interested in
the design reads the tests. Daily intervals in the evolution example.
Progressbar on. Vanilla plots, now possible given step 4. Drop the pixi
execution instructions --- that is our dev environment, not the reader's
concern.

**10. Data story and CI gating.** Two examples currently claim to run "offline
(bundled currents)", but `examples/data/` is gitignored, nothing in the repo
tracks or generates `cabo_verde_currents_hourly.nc`, and both examples open it.
A fresh clone fails on both, they can never be CI-gated, and their committed
outputs are unreproducible by anyone else. Resolve under the online-each-run
versus download-once distinction, committing no data. Then extend CI to gate
every example that can be gated --- today it runs one of four, so
"a broken example is treated like a failing test" is unenforced for three.
