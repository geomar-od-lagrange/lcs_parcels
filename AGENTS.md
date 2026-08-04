# Agent guidelines for LCS-Parcels

Conventions for working in this repository. These are derived from review
feedback and are binding unless a task explicitly overrides them.

## Audience

Everything written here — prose, docstrings, examples, output metadata — is
addressed to one of three readers. Know which one before writing; most rules
below are derivations of this model.

- **The trusting user** runs the tool and takes the numbers at face value. They
  need to know what a function does, what units come back, and how to call it.
- **The inquisitive user** wants to convince themselves the science is right
  *without becoming a developer*. They read the example end to end, display the
  datasets, and plot the intermediate fields. They are at home in the
  xarray/CF world; they are not going to read all of `src/`.
- **The developer** needs the design *decisions* — why the auxiliary grid stores
  its arms explicitly, why ridge-finding takes a field rather than a `FlowMap`.
  That audience is served by `docs/architecture.md` (what the types are and how
  they compose), `docs/numerics.md` (why the numbers come out right: the metres
  frame, the tuning parameters, the degeneracy guard), the plans, and the tests.

Rules that fall out of it:

- **Explain the non-obvious choice; do not re-teach the field.** The inquisitive
  user has read the papers. Say why *this* window, *this* stencil, *this*
  termination criterion — not what an FTLE is. Still, define a term properly
  where it is ambiguous.
- **Indirection is a tax in examples, an asset in library code.** The inquisitive
  user consumes an example line by line with minimal scrolling, so prefer
  explicit and even duplicated code over a helper. Library code is read by the
  developer instead, so factor freely there. This is a consequence of who reads
  what, not a blanket preference for duplication.
- **Reader-facing text carries no process narrative and no design rationale.**
  Module docstrings, `docs/api.md`, and examples address the trusting and
  inquisitive users; how we got here belongs in `docs/architecture.md`,
  `docs/numerics.md` or a plan, and agent-addressed asides belong nowhere.
- **A scope statement is not design rationale.** What the package *is and is
  not* — that it contains no Parcels code, that it emits a particle set and
  ingests advected positions, that you run the advection yourself — stays in the
  README. The trusting user cannot use the tool without it. The cut runs between
  the operational fact and its justification: *that* lon/lat pairs are
  keyword-only, *that* returns carry `units`, *that* `flowmap.ds` is the dataset
  are reader-facing; *why* the pairs are keyword-only, and that these classes
  wrap rather than subclass `xr.Dataset`, are `docs/architecture.md`. Of each
  sentence ask "could they use the package without this?", not "does this sound
  like design?".

## Documentation & writing

- **Short factual statements, never an aphoristic tone.** Write what a thing is
  and what it does. Do not reach for the resonant closing clause, the balanced
  pair of opposites, or the sentence that lands rather than informs. Simple
  test: if it sounds like it could be from *The Road Not Taken*, it is wrong.
  This applies to prose everywhere — docstrings, docs, examples, commit
  messages — and is the rule most often broken in a first draft.
- **Math in Markdown uses LaTeX, not unicode.** Write `$\nabla F$`, `$\xi_i$`,
  `$\lambda_{\max}$`, `$(\nabla F)^\top \nabla F$` with `$...$` (inline) and
  `$$...$$` (display). Do not use unicode math glyphs (`∇ ξ λ ² ᵀ → ±`) in prose
  or tables.
- **Citations always carry a DOI.** Include the DOI and its `https://doi.org/...`
  link whenever you cite a paper.
- **Keep notation in one place.** Symbols and conventions live in a dedicated
  notation doc; don't redefine them ad hoc across files.
- **Docs reflect the actual state.** Sketches and placeholders are replaced by
  real docs once the corresponding code exists. Don't let aspirational docs
  masquerade as current.
- **Plans have a lifecycle.** Design/implementation plans live in `plans/`. Once a
  plan is *fully* implemented, move its file to `plans/done/` to keep the active
  set small. Cross-links into a moved plan **from docs or other plans** are
  allowed to break — don't chase them; an archived plan is a historical record,
  not a maintained reference. The exemption stops there: `src/` and `tests/` do
  not link to plans at all, so there is nothing to break. A plan that is only
  partially done (e.g. a survey with deferred parts) stays in `plans/` until the
  rest lands.

## Examples & notebooks

Examples are written for the inquisitive user: someone verifying the science by
reading and running, not by studying the package. Every rule here follows from
that.

- **Examples stay current with the package.** Everything under `examples/` must
  run against the present API. When you change a signature, dim name, or data
  layout, update (or delete) every affected example in the same change — a broken
  or stale example is treated like a failing test, not a TODO. An example that no
  longer matches the code is worse than no example. This is enforced, not
  aspirational: `tests/test_examples.py` executes each `.py` source as a script,
  one test per example. It assumes its prerequisites are there and fails loudly
  when they are not — run it with `pixi run -e examples test-examples`, or name a
  single test to run just one.
- **Notebooks are jupytext-managed.** The `.py` (py:percent) is the source of
  truth; the paired `.md` and `.ipynb` are generated by `jupytext --sync`. Edit
  only the `.py`, then sync. Commit all three. Commit the `.ipynb` **executed**
  (run it end-to-end and keep the cell outputs) so the rendered example is
  self-demonstrating on the forge; the `.py` and `.md` stay the clean, reviewable
  source. After re-syncing from an edited `.py` (which regenerates a clean
  `.ipynb`), re-execute before committing.
- **Notebooks are human-facing, not scripts.** Lay them out as a sequence of
  small, well-scoped code cells, each doing one step, so the inquisitive user can
  run and follow them top to bottom. Don't dump the whole example into one
  mega-cell.
- **Narrate in markdown cells, not comment blocks.** Use `# %% [markdown]` cells
  for prose between steps; don't explain the flow with long `#` comment blocks
  inside code cells. Keep the prose terse. A single one-line `#` heading atop a
  code cell to label its one step is fine — that's a label, not the "comment
  block" this forbids; when one narrated section breaks into a few small cells,
  prefer a terse per-cell heading comment over a markdown cell per micro-step.
- **Show the essence, cut the scaffolding.** An example exists to demonstrate one
  idea (e.g. how our structures marry a given library); everything not serving
  that idea is noise. Strip incidental engineering — caching layers,
  papermill/parameter cells, domain/config helpers, defensive plumbing — and
  inline a helper that is called once rather than defining it. Being inefficient
  or re-downloading on every run is acceptable in an example; being longer than
  the idea requires is not. This is the opposite of production code: minimize the
  inquisitive user's effort, not the machine's.
- **Explicit over shared, even at the cost of duplication.** Write the same
  recovery kernel out in each notebook rather than importing it from a shared
  helper module. The inquisitive user, scrolling one notebook top to bottom,
  should rarely have to open a second file or scroll back and forth. (Library
  code takes the opposite rule — see [Audience](#audience).)
- **Prefer vanilla plots.** This governs the plot you *first write*: reach for
  `.plot()` / `.plot.pcolormesh()` and accept its defaults, which label axes,
  titles, and colorbars from the object's name, coords, and attrs — that is what
  output metadata is for. Don't *open* with hand-set titles, axis labels,
  colormaps, `vmin`/`vmax`, aspect, or multi-panel styling. Setting a keyword is
  fine once a default turns out to be wrong, or the point being made needs it, or
  it was asked for. This is an anti-over-styling rule, not a ban on ever passing
  an argument.
- **A notebook references only notebooks of lower rank.** Point at the notebook
  the current one builds on, never at one of the same rank, so the reading order
  stays a line and no pair of examples explains itself by the other. The ranks:
  `example_grid_pset` is standalone; `cabo_verde_ftle` < `cabo_verde_lcs` <
  `cabo_verde_lcs_evolution`.
- **No claimed result you haven't seen.** Never write a summary/conclusion cell
  (or "this shows X" prose) without actually running the notebook and reading the
  real output first. State what the run produced, not what you expect it to.

## Environments & dependencies (pixi)

- **Change the environment through the pixi CLI, never by hand-editing the
  manifest.** Use `pixi add`, `pixi add --pypi`, `pixi add --feature <f> ...`,
  `pixi project ...` so pixi edits `pyproject.toml` and `pixi.lock` together.
  Don't hand-write the `[tool.pixi.*]` tables.
- **Environments split on weight.** The `default` env stays **minimal** — the
  core package and its tests only. The `examples` env adds the heavy example
  stack (Parcels v4, `copernicusmarine`, matplotlib) and is pinned to **Python
  3.13**, the newest Parcels v4 supports. Run tests with `pixi run test`; run the
  real examples with `pixi run -e examples ...`. Do not push Parcels/CMEMS/
  plotting deps into the default env. One further env per supported Python
  carries the test matrix.
- **Supported versions follow SPEC 0**, which numpy, scipy, xarray, matplotlib
  and zarr all endorse: drop a Python version 3 years after its release, and a
  core dependency 2 years after its release
  ([SPEC 0](https://scientific-python.org/specs/spec-0000/)). So the window moves
  on a schedule rather than on an argument — bumping `requires-python` or a
  dependency floor is a date lookup against that table. A floor may sit *above*
  the SPEC 0 line where a measured failure puts it there; record the measurement
  next to the floor. It never sits below.
- **Ruff lints and formats everything, and CI gates it.** Run `pixi run lint`
  (`ruff check` plus `ruff format --check`) before handing work over; `ruff
  format` fixes the formatting half. It covers `src/`, `tests/`, and the example
  `.py` sources — the generated `.md`/`.ipynb` and vendored `.claude` skills are
  excluded. Reformatting an example's `.py` desyncs its notebook, so
  `jupytext --sync` after linting, not before: sync takes the *most recently
  modified* representation as the source and will happily overwrite the `.py`
  you just formatted.
- **Parcels is pinned to a git SHA.** Parcels v4 is alpha, so it is a pypi git
  dependency in the `examples` feature pinned to a specific `main` commit. Bump
  the rev deliberately; don't float it.

## Parcels integration (v4)

Parcels code lives only under `examples/`, never in `src/` (the package contains
no Parcels — see the boundary rules above). Notes for writing Parcels examples,
verified against the pinned v4 alpha:

- **Currents → FieldSet:** `copernicusmarine_to_sgrid(fields={"U": ds["uo"],
  "V": ds["vo"]})` then `FieldSet.from_sgrid_conventions(sds, mesh="spherical")`.
  There is no `from_netcdf`/`from_xarray`/`from_data` in v4. The CMEMS coords
  must carry CF `axis` attrs (T/Z/Y/X).
- **Positions are `x`/`y`/`z`, not `lon`/`lat`:** `ParticleSet(fieldset,
  pclass=Particle, x=lon, y=lat, z=z_surface, t=t0_array)`; read finals back as
  `np.asarray(pset.x)` / `pset.y`, aligned to seeding order.
- **Lost particles must be recovered to NaN.** By default an out-of-bounds or
  land-NaN particle aborts the whole run. Append a recovery kernel after
  `AdvectionRK4` that sets lost particles' `x`/`y` to NaN and their state to
  `StatusCode.EndofLoop` (never `StatusCode.Delete`, which shrinks the array and
  breaks alignment), so losses propagate as NaN through the diagnostics.

## Python & xarray

- **Use the high-level, label-based xarray API everywhere.** Prefer `.isel()`,
  `.sel()`, named dims, `.where()`, and broadcasting. Never use positional,
  numpy-style indexing on xarray objects — `ds.lon[:, 2, 2]` is bad;
  `ds.lon.isel(i=2, j=2)` is good. This holds even when you fully control dim
  order.
- **Every returned xarray data array carries `name`, `long_name`, and
  `units`.** The inquisitive user displays our datasets and plots them vanilla,
  so the metadata *is* the axis label, the title, and the colorbar caption. Two
  different quantities never come back under the same `name`. Go CF where it is
  cheap — `units`, `long_name` — and no further: no bounds, no intervals, no
  cell methods.
- **A repeated attribute set is a module constant.** When the same
  `name`/`long_name`/`units` block is attached to more than one object — the
  coordinates, a lon/lat pair — hoist it to a module constant so the copies
  cannot drift apart; a one-off on a single returned field stays inline at the
  return.
- **Never mutate an xarray object in place.** Rebuild it functionally with
  `assign_attrs` / `assign_coords` / `assign` rather than writing
  `x.attrs.update(...)` or `x.attrs["units"] = ...`: pandas has removed nearly
  all of its in-place operations and xarray may well follow.
- **Units are SI.** Return 1/s, metres, seconds; deviate only for a field with a
  strong convention of its own (Sverdrups and the like). Conversion for display
  is the trusting user's call, made visible in the example for the inquisitive
  one, never baked into the package.
- **Let xarray do the work.** Rely on broadcasting (e.g. extra `t0`/`T` axes) and
  NaN propagation (e.g. lost particles) rather than writing special-case
  machinery for what xarray already handles. Don't plan around problems xarray
  solves for free.

## Design & architecture

- **Prefer explicit classes over runtime introspection.** Model distinct
  concepts as distinct types. Don't branch on `"displacement" in ds.dims` or
  similar sniffing to decide behavior.
- **Keep external dependencies at the boundary.** This package contains no
  Parcels code. A `Seed` is **time-free** and *emits* a particle set via
  `Seed.to_parcels_pset()` (a 2-tuple `(lon, lat)`); ingest is
  `Seed.pset_to_flowmap(*, lon, lat, t0, t1) -> FlowMap`, which takes **both**
  `t0` and `t1` and derives the signed window $T = t_1 - t_0$ (so direction is
  `sign(T)`). A `FlowMap` collapses back to a time-free seed via
  `FlowMap.to_seed()`. The package neither imports nor drives Parcels.
- **Avoid over-engineering.** Favor a small, concrete API — a few well-named
  methods — over layered adapters and indirection. Add structure when a concrete
  need appears, not before.
- **Tuning parameters are scale-free.** Express a knob so its default means the
  same thing at another resolution, window, or flow regime: a dimensionless ratio
  where one exists, a physical quantity otherwise. `min_anisotropy` floors
  $\lambda_2/\lambda_1$ and so retunes with neither grid nor window. Physical is
  not sufficient — a floor in 1/day is scale-free in no useful sense, since a
  rate that suits a fast flow is meaningless in a slow one.
- **Same-typed adjacent arguments are keyword-only.** A public entry point that
  takes a lon/lat pair (or any other run of interchangeable-looking arguments)
  takes it by keyword, so the trusting user cannot silently transpose them: a
  swap must be a `TypeError`, not a plausible answer off the coast of nowhere.

## Process & change discipline

- **Follow through on findings; don't triage.** At this scaffolding/design stage
  the contract is small and unimplemented, so fixing everything now is cheap and
  fixing it later is expensive. When you act on review feedback, resolve *all* of
  it and propagate each change through every file it touches — code, tests,
  plans, and docs — leaving nothing half-migrated. Prioritizing or deferring
  issues ("let's do the important ones first") is an antipattern here.
- **PRs land squashed onto a linear `main`.** Merge with squash-and-rebase
  (`gh pr merge --squash --delete-branch`), never a merge commit: the branch's
  review churn — fixup commits, applied suggestions, formatting passes — is
  history the repo does not need, and one commit per PR keeps `main` bisectable.
  The squash message is the PR title and body, so write the PR body as the
  commit message it will become, `Closes #N` included, and let the merge close
  the issues.
- **Breaking changes are allowed, but they are announced.** The package is
  published on PyPI, so there are users who are not developers of it and who did
  not read the PR that changed things under them. Change signatures, data
  layouts, dim names, and file formats when the design improves — that freedom
  stays. What changes is that a break is no longer silent: a release that breaks
  a documented call records what broke and what to write instead, in the release
  notes for that version.
  Still do not add deprecation shims, compatibility aliases, migration code, or
  "legacy" branches. Delete the old form outright and update all call sites; the
  release notes carry the migration, not the code.
  This governs the compatibility contract, not what gets written where: the
  three-reader model above still decides what belongs in which file.

## Releases

- **CalVer, `YYYY.MM.DD.N`.** The version is a date plus a same-day counter `N`
  (start at `1`, bump only for a second release on the same day). No semantic
  version — there is no compatibility contract to signal.
- **The version lives in two places that must stay identical:** `[project]`
  `version` in `pyproject.toml` and `__version__` in
  `src/lcs_parcels/__init__.py`. A release is a single **"Release vYYYY.MM.DD.N"**
  commit that bumps *both*, so the package metadata is self-consistent.
- **Tag the released commit** `vYYYY.MM.DD.N` and push the tag; cut a GitHub
  release from it. The bump commit may ride in the feature PR it releases — no
  separate release PR is needed. The tag, not the branch, is the release of
  record.
- **Never pin the version value in a test.** `test_version` asserts a version
  string exists, not what it is, so a bump needs no test edit.
