# Agent guidelines for LCS-Parcels

Conventions for working in this repository. The conventions are derived from
review feedback and are binding unless a task explicitly overrides them.

## Audience

Everything written here (prose, docstrings, examples, output metadata) is
addressed to one of three readers. Know which one before writing; most rules
below are derivations of this model.

- **The trusting user** runs the tool and takes the numbers at face value. They
  need to know what a function does, what units come back, and how to call it.
- **The inquisitive user** wants to convince themselves the science is right
  *without becoming a developer*. They read the example end to end, display the
  datasets, and plot the intermediate fields. They are at home in the
  xarray/CF world; they are not going to read all of `src/`.
- **The developer** needs the design *decisions*: why the auxiliary grid stores
  its arms explicitly, why ridge-finding takes a field rather than a `FlowMap`.
  That audience is served by `docs/architecture.md` (what the types are and how
  they compose), `docs/numerics.md` (why the numbers come out right: the metres
  frame, the tuning parameters, the well-definedness guard), the plans, and the
  tests.

Rules that fall out of it:

- **Explain the non-obvious choice; do not re-teach the field.** The inquisitive
  user has read the papers. Say why *this* window, *this* stencil, *this*
  termination criterion, not what an FTLE is. Still, define a term properly
  where it is ambiguous.
- **Avoid indirection in examples; use it freely in library code.** The
  inquisitive user consumes an example line by line with minimal scrolling, so
  prefer explicit and even duplicated code over a helper. Library code is read
  by the developer instead, so factor freely there. This is a consequence of who
  reads what, not a blanket preference for duplication.
- **Reader-facing text carries no process narrative and no design rationale.**
  Module docstrings, `docs/api.md`, and examples address the trusting and
  inquisitive users; how we got here belongs in `docs/architecture.md`,
  `docs/numerics.md` or a plan, and agent-addressed asides belong nowhere.
- **A scope statement is not design rationale.** What the package *is and is
  not* (that it contains no Parcels code, that it emits a particle set and
  ingests advected positions, that you run the advection yourself) stays in the
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
  This applies to prose everywhere, in docstrings, docs, examples and commit
  messages, and is the rule most often broken in a first draft.
- **A length budget, and it is a number.** A docstring paragraph explains one
  thing in at most three lines. A code comment is at most two. Over budget means
  the material belongs in `docs/numerics.md` or `docs/architecture.md`, or does
  not need writing at all. "It is all true and all relevant" is not a defence:
  the budget is what keeps a docstring usable as reference, and a first draft
  overruns it almost every time. A `# --- section ---` banner is structure
  rather than prose and does not count against the two lines that follow it.
- **Place a sentence about *why* before writing it.** The destinations are
  `docs/numerics.md` (why the numbers are right), `docs/architecture.md` (why
  the types are shaped this way), a test name (what we measured), or nowhere. A
  docstring says what the function does, what it returns, and what would
  surprise the caller. A comment says why *this line* is not the obvious one.
  Neither carries the history of the decision, the alternative that lost, or the
  observation that prompted the feature. This extends the reader-facing rule in
  [Audience](#audience) to every docstring and every comment, which is where the
  rationale actually accumulates.
- **Working correctly is not a feature.** Document behaviour a reader would
  otherwise be surprised by. Do not document that a case any reader assumes is
  handled is in fact handled. The antimeridian is the running example: *that*
  longitudes may arrive in any convention is a fact the caller needs, and *that*
  differences are wrapped to achieve it is one comment at the wrapping site, not
  a paragraph in four docstrings.
- **Punctuation carries its own tics.** No call-out colon introducing a reason
  (`no shared frame: each pair defines its own`). Use a preposition or a
  conjunction. No `--` or em-dash as an aside. A semicolon joining two
  independent clauses becomes a conjunction. The one dash that stays is the one
  introducing a definition on first use, as in `tensorlines.py`'s "A repelling
  LCS is a *shrink line* -- a curve tangent to ...".
- **No downward or lateral references.** A docstring may name what it depends
  on, never what depends on it. A base class does not name its subclasses, a
  helper does not enumerate its call sites, and a module does not explain itself
  through a sibling module. Those references go stale silently, because nothing
  checks them. Naming the other half of a documented 1:1 pair is fine, so
  `AuxiliaryFlowMap` and `AuxiliarySeedGrid` may point at each other.
- **Local names match the data names they hold.** `lon_0`, not `lon0`;
  `lon_advected`, not `lon_arm`. A variable holding a named data variable takes
  that name, so the reader does not have to hold a second vocabulary.
- **Math in Markdown uses LaTeX, not unicode.** Write `$\nabla F$`, `$\xi_i$`,
  `$\lambda_{\max}$`, `$(\nabla F)^\top \nabla F$` with `$...$` (inline) and
  `$$...$$` (display). Do not use unicode math glyphs (`∇ ξ λ ² ᵀ → ±`) in prose
  or tables.
  **`README.md` is the exception, in both directions.** It is the PyPI landing
  page as well as the repository front page, and PyPI renders neither LaTeX nor
  relative links: `$\nabla F$` arrives as those eight literal characters, and
  `](docs/numerics.md)` resolves under `pypi.org`. So the README writes math as
  plain text or a code span (`` `grad F` ``, `1e-6`) and every link in it is
  absolute. Nothing else changes, and the unicode-glyph ban still holds there.
  Check with `readme_renderer`, which is what PyPI uses; the `build` CI job runs
  `twine check --strict` over it.
- **Citations always carry a DOI.** Include the DOI and its `https://doi.org/...`
  link whenever you cite a paper.
- **Keep notation in one place.** Symbols and conventions live in a dedicated
  notation doc; don't redefine them ad hoc across files.
- **Docs reflect the actual state.** Sketches and placeholders are replaced by
  real docs once the corresponding code exists. Don't let aspirational docs
  masquerade as current.
- **`docs/` is a built site as well as a directory.** Sphinx builds it with
  MyST, and `-W` makes every warning a failure, so a broken cross-reference or a
  page missing from a `toctree` fails the `docs` CI job. A new page needs a
  `toctree` entry in `docs/index.md`. A link out of `docs/`, into `src/` or
  `plans/`, is an absolute repository URL, because a relative one resolves
  against the site. Build with `pixi run -e docs docs-build`, or `docs-serve` to
  watch.
- **Nothing on the site is a maintained second copy.** The home page includes
  `README.md`, the examples land on `examples/README.md` and the executed
  notebooks, and `docs/reference.md` generates the API reference from the
  docstrings. `docs/examples/` and `docs/generated/` are produced at build time
  and gitignored. Duplicated *output* is free; duplicated *effort* is what to
  avoid, so a hand-written page that restates a docstring is the thing to catch.
  `docs/api.md` is deliberately not that: it is the written guide for onboarding
  a developer or an agent, and the generated reference carries the signatures.
- **Plans have a lifecycle.** Design/implementation plans live in `plans/`. Once a
  plan is *fully* implemented, move its file to `plans/done/` to keep the active
  set small. Cross-links into a moved plan **from docs or other plans** are
  allowed to break, so don't chase them. An archived plan is a historical record,
  not a maintained reference. The exemption stops there: `src/` and `tests/` do
  not link to plans at all, so there is nothing to break. A plan that is only
  partially done (e.g. a survey with deferred parts) stays in `plans/` until the
  rest lands.

## Examples & notebooks

Examples are written for the inquisitive user: someone verifying the science by
reading and running, not by studying the package. Every rule here follows from
that reader.

- **Examples stay current with the package.** Everything under `examples/` must
  run against the present API. When you change a signature, dim name, or data
  layout, update (or delete) every affected example in the same change. A broken
  or stale example is treated like a failing test, not a TODO. This is enforced:
  `tests/test_examples.py` executes each `.py` source as a script, one test per
  example. It assumes its prerequisites are there and fails loudly when they are
  not. Run it with `pixi run -e examples test-examples`, or name a single test
  to run just one.
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
  code cell to label its one step is fine, since that's a label, not the "comment
  block" this forbids; when one narrated section breaks into a few small cells,
  prefer a terse per-cell heading comment over a markdown cell per micro-step.
- **Show the essence, cut the scaffolding.** An example exists to demonstrate one
  idea (e.g. how our structures marry a given library); everything not serving
  that idea is noise. Strip incidental engineering (caching layers,
  papermill/parameter cells, domain/config helpers, defensive plumbing) and
  inline a helper that is called once rather than defining it. Being inefficient
  or re-downloading on every run is acceptable in an example; being longer than
  the idea requires is not. Unlike production code, an example is written to
  minimize the inquisitive user's effort rather than the machine's.
- **Explicit over shared, even at the cost of duplication.** Write the same
  recovery kernel out in each notebook rather than importing it from a shared
  helper module. The inquisitive user, scrolling one notebook top to bottom,
  should rarely have to open a second file or scroll back and forth. (Library
  code takes the opposite rule; see [Audience](#audience).)
- **Prefer vanilla plots.** This governs the plot you *first write*: reach for
  `.plot()` / `.plot.pcolormesh()` and accept its defaults, which label axes,
  titles, and colorbars from the object's name, coords, and attrs, which is what
  output metadata is for. Don't *open* with hand-set titles, axis labels,
  colormaps, `vmin`/`vmax`, aspect, or multi-panel styling. Setting a keyword is
  fine once a default turns out to be wrong, or the point being made needs it, or
  it was asked for. This is an anti-over-styling rule, not a ban on ever passing
  an argument.
- **A notebook references only notebooks of lower rank.** Point at the notebook
  the current one builds on, never at one of the same rank, so the reading order
  stays linear and no two examples explain themselves by each other. The ranks:
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
- **Environments split on weight.** The `default` env stays **minimal**, holding the
  core package and its tests only. The `examples` env adds the heavy example
  stack (Parcels v4, `copernicusmarine`, matplotlib) and is pinned to **Python
  3.13**, the newest Parcels v4 supports. Run tests with `pixi run test`; run the
  real examples with `pixi run -e examples ...`. Do not push Parcels/CMEMS/
  plotting deps into the default env.
- **One environment per supported Python carries the test matrix.** Features
  `py312`, `py313` and `py314` pin nothing but the interpreter, and the
  environments of the same name pair each with the default feature. `default`
  uses `py313`, so `pixi run test` and the examples agree on an interpreter.
  Adding or dropping a Python version means a feature, an environment and a line
  in the CI matrix, and `requires-python` and the `Programming Language ::`
  classifiers move with it.
- **Supported versions follow SPEC 0**, which numpy, scipy, xarray, matplotlib
  and zarr all endorse: drop a Python version 3 years after its release, and a
  core dependency 2 years after its release
  ([SPEC 0](https://scientific-python.org/specs/spec-0000/)). The window
  therefore moves on a schedule: bumping `requires-python` or a dependency floor
  is a date lookup against that table, not a judgement call. A floor may sit
  *above* the SPEC 0 line where a measured failure puts it there; record the
  measurement next to the floor. It never sits below.
- **Ruff lints and formats everything, and CI gates it.** Run `pixi run lint`
  (`ruff check` plus `ruff format --check`) before handing work over; `ruff
  format` fixes the formatting half. It covers `src/`, `tests/`, and the example
  `.py` sources, while the generated `.md`/`.ipynb` and vendored `.claude` skills are
  excluded. Reformatting an example's `.py` desyncs its notebook, so
  `jupytext --sync` after linting, not before: sync takes the *most recently
  modified* representation as the source and will overwrite the `.py` you just
  formatted.
- **Parcels is pinned to a git SHA.** Parcels v4 is alpha, so it is a pypi git
  dependency in the `examples` feature pinned to a specific `main` commit. Bump
  the rev deliberately; don't float it.

## Parcels integration (v4)

Parcels code lives only under `examples/`, never in `src/` (the package contains
no Parcels; see the boundary rules above). Notes for writing Parcels examples,
verified against the pinned v4 alpha:

- **Currents to FieldSet:** `copernicusmarine_to_sgrid(fields={"U": ds["uo"],
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
  numpy-style indexing on xarray objects. `ds.lon[:, 2, 2]` is bad and
  `ds.lon.isel(i=2, j=2)` is good. This holds even when you fully control dim
  order.
- **Every returned xarray data array carries `name`, `long_name`, and
  `units`.** The inquisitive user displays our datasets and plots them vanilla,
  so the metadata *is* the axis label, the title, and the colorbar caption. Two
  different quantities never come back under the same `name`. Go CF where it is
  cheap (`units`, `long_name`) and no further: no bounds, no intervals, no
  cell methods.
- **A repeated attribute set is a module constant.** When the same
  `name`/`long_name`/`units` block is attached to more than one object (the
  coordinates, a lon/lat pair), hoist it to a module constant so the copies
  cannot drift apart; a one-off on a single returned field stays inline at the
  return.
- **Never mutate an xarray object in place.** Rebuild it functionally with
  `assign_attrs` / `assign_coords` / `assign` rather than writing
  `x.attrs.update(...)` or `x.attrs["units"] = ...`: pandas has removed nearly
  all of its in-place operations and xarray may well follow.
- **Units are SI.** Return 1/s, metres, seconds; deviate only for a field with a
  strong convention of its own (Sverdrups and the like). Conversion for display
  is the trusting user's call, and the examples show it being made for the
  inquisitive user. The package never converts.
- **Let xarray do the work.** Rely on broadcasting (e.g. extra `t0`/`T` axes) and
  NaN propagation (e.g. lost particles) rather than writing special-case
  machinery for what xarray already handles. Don't plan around problems xarray
  solves for free.

## Design & architecture

- **Prefer explicit classes over runtime introspection.** Model distinct
  concepts as distinct types. Don't branch on `"displacement" in ds.dims` or
  similar sniffing to decide behavior.
- **Keep external dependencies at the boundary.** This package contains no
  Parcels code. A `SeedGrid` carries no time and *emits* a particle set via
  `SeedGrid.to_parcels_pset()` (a 2-tuple `(lon, lat)`); ingest is
  `SeedGrid.pset_to_flowmap(*, lon, lat, t0, t1) -> FlowMap`, which takes **both**
  `t0` and `t1` and derives the signed window $T = t_1 - t_0$ (so direction is
  `sign(T)`). A `FlowMap` collapses back to a seed grid via
  `FlowMap.to_seed()`. The package neither imports nor drives Parcels.
- **Avoid over-engineering.** Favor a small, concrete API of a few well-named
  methods over layered adapters and indirection. Add structure when a concrete
  need appears, not before.
- **Tuning parameters are scale-free.** Express a knob so its default means the
  same thing at another resolution, window, or flow regime: a dimensionless ratio
  where one exists, a physical quantity otherwise. `min_anisotropy` floors
  $\lambda_2/\lambda_1$ and so retunes with neither grid nor window. Physical is
  not sufficient, because a floor in 1/day is not scale-free in any useful sense, since
  a rate that suits a fast flow is meaningless in a slow one.
  The rule governs what a knob *defaults* to. A parameter that is deliberately
  not scale-free is allowed where an analysis needs exactly that. `ftle_min`
  sets the ridge floor as an absolute FTLE so several runs can be compared
  against one threshold, but it is opt-in beside a scale-free default
  (`quantile`), and the reason it exists is written down for the developer
  (`docs/architecture.md`, "Why an absolute FTLE floor exists at all").
- **Same-typed adjacent arguments are keyword-only.** A public entry point that
  takes a lon/lat pair (or any other run of interchangeable-looking arguments)
  takes it by keyword, so the trusting user cannot silently transpose them: a
  swap must raise `TypeError` rather than return a plausible-looking answer at
  the wrong location.

## Process & change discipline

- **Follow through on findings; don't triage.** At this scaffolding/design stage
  the contract is small and unimplemented, so fixing everything now is cheap and
  fixing it later is expensive. When you act on review feedback, resolve *all* of
  it and propagate each change through every file it touches (code, tests,
  plans, and docs), leaving nothing half-migrated. Prioritizing or deferring
  issues ("let's do the important ones first") is an antipattern here.
- **PRs land squashed onto a linear `main`.** Merge with squash-and-rebase
  (`gh pr merge --squash --delete-branch`), never a merge commit: the branch's
  review churn (fixup commits, applied suggestions, formatting passes) is
  history the repo does not need, and one commit per PR keeps `main` bisectable.
  The squash message is the PR title and body, so write the PR body as the
  commit message it will become, `Closes #N` included, and let the merge close
  the issues.
- **Breaking changes are the norm, but they are announced.** This is
  research-grade alpha code. Break the API whenever the design improves, in
  signatures, data layouts, dim names or file formats, and never keep backward
  compatibility that complicates the code. No deprecation shims, no compatibility
  aliases, no migration code, no "legacy" branches: delete the old form outright
  and update all call sites.
  What publishing on PyPI changes is only that a break now reaches people who did
  not read the PR that made it. So a release that breaks a documented call says
  what broke and what to write instead, in that version's release notes. The
  migration instructions go in the release notes; no migration code goes in the
  package.
  This governs the compatibility contract, not what gets written where: the
  three-reader model above still decides what belongs in which file.

## Releases

- **CalVer, `YYYY.M.D.N`, unpadded.** The version is a date plus a same-day
  counter `N` (start at `1`, bump only for a second release on the same day). No
  semantic version, because there is no compatibility contract to signal.
  Write the month and day without a leading zero, because that is the form PEP
  440 normalises to, and a padded `v2026.08.04.1` would therefore be published
  as `2026.8.4.1` and disagree with its own tag. The two tags predating this
  rule, `v2026.07.17.1` and `v2026.08.04.1`, stay as they are.
- **The tag is the only place a version is written.** `hatch-vcs` derives
  `[project] version` from the git tag at build time, and `__version__` reads it
  back off the installed distribution. Nothing in the source tree carries a
  version literal, so there is no bump commit and nothing to keep in sync.
- **Releasing is tagging.** Tag the commit `vYYYY.M.D.N` and push the tag.
  `publish.yml` builds it, waits for an approval on the `pypi` environment, then
  publishes over Trusted Publishing and cuts the GitHub release from this
  version's `CHANGELOG.md` section. Only `v*` tags can deploy that environment.
- **A build between tags is a dev version**, `2026.8.5.1.dev4+g1a2b3c4`, naming
  the commit. A checkout with no tags at all falls back to `0.0.0`, which the
  publish workflow rejects rather than uploading.
- **A release that breaks a documented call says so in its release notes**, per
  the compatibility rule above: what broke and what to write instead.
- **Never pin the version value in a test.** `test_version` asserts a version
  string exists and `test_version_is_pep440` that it parses, neither of which a
  release touches.
