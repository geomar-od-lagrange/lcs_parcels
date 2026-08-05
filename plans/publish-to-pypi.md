# Plan: publish to PyPI

Get the package to a state worth publishing, then publish it. Covers issues #12,
#13, #18, #21, #22, #25. Excludes #8, #9, #11, #14, #20, #26 by decision.

## Measured state

Checked on 2026-08-04, at `2c71ccb`:

| Check | Result |
|---|---|
| PyPI name `lcs_parcels` | free (404), and so is `lcs-parcels` |
| `python -m build` | sdist and pure-Python wheel build clean |
| Coverage, statement and branch | 99% over 102 tests |
| Uncovered lines | `grids.py:273,452,509`, all `raise NotImplementedError` in abstract methods |

Coverage is not thin. The coverage bullet in #12 was written at the #7 review and
no longer describes the repository; it becomes a CI gate, not a project.

The published name is `lcs_parcels`, matching the import. PyPI normalises it to
`lcs-parcels` in index URLs, so `pip install lcs-parcels` resolves to the same
project and the hyphen form cannot be claimed by anyone else.

Three defects the build surfaced:

- PEP 440 normalises `2026.08.04.1` to `2026.8.4.1`. The git tag, `__version__`,
  and the PyPI version would disagree.
- The sdist ships `.claude/skills/`, `CLAUDE.md`, `AGENTS.md`, `plans/`, and
  `pixi.lock`.
- `requires-python = ">=3.11"` is never tested. Both pixi environments pin 3.13.

## Governance

Two `AGENTS.md` rules change with this plan, because publishing invalidates their
premise. Both edits are in this PR.

- The greenfield rule said the users are ~100% the developers and there is no
  external user base. On PyPI that stops being true. Breaking changes stay
  allowed and deprecation shims stay banned; what is added is that a break is
  recorded in the release notes for the version that makes it.
- The environments rule gave Python 3.13 as a universal pin. Supported versions
  now follow SPEC 0, so the window moves on a published schedule and a floor bump
  is a date lookup.

## Sequencing

Three orderings force the split into separate PRs:

- `pip install lcs_parcels` can only go in the README after the name resolves on
  PyPI.
- The publish workflow is only proven by a release that runs it.
- A Read the Docs link can only go in the README after the site builds.

Everything that changes numbers or signatures lands before the first release.

## PR 1 — metric frame and longitude arithmetic

Closes #18, #13. Both change how positions are laid out and measured; #18 asks
for them to be designed together.

**Decision — metric frame: per-point local east/north.** Drop the shared
standard parallel entirely. Difference each point in its own local frame, using
its own $\cos\phi$. Removes the frame concept from $\nabla F$ and from
tensor-line stepping alike, so there is no longer a "regional domains only"
caveat to explain. Same size of edit as the alternative.

This is the diagonal rescaling of #18,
$\nabla F = \mathrm{diag}(c_1, 1) \cdot \partial(\lambda_1,\phi_1)/\partial(\lambda_0,\phi_0) \cdot \mathrm{diag}(1/c_0, 1)$,
taken to its limit: with each separation read in its own local frame there is no
residual frame to correct. The measured 3.0% median error over 20 degrees and
15% over 60 degrees go to zero rather than shrinking.

**Decision — branch cut: wrap-aware helpers only.** No normalisation on ingest.
Accept whatever longitude convention the user hands us, and do every difference
and every mean through a wrapping helper. Nothing is silently rewritten, and a
user who plots our output gets back the convention they came in with.

Work:

- Replace `_lonlat_to_meters` / `_reference_lonlat` / `_to_meters` with a
  local-frame separation helper: the east/north components in metres between two
  lon/lat pairs, longitude difference wrapped to $[-180, 180]$ and the east
  component scaled by the cosine of the pair's mid-latitude. Both
  `deformation_gradient` implementations difference through it.
- Lay the `AuxiliarySeed` arms with each grid point's own $\cos\phi$, so the arm
  span is exactly $2s$ everywhere instead of only at the grid centroid.
- Circular mean for the auxiliary centroid (`AuxiliaryFlowMap.grid_image`),
  anchored on the first arm so the result keeps the input's branch.
- Forward stepping in `tensorlines.py`: invert the measurement — northward
  $R\,d\phi$, eastward $R\cos\phi_{\mathrm{mid}}\,d\lambda$ about the same
  mid-latitude — instead of dividing by one reference cosine.
- Re-anchor the advected longitudes on their grid point's branch inside
  `FlowMap.image`, so an advection that returns positions wrapped to
  $[-180, 180)$ does not tear under the interpolation.
- Reject an auxiliary separation that spans 90 degrees of longitude or more,
  which otherwise aliases through the wrap into an arm on the far side of the
  pole. The guard trips closer to a pole than $2s/\pi$: 640 m for the default
  1 km arms, 32 km for 50 km arms.
- Tests: the rigid meridional translation from #18, which must stop reporting
  zero FTLE; a seed straddling the antimeridian; a high-latitude seed. None
  exist today. The shared axis fixtures run every operator, grid and tensor-line
  test in all three regions, so the sensitive cases are not one file's problem.

Two things measured during the work, both of which changed the design:

- The direct great-circle step is the wrong forward map here. A great-circle arc
  leaves a heading-invariant direction field at
  $(\delta/R)^2\tan\phi/2$ per step, which accumulates *linearly* in the step
  count: a due-east field at 70 N drifts 3.4 km off its parallel over an 800 km
  line at a 20 km step, and halving the step only halves it. Inverting
  `_separation_m` instead leaves it exactly, and makes the forward and inverse
  operations exact inverses.
- The mid-latitude cosine is a midpoint rule, so its accuracy is set by the
  separation of the *pair* and its latitude, not by the size of the domain. For a
  zonal pair the relative error against the great-circle distance is
  $(\Delta\lambda\sin\phi)^2/24$, crossing $10^{-6}$ at $31.2\ \mathrm{km}/\tan\phi$
  — 54 km at 30 N, 18 km at 60 N, 5.5 km at 80 N, and never at the equator, where
  a parallel is itself a great circle. The default 1 km auxiliary arms sit far
  inside that; a neighbour stencil differences over *two* grid cells and on a
  coarse grid does not, but pays its own finite-difference truncation first. The
  docs state the number rather than claiming the frame is exact.

  A first draft of this plan put that crossing at "36 km at the equator, 11 km at
  60 N, 3 km at 80 N". Those numbers were never measured and the equator entry is
  impossible, since the error term vanishes there. Adversarial review caught it
  after it had propagated into the source docstring.
- Rewrite the *Scope: regional domains only* section of `README.md` and the
  metric-frame part of `docs/numerics.md` to the state after the fix.

No third-party geodesy is involved. The frame is `EARTH_RADIUS_M` and a cosine in
`grids.py`; the fix keeps it that way.

Haversine does not apply here. It solves the inverse problem — great-circle
distance between two given points — and $\nabla F$ needs a local linear map
between tangent spaces, so it needs signed east and north *components*, not a
scalar separation. The forward direct problem is what the tensor-line stepping
wants, and that is written out above.

What stays a regional restriction is the rectilinear-grid assumption, not the
metric: `FlowMap.image` and `shrink_lines` interpolate along the `lon_grid` axis,
so a domain crossing the antimeridian must be seeded on a monotonic longitude
axis (170, 175, 180, 185) rather than a wrapped one (170, 175, 180, -175). That
is the axis, not the arithmetic, and it is what the README says after this PR.

## PR 2 — ridge-selection knobs

Closes #21, #22. Both touch the `ftle_ridge_seeds` signature, so the signature
changes once.

**Decision — `window_m`: keep it, report and warn.** `window_m` stays the
neighbourhood side. Report the implied minimum seed spacing (about `window_m /
2`) and the cell counts used in the returned metadata, and warn when `window_m`
spans fewer than a few grid cells. Nothing silently changes meaning. #21's option
3 — redefining `window_m` as the minimum seed separation — is rejected on that
ground: it would leave existing calls running with a knob that means something
else.

**Decision — ridge selection: `quantile` default, absolute available.** Add an
absolute FTLE floor alongside `quantile`, with `quantile` staying the default.
Nothing changes for existing calls, and the absolute option is there when a run
needs comparability across windows or regions.

Work:

- #21: report the implied minimum seed spacing (about `window_m / 2`; measured
  nearest-neighbour ratios 1.67–1.89) and the cell counts used, in the returned
  metadata. Warn when `window_m` spans fewer than a few grid cells.
- #22: add an absolute FTLE floor alongside `quantile`.

As built:

- `ftle_ridge_seeds(ftle, *, window_m=30_000.0, quantile=None, ftle_min=None)`
  returns an `xr.Dataset` — `lon`/`lat` on a `seed` dim with a `seed` index
  coordinate — where it returned a `(lon, lat)` tuple of arrays. That is the
  metadata carrier the two decisions above need, and it removes the unpacking
  the keyword-only `shrink_lines` seed pair made awkward. Call sites read
  `shrink_lines(fm, seed_lon=seeds["lon"], seed_lat=seeds["lat"])`.
- The dataset's `attrs` carry `long_name`, `selector` (`"quantile"` or
  `"ftle_min"`), `ftle_threshold`, and every key of `_window_geometry`
  (`window_m`, `window_cells_i`, `window_cells_j`, `grid_spacing_i_m`,
  `grid_spacing_j_m`, `min_seed_separation_m`). `_window_cells` is replaced by
  `_window_geometry`, which computes the separation rather than estimating it.
- The warning threshold is *three* cells in either dimension, the point at
  which the local-maximum test stops selecting.
- Passing both `quantile` and `ftle_min` raises `ValueError`; passing neither is
  `quantile=0.90`, exactly as before.
- `FlowMap.hyperbolic_lcs` gained `ftle_min` and forwards it, and the
  ridge-selection attrs (all but the seeds' own `long_name`) ride along on the
  dataset it returns.
- `min_seed_separation_m` is computed off the grid's **smallest** cell, not its
  median one. The first version used the median and was not a bound: adversarial
  review measured it violated by 11% over a 30-degree latitude band and by a
  factor of 3 over 75 degrees, because the window is a count of cells and the
  cells converge poleward. It bounds *strict* maxima; a plateau of exactly equal
  values ties for the windowed maximum at every one of its cells, which is
  asserted as documented behaviour rather than left unnoticed.

## PR 3 — prose pass

Closes #25. Prose only, no behaviour change. Runs after PR 1 and PR 2 so it is
not rewriting text those PRs replace.

Order by likely density: `docs/architecture.md`, `docs/numerics.md`,
`docs/notation.md`, `docs/api.md`, the `src/lcs_parcels/` docstrings, the five
example notebooks, `README.md`, `examples/README.md`, `AGENTS.md`. Enforce LaTeX
over unicode math and a DOI per citation while reading every line. Re-execute any
notebook whose cells change.

Gauge the English against NASA **KSC-DF-107, Revision F** (the Kennedy Space
Center documentation style guide), not against the house rules alone. The
`AGENTS.md` prose rules say what not to do; KSC-DF-107 is a positive standard for
technical writing at sentence level, and this repository's audience — three
readers, each wanting a different thing from the same sentence — is the case it
is written for.

## PR 4 — packaging, CI matrix, coverage gate, Dependabot, badges

The bulk of #12. No release yet.

**CalVer.** Adopt unpadded `YYYY.M.D.N`, which is what PEP 440 normalises to.
Update the release section of `AGENTS.md`. Existing tags `v2026.07.17.1` and
`v2026.08.04.1` stay as they are.

**Metadata** in `pyproject.toml`:

- `authors = [{name = "Henry Adjei"}, {name = "Willi Rath", email = "wrath@geomar.de"}]`
- `license = "MIT"` plus `license-files = ["LICENSE"]`. The current
  `license = { file = "LICENSE" }` puts the whole MIT text in the `License:`
  field; the build already emits Metadata 2.4, which takes the SPDX form.
- `[project.urls]` for Homepage, Repository, Issues.
- `classifiers` and `keywords`.

**sdist.** Add `[tool.hatch.build.targets.sdist]` excluding `.claude`,
`CLAUDE.md`, `AGENTS.md`, `plans`, `pixi.lock`, `.github`.

**Dependency bounds.** `dependencies = ["numpy", "xarray", "scipy"]` is currently
unbounded in both directions. Give each a floor. Do not pin, and do not add an
upper cap: a pin in a library's metadata propagates into every downstream
resolution and stops `lcs_parcels` being co-installable. `pixi.lock` pins the
development environment; `[project]` declares what the library can live with.

All three floors come from SPEC 0. None sits above it:

| | floor | oldest release inside the 2-year window |
|---|---|---|
| numpy | `>=2.2` | 2.1 was 2024-08-18 and drops out this month |
| scipy | `>=1.15` | 1.14.1 was 2024-08-21 and drops out this month |
| xarray | `>=2024.9` | 2024.9.0 was 2024-09-11; 2024.7.0 is five days outside |

These are policy rather than capability. The whole third-party surface is
`np.linalg.eigh`, `np.column_stack`, `xr.apply_ufunc`, `xr.concat` and
`scipy.interpolate.RegularGridInterpolator`; the suite passes on numpy 1.26 and
scipy 1.11. There is no reason to promise support that far back.

The xarray floor was measured at 2025.11 before it was fixed. Everything from
2023.12 to 2025.10 failed the same three tests — `test_diagnostics_are_labelled`
for both seeds and `test_hyperbolic_lcs_output_is_labelled` — because those
versions of `xr.dot` drop the attrs of every dimension coordinate they carry
through, so `i` came back with no `long_name` and output metadata stopped being
the axis label. `xr.dot` was the only affected call; `apply_ufunc`, `concat`,
`broadcast`, `where`, `isel` and arithmetic all preserve attrs on every version
tested.

`xr.dot` stays — it names the contraction, and the tensor is 2x2, so nothing
here is on a hot path. The call is followed by an `assign_coords` that restores
each carried-through coordinate's attrs from the operand. The suite passes at
xarray 2024.9, 2025.1, 2025.10 and 2026.1.

The restore reads its list of coordinates from the operand rather than naming
them. Measured on 2024.9, `xr.dot` strips `i`, `j`, `lon_grid`, `lat_grid`, `t0`
and `T`; naming `i` and `j` would have missed four, because `assert_labelled`
stops at the first failure and never reported the rest.

The fix is in this PR, since it is what lets the floor follow the policy.

Test the floor rather than assert it: a CI job resolving `--resolution
lowest-direct` and running the suite, so a floor that is wrong fails.

**Python range.** Set `requires-python = ">=3.12"` and test 3.12, 3.13 and 3.14.
This is SPEC 0 applied on today's date: a Python is supported for 3 years after
release, so 3.11's drop date was 2025-10-23 and 3.12's is 2026-10-01.

Three facts behind that choice:

- 3.14 was released 2025-10-07 and is currently tested nowhere. The suite passes
  on it unmodified — 102 passed, verified. The gap in coverage is at the top of
  the range, not the bottom.
- numpy 2.5 and scipy 1.18 require Python 3.12, because they follow SPEC 0 too.
  The default pixi environment holds exactly those. A 3.11 floor would mean the
  declared range resolves a dependency stack we never run.
- 3.12 drops out on 2026-10-01, two months out. Expect to move to `>=3.13` then;
  under the SPEC 0 rule in `AGENTS.md` that is a date lookup, not a discussion.

Add pixi features `py312`, `py313`, `py314`, each pinning its interpreter, and
the matching test environments. `examples` stays on 3.13, the newest Parcels v4
supports. Update the environments section of `AGENTS.md`, which states there are
two environments and gives the 3.13 pin as universal, and add SPEC 0 as the rule
that moves these numbers.

CI: keep the OS matrix at 3.13, add ubuntu jobs for 3.12 and 3.14.

**Coverage.** Add `.coverage` and `htmlcov/` to `.gitignore`; neither is there
today. Add `pytest-cov`, a `test-cov` pixi task running
`--cov=lcs_parcels --cov-branch`, and a CI step. Set `exclude_also =
["raise NotImplementedError"]` in the coverage config, which takes the three
abstract-method bodies out and makes 100% the honest number, then gate on
`--cov-fail-under=100`.

**Dependabot.** `github-actions` only. Verified: pixi is not a supported
ecosystem, and Dependabot's pip ecosystem does not read `[tool.pixi.*]`, so it
cannot see the Parcels git dependency and cannot float the pinned SHA. The
`[project]` dependencies carry floors and no caps, so there is nothing else for
it to bump either.

**README.** Add `pip install git+https://github.com/geomar-od-lagrange/lcs_parcels.git`,
so users are not forced through pixi before PyPI exists.

**Badges.** A CI-is-green badge and a license badge. No Codecov: coverage stays a
CI gate with no badge and no third-party service. The PyPI version badge lands
with PR 6, once there is a version to report, and the docs badge with PR 7.

## PR 5 — first release

- Add `.github/workflows/publish.yml`: triggered on a `v*` tag, builds sdist and
  wheel, uploads with PyPI Trusted Publishing (OIDC, no token in secrets).
- Release commit bumping `version` in `pyproject.toml` and `__version__` in
  `src/lcs_parcels/__init__.py` to the same unpadded CalVer string, then the tag.
- Release notes, now that the loosened compatibility rule requires them. PRs 1
  and 2 change `deformation_gradient` output, and the `ftle_ridge_seeds`
  signature *and return type* — a `(lon, lat)` tuple becomes an `xr.Dataset`.
  Those go in the notes for this first version even though nobody can have
  depended on them yet, so the format starts as it continues.

**User action, blocking:** a pending publisher for `lcs_parcels` must be
configured on pypi.org before the tag is pushed. Only the account owner can do
this.

## PR 6 — README install instructions

After the name resolves on PyPI. `pip install lcs_parcels` becomes the primary
instruction, the `git+https://` form stays as the way to get unreleased `main`,
pixi stays as the development path. Add a PyPI version badge.

## PR 7 — Read the Docs

Independent of PR 5 and PR 6; can run in parallel.

- `docs/` is Markdown already, so MkDocs with mkdocs-material is the shorter
  path than Sphinx and MyST.
- Relative links out of `docs/` break on a built site: `docs/api.md:328`,
  `docs/architecture.md:4,14,208`, `docs/notation.md:152,191` point into
  `../src/` and `../plans/`. Resolve each to a repository URL or drop it.
- `.readthedocs.yaml`, a `docs` pixi feature for the build dependencies, and the
  RTD project connected to the repository.
- Add the docs link and badge to `README.md` once the site builds.

**Decision — MkDocs Material.** `docs/` is Markdown, so it builds as-is; Sphinx
with MyST would carry a `conf.py` and an extension for the same pages. The one
thing Sphinx would add is generated API pages from the docstrings, and
`docs/api.md` is hand-written and PR 3 will have just gone over it, so that is
not a gap being filled. Revisit if hand-maintaining `docs/api.md` starts to
drift from the docstrings.

**Open again, and not on the grounds the decision was taken on.** Building the
site surfaced a warning Material for MkDocs prints on every build. MkDocs 1.x
has had no release in 18 months; MkDocs 2.0 is a ground-up rewrite that removes
the plugin system, switches the config from YAML to TOML, takes a closed
contribution model and is currently unlicensed; and Material is incompatible
with it, pins `mkdocs<2`, and is steering users to its own replacement,
Zensical. Material's own post suggests Sphinx for projects that want long-term
stability.

None of that stops the site building — it does build, in `strict` mode, on
mkdocs 1.6.1 and mkdocs-material 9.7.7, and `mkdocs>=1.6,<2` is pinned
explicitly rather than left to Material to enforce. The exposure is small: five
Markdown files, no plugin beyond `pymdownx`, and no autodoc. Switching to Sphinx
with MyST later is an afternoon, not a migration.

So this ships as MkDocs and the choice is recorded as contingent rather than
settled. The trigger to revisit is Zensical reaching a state worth adopting, or
mkdocs 1.x losing a security fix — not the API-page argument above.
