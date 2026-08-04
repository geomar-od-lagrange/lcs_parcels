# Plan: publish to PyPI

Get the package to a state worth publishing, then publish it. Covers issues #12,
#13, #18, #21, #22, #25. Excludes #8, #9, #11, #14, #20, #26 by decision.

## Measured state

Checked on 2026-08-04, at `2c71ccb`:

| Check | Result |
|---|---|
| PyPI name `lcs-parcels` and `lcs_parcels` | free (404 on both) |
| `python -m build` | sdist and pure-Python wheel build clean |
| Coverage, statement and branch | 99% over 102 tests |
| Uncovered lines | `grids.py:273,452,509`, all `raise NotImplementedError` in abstract methods |

Coverage is not thin. The coverage bullet in #12 was written at the #7 review and
no longer describes the repository; it becomes a CI gate, not a project.

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

- Apply the diagonal rescaling from #18:
  $\nabla F = \mathrm{diag}(c_1, 1) \cdot \partial(\lambda_1,\phi_1)/\partial(\lambda_0,\phi_0) \cdot \mathrm{diag}(1/c_0, 1)$,
  in both `NeighborFlowMap.deformation_gradient` and
  `AuxiliaryFlowMap.deformation_gradient`.
- Add a longitude-difference helper wrapping to $(-180, 180]$ and a circular-mean
  helper. Use them at every site listed in #13: `FlowMap.image`, the auxiliary
  centre, `_reference_lonlat`, `_to_meters`/`_lonlat_to_meters`, both
  `deformation_gradient` implementations, and `tensorlines.py:123` and `:148-150`.
- Tests: the rigid meridional translation from #18, which must stop reporting
  zero FTLE; a seed straddling the antimeridian; a high-latitude seed. None
  exist today.
- Rewrite the *Scope: regional domains only* section of `README.md` and the
  metric-frame part of `docs/numerics.md` to the state after the fix.

**Open, needs a decision:** whether longitudes are normalised on ingest or the
dataset carries a branch-cut convention. #13 lists both.

## PR 2 — ridge-selection knobs

Closes #21, #22. Both touch the `ftle_ridge_seeds` signature, so the signature
changes once.

- #21: report the implied minimum seed spacing (about `window_m / 2`; measured
  nearest-neighbour ratios 1.67–1.89) and the cell counts used, in the returned
  metadata. Warn when `window_m` spans fewer than a few grid cells.
- #22: add an absolute FTLE floor alongside `quantile`.

**Open, needs a decision:** #21 option 3 — redefining `window_m` as the minimum
seed separation — is the honest fix and changes the meaning of the default. This
plan proposes options 1 and 2 instead. #22 is a science decision: proposal is
that both selectors are available and `quantile` stays the default.

## PR 3 — prose pass

Closes #25. Prose only, no behaviour change. Runs after PR 1 and PR 2 so it is
not rewriting text those PRs replace.

Order by likely density: `docs/architecture.md`, `docs/numerics.md`,
`docs/notation.md`, `docs/api.md`, the `src/lcs_parcels/` docstrings, the five
example notebooks, `README.md`, `examples/README.md`, `AGENTS.md`. Enforce LaTeX
over unicode math and a DOI per citation while reading every line. Re-execute any
notebook whose cells change.

## PR 4 — packaging, CI matrix, coverage gate, Dependabot

The bulk of #12. No release yet.

**CalVer.** Adopt unpadded `YYYY.M.D.N`, which is what PEP 440 normalises to.
Update the release section of `AGENTS.md`. Existing tags `v2026.07.17.1` and
`v2026.08.04.1` stay as they are.

**Metadata** in `pyproject.toml`:

- `authors = [{name = "Henry Adjei"}, {name = "Willi Rath", email = "rath.willi@googlemail.com"}]`
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

Replacing the one call with `(gradF * gradF.rename(col="col_b")).sum("row",
skipna=False)` removes the version sensitivity. The suite passes fully at xarray
2024.7, 2024.9, 2025.1 and 2025.10. `skipna=False` is load-bearing and not
cosmetic: `xr.dot` does not skip NaN but `.sum()` does by default, so the first
version of this change summed a lost particle as zero and broke
`test_nan_propagates_through_chain` on every xarray including the current one.
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
so users are not forced through pixi before PyPI exists. Add a CI badge.

**Open, needs a decision:** whether to add Codecov for a coverage badge, or keep
coverage as a CI gate with no badge. This plan proposes the gate alone.

## PR 5 — first release

- Add `.github/workflows/publish.yml`: triggered on a `v*` tag, builds sdist and
  wheel, uploads with PyPI Trusted Publishing (OIDC, no token in secrets).
- Release commit bumping `version` in `pyproject.toml` and `__version__` in
  `src/lcs_parcels/__init__.py` to the same unpadded CalVer string, then the tag.
- Release notes, now that the loosened compatibility rule requires them. PRs 1
  and 2 change `deformation_gradient` output and the `ftle_ridge_seeds`
  signature; those go in the notes for this first version even though nobody can
  have depended on them yet, so the format starts as it continues.

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

**Open, needs a decision:** MkDocs Material against Sphinx with MyST. Sphinx
would give generated API pages from the docstrings; `docs/api.md` is hand-written
today and PR 3 will have just gone over it.
