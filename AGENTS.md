# Agent guidelines for LCS-Parcels

Conventions for working in this repository. These are derived from review
feedback and are binding unless a task explicitly overrides them.

## Documentation & writing

- **Math in Markdown uses LaTeX, not unicode.** Write `$\nabla F$`, `$\xi_i$`,
  `$\lambda_{\max}$`, `$(\nabla F)^\top \nabla F$` with `$...$` (inline) and
  `$$...$$` (display). Do not use unicode math glyphs (`∇ ξ λ ² ᵀ → ±`) in prose
  or tables.
- **Citations always carry a DOI.** Include the DOI and its `https://doi.org/...`
  link whenever you cite a paper.
- **Keep notation in one place.** Symbols and conventions live in a dedicated
  notation doc; don't redefine them ad hoc across files.
- **Docs reflect the actual state.** Sketches and placeholders (e.g.
  `docs/api.md`) are replaced by real docs once the corresponding code exists.
  Don't let aspirational docs masquerade as current.

## Python & xarray

- **Use the high-level, label-based xarray API everywhere.** Prefer `.isel()`,
  `.sel()`, named dims, `.where()`, and broadcasting. Never use positional,
  numpy-style indexing on xarray objects — `ds.lon[:, 2, 2]` is bad;
  `ds.lon.isel(i=2, j=2)` is good. This holds even when you fully control dim
  order.
- **Let xarray do the work.** Rely on broadcasting (e.g. extra `t0`/`T` axes) and
  NaN propagation (e.g. lost particles) rather than writing special-case
  machinery for what xarray already handles. Don't plan around problems xarray
  solves for free.

## Design & architecture

- **Prefer explicit classes over runtime introspection.** Model distinct
  concepts as distinct types. Don't branch on `"displacement" in ds.dims` or
  similar sniffing to decide behavior.
- **Keep external dependencies at the boundary.** This package contains no
  Parcels code. Provide objects that *emit* particle sets
  (`.to_parcels_pset()`) and factory methods that *ingest* results
  (`from_parcels_pset_lon_lat(...)`). The package neither imports nor drives
  Parcels; the seed grid owns its release time `t0`, and ingest takes the end
  time `t1` and derives the signed window `T = t1 - t0` (so direction is
  `sign(T)`).
- **Avoid over-engineering.** Favor a small, concrete API — a few well-named
  methods — over layered adapters and indirection. Add structure when a concrete
  need appears, not before.

## Process & change discipline

- **Follow through on findings; don't triage.** At this scaffolding/design stage
  the contract is small and unimplemented, so fixing everything now is cheap and
  fixing it later is expensive. When you act on review feedback, resolve *all* of
  it and propagate each change through every file it touches — code, tests,
  plans, and docs — leaving nothing half-migrated. Prioritizing or deferring
  issues ("let's do the important ones first") is an antipattern here.
- **Greenfield: the user is the developer; no backward compatibility.** This is a
  specialized research tool whose users are (to ~100%) its developers; there is
  no external user base and no compatibility contract. Change signatures, data
  layouts, dim names, and file formats freely when the design improves. Do not
  add deprecation shims, compatibility aliases, migration code, or "legacy"
  branches — delete the old form outright and update all call sites.
