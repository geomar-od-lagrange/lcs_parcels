# Pruning degenerate shrink lines

Status: implemented. Measurements below were taken on 2026-08-27 on the
`cabo_verde_lcs` example with throwaway scripts that are not part of the
repository.

## Problem

`ftle_ridge_seeds` places seeds at windowed maxima of the FTLE field. A ridge is
a curve, so any ridge longer than `window_m` receives several seeds along it,
and a wide ridge receives seeds displaced across it. All of those seeds lie on
(nearly) the same $\xi_1$ tensor line, so `shrink_lines` returns bundles of
near-identical curves. The seeding cannot prevent this. Seeds along one ridge are
also useful, because a single trace can terminate early on a degenerate cell and
the others complete the curve. So the duplicates are removed after tracing.

Two phenomena are to be handled differently:

1. Several traces of one ridge. Keep the most prominent one, drop the rest.
2. Two curves that run together over part of their length and then separate.
   Keep both, whole.

## Prior art

Farazmand & Haller 2012, *Computing Lagrangian coherent structures from their
variational theory*, Chaos 22, 013128,
[doi:10.1063/1.3690153](https://doi.org/10.1063/1.3690153), trace strainlines
from a dense seed set and keep, among neighbouring strainlines, the one with the
largest $\lambda_2$ averaged along it. LCS Tool (Onu, Huhn & Haller 2015,
[doi:10.1016/j.jocs.2014.05.002](https://doi.org/10.1016/j.jocs.2014.05.002))
implements a distance-based filter that *trims* the weaker line wherever it runs
within a radius of a stronger one.

## Design

Greedy non-maximum suppression over whole lines.

- **Score.** The line integral of the FTLE along the line,
  $\int \mathrm{FTLE}\,\mathrm{d}s$, with the FTLE interpolated at the line's
  points from the gridded field the seeds were picked from (same
  `RegularGridInterpolator` pattern as the tensor). In a bundle every member has
  the same FTLE per point, so the integral ranks by length and the longest trace
  wins. A line is never dropped in favour of one of its own sub-segments, since
  the superset has the larger integral where the FTLE is non-negative. The mean
  of Farazmand & Haller cannot separate bundle members, and was measured to give
  no gap in the distance distribution (below).
- **Coverage.** Points are embedded on the unit sphere as $(x, y, z)$ and
  distances are chord lengths times the Earth radius, so no projection and no
  standard parallel enter, consistent with `docs/numerics.md`. A point of a
  candidate line is *covered* when its nearest point on the union of the
  already-kept lines is closer than the tube radius `window_m / 2`. The
  candidate's *new length* is the arc length of its segments whose both
  endpoints are uncovered.
- **Rule.** Sort by score, descending. Keep the first line. Keep each further
  line unless it has at least one covered point *and* its new length is below
  `window_m`. A line that shares nothing with a stronger line is always kept,
  whatever its length, so pruning removes duplicates and never acts as a length
  filter.
- **Whole lines.** A line is kept or dropped entire. Trimming the covered stretch
  (LCS Tool) would break lines into segments and the `(line, point)` rectangle
  with them. Under the whole-line rule the type-2 curves above stay in full.
- **One knob, and it already exists.** `window_m` is the resolution below which
  two ridges are one ridge, declared when the seeds were picked. The tube radius
  is `window_m / 2`, the closest two seeds can be, and the minimum new length is
  `window_m`. Neither is a new parameter, and the sweep below shows the kept
  count is flat around that pair.

## Measurement (Cabo Verde, forward map, `window_m` 30 km)

53 seeds, 38 traceable lines. For each line, the directed Hausdorff distance to
the nearest stronger line (max over its points of the distance to that line):

| range | lines |
|---|---|
| 0–10 km | 11 |
| 10–17.5 km | 3 |
| 17.5–22.5 km | 0 |
| 22.5–100 km | 18 |
| over 100 km | 5 |

The cluster below 10 km is RK2 drift plus a few cells across the ridge; the empty
band at 17.5–22.5 km separates it from the tail. The same statistic with the
mean in place of the max has no empty band (13 of 37 below 5 km, then a smear).

Lines kept by the greedy walk:

| tube radius | min new length | kept |
|---|---|---|
| 10 km | 20 km | 24 |
| 15 km | 15 km | 23 |
| 15 km | 30 km | 22 |
| 20 km | 40 km | 18 |
| 15 km | (max-distance rule) | 25 |

The three lines the new-length rule drops beyond the max-distance rule are two
with no new length and one 525 km line with 27 km of new length. The rule as
finally stated above (a line with no covered point is always kept) keeps one 6 km
stub that shares nothing with any other line, so the expected count at
(15 km, 30 km) is 23 rather than 22.

The bundles that remain are curves that run together for 50–150 km and then
separate by more than 30 km (a triple at 26.5 W 16.3 N, a fan at 24.8 W 16 N, a
pair along 22 W). They are phenomenon 2 and stay by design.

## API

```python
prune_shrink_lines(lines, ftle, *, window_m=30_000.0) -> xr.Dataset
```

in `src/lcs_parcels/tensorlines.py`, exported from `lcs_parcels`, listed in
`__all__`.

- `lines` is a `shrink_lines` dataset, `lon`/`lat` on `(line, point)`, NaN past
  termination. Any extra data variables on `line` or `(line, point)` pass through
  by the same row selection.
- `ftle` is the field the seeds were picked from, on `(i, j)` with
  `lon_grid`/`lat_grid`. FTLE interpolated as NaN (off-grid, NaN cell) counts as
  zero in the integral.
- Returns the same dataset restricted to the kept rows, with the `line` coord
  keeping its **original labels**, so a kept line can be matched against the
  unpruned set. Rows that are NaN at every point (untraceable seeds) are dropped
  first and never count as covering anything.
- Two data variables on `line` are added, each with `name`, `long_name`,
  `units`: `ftle_mean` (mean FTLE along the line, 1/s) and `length_m` (arc
  length of the line, m). The ranking is their product, the line integral. The
  two are returned rather than the product because each has a plain meaning and
  a unit, and the ranking rule is one sentence in the docstring.
- Attributes: `window_m`, `tube_radius_m`, `min_new_length_m`, `n_lines_in`,
  `n_lines_dropped`.
- `FlowMap.hyperbolic_lcs` runs the step after `shrink_lines`, always, passing
  its `window_m` (the default when unset). Its returned dataset therefore has no
  all-NaN rows, carries `ftle_mean` and `length_m`, and its `line` labels are
  the seed indices that survived. The docstring and `docs/api.md` say so. Driving
  the steps by hand is the way to see the unpruned set, and the example does.

Implementation notes for the developer:

- The KD-tree (`scipy.spatial.cKDTree`) over the kept lines' points is rebuilt
  per candidate. The candidate count is the seed count, so the cost is
  negligible next to the marching. Issue #11 is unaffected; this is
  post-processing on the returned block.
- Chord versus arc is below 1e-6 relative at 30 km and needs no comment in the
  code.
- The one place the label-based xarray API is left is the same one
  `shrink_lines` already leaves, the per-point geometry in numpy. Row selection
  back into the dataset is `.isel(line=...)`.

## Tests (`tests/test_tensorlines.py`, new section `# --- prune_shrink_lines ---`)

The function takes a lines dataset, so the tests build polylines by hand on a
uniform FTLE field rather than tracing. Cases, each its own test:

1. Two identical copies of a line: one kept, the original `line` label of the
   kept one preserved, `n_lines_dropped == 1`.
2. A line and a sub-segment of it, in a uniform field: the superset is kept, the
   sub-segment dropped, regardless of row order.
3. A line and a copy offset by less than `window_m / 2` all along: one kept.
4. Two lines further than `window_m / 2` apart everywhere: both kept.
5. A line that coincides with a stronger one and then departs for longer than
   `window_m`: both kept. Departing for less than `window_m`: dropped.
6. A short isolated line, shorter than `window_m` and far from everything: kept.
7. All-NaN rows are dropped and do not cover anything (an all-NaN row plus one
   real line returns the one real line).
8. Two coincident lines on either side of the antimeridian (`179.9` and
   `-179.9`) are recognised as one, because the sphere embedding needs no
   wrapping.
9. `ftle_mean`, `length_m`, `line`, and the dataset carry `long_name`/`units`
   (metadata test in `tests/test_metadata.py`, next to
   `test_shrink_lines_output_is_labelled`); `length_m` equals the summed chord
   lengths.
10. `hyperbolic_lcs` returns no all-NaN rows, carries the two variables and the
    pruning attributes, and its kept rows equal
    `prune_shrink_lines(shrink_lines(...), ftle, window_m=...)` run by hand
    (extend `test_hyperbolic_lcs_matches_the_manual_pipeline`).

Existing `hyperbolic_lcs` tests that count lines or index rows may change with
pruning. Adjust them to the pruned semantics with the reason in the test
docstring, and do not weaken them.

## Documentation

- `docs/numerics.md`: a section *Why pruning is a run length in a tube* with the
  measured tables above (the distance distribution, the sweep, the mean-based
  comparison), why the score is the line integral, and why the tube radius and
  the minimum new length are `window_m / 2` and `window_m`.
- `docs/architecture.md`: the walkthrough gains the fourth step; a subsection
  *Why pruning drops whole lines* (the `(line, point)` layout, trimming rejected)
  and *Why pruning takes `window_m`* (same resolution as the seeding, no second
  knob).
- `docs/api.md`: the function under *Hyperbolic LCS: shrink lines*, the change
  to `hyperbolic_lcs()`'s result, rows in the output-metadata table.
- `docs/notation.md`: nothing, unless a symbol is introduced; prefer words.
- `CHANGELOG.md`, *Unreleased*: *Added* `prune_shrink_lines`; *Changed*
  `hyperbolic_lcs()` now prunes, returns no all-NaN rows, and its `line` labels
  are seed indices, with what to write to get the unpruned set.
- `README.md`: unchanged, the one-call snippet still holds.
- The tensorlines module docstring names the third function.

## Examples

- `examples/cabo_verde_lcs.py`: after the `shrink_lines` cells and the
  untraceable-seed count, a section *Pruning the bundles* that calls
  `prune_shrink_lines(repelling_lcs, ftle_forward, window_m=window_m)`, shows
  the kept count against the input count, and plots the dropped lines in light
  grey under the kept ones in red over the FTLE. The later plots use the pruned
  set. The attracting family comes out of `hyperbolic_lcs` already pruned; one
  sentence says so. Prose in markdown cells, one idea per cell, no claimed
  numbers that were not read off the executed output.
- `examples/cabo_verde_lcs_evolution.py`: the sentence listing what
  `hyperbolic_lcs` runs gains the pruning step. Nothing else changes.
- `examples/cabo_verde_ftle.py`, `example_grid_pset.py`: untouched.
- Sync and execute with `pixi run -e examples jupytext --sync --execute` and no
  `MPLBACKEND`; count the `image/png` outputs in each `.ipynb` against the
  previous commit before committing. Run `pixi run lint` before the sync, never
  after.

## Order of work

1. Red: the tests above, failing on import.
2. Green: `prune_shrink_lines`, the export, the `hyperbolic_lcs` change, until
   `pixi run test` and `pixi run lint` pass.
3. Docs and examples, in parallel, then `pixi run -e docs docs-build` and
   `pixi run -e examples test-examples`.
4. One PR, squashed onto `main`; this file moves to `plans/done/` in it.
