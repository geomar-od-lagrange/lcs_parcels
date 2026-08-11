# Review: one interpolator, four classes

Measurements against `one-interpolator.md`, taken 2026-08-08 at `f5a3a5a`,
darwin arm64, scipy 1.18.0, numpy 2.5.1. Scripts were throwaway and the numbers
reproduce. Two of that plan's seven items are worth doing, one of them
differently from how it is written, and the collapse is not.

## Reproducing the headline table

`one-interpolator.md` claims `LinearNDInterpolator.__call__` is 30x to 1620x
slower than the same interpolant computed directly, and its section "Origin of
the plan" rests on that. Measured over 300 000 scattered query points:

| grid points | plan's figure | measured `__call__` | measured direct | ratio |
|---|---|---|---|---|
| $10^4$ | 156.5 s | 0.99 s | 0.39 s | 2.5x |
| $9 \times 10^4$ | | 6.85 s | 1.47 s | 4.6x |
| $10^6$ | 15 617 s | 71.8 s | 12.6 s | 5.7x |

The plan's column is about 300 times the triangulation build in its own tables
($156.5 / 0.520 = 301$ and $15617 / 53.27 = 293$). That ratio is what a
benchmark rebuilding the interpolator inside its 300-seed loop produces. Counting the
`tri.transform` build that the direct path needs and `LinearNDInterpolator` does
not, the cold end-to-end ratio is about 1.1x.

PR #47's own measurement stands. Timing `shrink_lines` end to end on identical
fields with 58 seeds and 1500 km lines at 3 km steps, changing only which
`_interpolator` runs, confirms it. The ratio is 55x at 19 600 grid points, 247x
at 102 400, and 1385x at $10^6$.

The amortisation section then compares the wrong pair. It scores a cached
triangulation against a rebuilt one and reports 5.0x and 7.8x. The decision on
the table is whether to delete the rectilinear path, so the comparison that
bears on it is against bilinear, which builds nothing and caches nothing.
"Geometry is worth 19.8 maps" is true and does not answer it.

## Measuring the warm start's cost

The plan's warm-start table holds, measured at 1.6x, 1.9x and 3.8x bilinear
against its 1.7x, 2.0x and 2.0x. It is taken with the triangulation built and
`tri.transform` forced, and that setup is 47.7 s at $10^6$ against 0.092 s of
tracing. Seeding the walk from a k-d tree over the grid points, and taking
barycentric weights from the triangle vertices per query point, touches neither
`find_simplex` nor `tri.transform`. A `shrink_lines`-shaped workload of 58
points over 500 steps, build included:

| grid points | as shipped | plan items 1 and 2 | k-d seeded | bilinear |
|---|---|---|---|---|
| 19 600 | 2.44 s | 1.05 s | 0.13 s | 0.021 s |
| 102 400 | 10.73 s | 5.46 s | 0.63 s | 0.022 s |
| $10^6$ | 71.6 s | 54.2 s | 6.59 s | 0.024 s |

It agrees with `LinearNDInterpolator` to $4.4 \times 10^{-16}$ over 200 000
queries and reproduces its NaN pattern at the hull boundary exactly. Its warm
result is bit-identical to a cold search over a 500-step march. The k-d tree is
a structure the package already builds for the windowing, and the
vertex-to-simplex map costs 14 ms at $10^6$.

## Measuring windowing memory

Resident growth through `neighborhood_maximum` at a 15 km radius measures
0.24 GiB at 102 400 grid points and 2.41 GiB at $10^6$. The plan quotes
0.32 GiB of int64 pair indices. The difference is that `query_ball_point`
returns Python lists of Python ints before `np.concatenate` runs. Its 1/50
degree row is therefore about 38 GiB rather than 5.05 GiB, and the wall
arrives one resolution earlier than its table says.

## What to do

Chunk `query_ball_point` (item 7). It is a regression already in `main`'s path
rather than a design question, it is contained, and it adds no layout branch.
Leave `bottleneck` and a restored rolling window alone, which buy a
rectilinear-only win by reintroducing the branch #47 removed, and leave
bin-and-prune until someone runs 1/100 degree.

Rewrite the scattered interpolator k-d seeded rather than as items 1 and 2
specify. Item 3 then goes, because the interpolator can hold its own last
simplex and re-seed when the query count changes. This design leaves
`f(points) -> values` unchanged, moves neither `image` nor the tensor-line code,
and is observable nowhere given the bit-identity above. Item 6 is more important
than question 1 concludes, because the build is now the whole cost.
Per-flow-map caching is still the right default.

## What not to do

Items 4 and 5, the collapse. At its best the scattered path is 6x bilinear at 19 600 grid points, 28x at
102 400 and 275x at $10^6$. The last of those figures is almost entirely the
Delaunay build. Caching amortises it to about 29x over a ten-map session, and
never removes it. Keeping the rectilinear path costs one abstract method and
about 25 lines. The plan retracts the recommendation in its own opening. Its
basin-scale section shows the Delaunay is the one structure of the three that
does not tile cleanly. Its question 9 asks whether deleting the path on the
roadmap is wise.

The radius-$r$ ring gradient (question 10). It needs about 1/200 degree before a
1 km neighbourhood exists, is knife-edged in $r$ on a lattice, and would be a
fourth pair carrying its own flow-map contract. Question 11 goes with it.

## What is left

`one-interpolator.md` needs reconciling. Its sections "Term growth at basin
scale" and "A gradient from the Delaunay ring" stand. "Origin of the plan"
rests on the table above and needs rewriting, and "The change" items 4 and 5
with the propagation paragraph that follows them go.

Questions 2, 5, 7 and 8 stop existing without the collapse. On question 3, keep
`Neighbor*`, because one particle per grid point against four is the binding
constraint at basin scale. On question 4, leave `image` cold, the k-d seed
making the one-shot case cheap enough. On question 6, a fixed block. Three
contained changes remain, with no public API movement.
