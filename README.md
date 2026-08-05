# LCS-Parcels

[![CI](https://github.com/geomar-od-lagrange/lcs_parcels/actions/workflows/ci.yml/badge.svg)](https://github.com/geomar-od-lagrange/lcs_parcels/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/LICENSE)

Lagrangian coherent structure (LCS) diagnostics on top of
[Parcels](https://oceanparcels.org/): deformation gradient `grad F`,
Cauchy-Green tensor `(grad F)^T grad F`, its eigen-analysis, and the
finite-time Lyapunov exponent (FTLE), following Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

**This package contains no Parcels code.** It *emits* a particle set to release
and *ingests* the advected positions to diagnose. You run Parcels (or anything
else) in between.

<!-- TODO: Condense to 1-2 sentences. This is a fair warning. But we don't need to include the whole justification and details here. -->
## Scope: rectilinear grids, away from the poles

Longitude differences and means wrap, so a domain crossing the antimeridian is
fine. Longitudes are never normalised on ingest: they come back in the
convention you handed in, and advected positions may arrive on any branch.

`NeighborSeed`, `FlowMap.image` and the tensor-line functions need a rectilinear
grid — `lon_grid` varying along `i`, `lat_grid` along `j` — with a monotonic
`lon_grid` axis, so a domain crossing the antimeridian is seeded on `170, 175,
180, 185` rather than `170, 175, 180, -175`. On a wrapped axis `shrink_lines`
and `hyperbolic_lcs` raise, and `image` reads the axis as if it were sorted.
The traced tensor lines carry no such requirement and cross the antimeridian
freely. `AuxiliaryFlowMap.deformation_gradient` differences a stencil laid
around each grid point, so it also takes a curvilinear grid.

The poles are excluded. `AuxiliarySeed.from_axes` raises `ValueError` when
`aux_separation_m` would span 90 degrees of longitude or more, which happens
closer to a pole than about 0.64 times the arm separation: 640 m for the
default 1 km arms, 32 km for 50 km arms. Latitude is not folded at 90 degrees
either, so a tensor line stepped past the pole leaves the domain instead of
continuing across it.

Separations are measured in metres, and the accuracy of a separation is set by
how far apart the two points are and at what latitude, not by the size or
placement of the domain. Measured against the great-circle distance, a zonal
pair is off by 1e-6 at a separation of 54 km at 30 N, 18 km at 60 N and
5.5 km at 80 N; the full series is in [`docs/numerics.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/docs/numerics.md). On
the equator the error vanishes at every separation. The default 1 km auxiliary
arms sit far inside those thresholds away from the pole; the zonal threshold
falls to 1 km itself only above about 88 N. A neighbour stencil differences over
two
grid cells, so read the series at twice the grid spacing; on a coarse grid the
finite-difference truncation of that same span is the larger error.

## Install

Python 3.12 or newer:

<!-- TODO: Explicit @main? -->
```console
$ pip install git+https://github.com/geomar-od-lagrange/lcs_parcels.git
```

That gets the current `main`. There is no release on PyPI yet.

To work on the package code instead, or to run the examples, use
[pixi](https://pixi.sh):

```console
$ pixi install
$ pixi run test                       # run the test suite, including the Parcels-free example
$ pixi run -e examples get-data       # download the CMEMS subset the Cabo Verde examples read (once)
$ pixi run -e examples test-examples  # execute every example against the current API
```

## Quickstart

```python
import numpy as np
from lcs_parcels import NeighborSeed

# TODO: Drop the time-free (here and whereever it's not absolutely necessary.)
# 1. Lay out a time-free seed grid and emit a particle set.
seed = NeighborSeed.from_axes(
    lon=np.linspace(-25.0, -20.0, 6),
    lat=np.linspace(15.0, 20.0, 5),
)
lon0, lat0 = seed.to_parcels_pset()

# 2. Advect (lon0, lat0) from t0 to t1 with Parcels -- not part of this
#    package -- and collect the final positions (lon1, lat1).

# 3. Ingest the advected positions into a flow map and diagnose it.
flowmap = seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)
ftle = flowmap.ftle()  # xr.DataArray of the FTLE (1/s) on the (i, j) grid

# 4. Or go straight to the hyperbolic LCS curves, FTLE field included.
lcs = flowmap.hyperbolic_lcs()
```

Lon/lat pairs are keyword-only throughout, so a transposed call raises
`TypeError`. Every returned array carries `long_name` and `units`, so
`ftle.plot(x="lon_grid", y="lat_grid")` labels itself.

`print(flowmap)` gives a one-line summary and `flowmap.ds` gives the dataset
itself:

```pycon
>>> print(flowmap)
<NeighborFlowMap 6x5 grid, lon -25.00..-20.00, lat 15.00..20.00, t0 2020-01-01T00:00:00, T +7.0 days>
```

## Examples

Examples live under [`examples/`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/). Each is a jupytext triplet
(`.py`/`.md`/`.ipynb`) sharing one source; the rendered `.ipynb` is the one to
read:

- [`cabo_verde_ftle`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_ftle.ipynb) — the Parcels v4 wiring:
  the FTLE from CMEMS currents. Needs CMEMS credentials.
- [`cabo_verde_lcs`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_lcs.ipynb) — repelling and attracting
  LCS as strain tensor lines.
- [`cabo_verde_lcs_evolution`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_lcs_evolution.ipynb) — an
  extracted LCS evolved as a material curve.
- [`example_grid_pset`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/example_grid_pset.ipynb) — the package API
  exercised on its own, without Parcels.

The Parcels examples need the `examples` pixi environment (`pixi install -e
examples`) and a CMEMS currents file you save yourself; see
[`examples/README.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/README.md) for the reading order and the data.

## License

[MIT](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/LICENSE)
