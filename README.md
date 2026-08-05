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

## Scope: rectilinear grids, away from the poles

Longitudes are taken in whatever convention you hand in and are never
renormalised, so what comes back sits on the branch that went in. Most of the
package needs a rectilinear grid whose `lon_grid` axis is monotonic, so seed a
domain crossing the antimeridian on `170, 175, 180, 185` rather than on
`170, 175, 180, -175`. On a wrapped axis `shrink_lines` and `hyperbolic_lcs`
raise, and `image` reads the axis as if it were sorted and returns `NaN` or the
wrong cell.

The poles are excluded. `AuxiliarySeedGrid.from_axes` raises `ValueError` when
`aux_separation_m` would span 90 degrees of longitude or more, and latitude is
never folded at 90 degrees, so a tensor line stepped past the pole leaves the
domain instead of continuing across it. The accuracy of a separation, and the
rest of the limits, are set out in
[`docs/numerics.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/docs/numerics.md)
and [`docs/api.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/docs/api.md).

## Install

Python 3.12 or newer:

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
from lcs_parcels import NeighborSeedGrid

# 1. Lay out a seed grid and emit a particle set.
seed = NeighborSeedGrid.from_axes(
    lon=np.linspace(-25.0, -20.0, 6),
    lat=np.linspace(15.0, 20.0, 5),
)
lon_0, lat_0 = seed.to_parcels_pset()

# 2. Advect (lon_0, lat_0) from t0 to t1 with Parcels, which is not part of
#    this package, and collect the final positions.

# 3. Ingest the advected positions into a flow map and diagnose it.
flowmap = seed.pset_to_flowmap(
    lon=lon_advected, lat=lat_advected, t0=t0, t1=t1
)
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
(`.py`/`.md`/`.ipynb`) sharing one source, and the rendered `.ipynb` is the one
to read. They are listed in increasing scope:

- [`get_data`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/get_data.ipynb) downloads the CMEMS subset the
  Cabo Verde notebooks read. Run once, before them. Needs CMEMS credentials.
- [`example_grid_pset`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/example_grid_pset.ipynb) exercises the package API
  on its own, without Parcels.
- [`cabo_verde_ftle`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_ftle.ipynb) shows the Parcels v4 wiring
  and the FTLE from CMEMS currents.
- [`cabo_verde_lcs`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_lcs.ipynb) builds repelling and attracting
  LCS as strain tensor lines.
- [`cabo_verde_lcs_evolution`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/cabo_verde_lcs_evolution.ipynb) evolves an
  extracted LCS as a material curve.

The Parcels examples need the `examples` pixi environment (`pixi install -e
examples`) and a CMEMS currents file you save yourself. See
[`examples/README.md`](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/examples/README.md) for the reading order and the data.

## License

[MIT](https://github.com/geomar-od-lagrange/lcs_parcels/blob/main/LICENSE)
