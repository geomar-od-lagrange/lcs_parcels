# LCS-Parcels

Lagrangian coherent structure (LCS) diagnostics on top of
[Parcels](https://oceanparcels.org/): deformation gradient $\nabla F$,
Cauchy-Green tensor $(\nabla F)^\top \nabla F$, its eigen-analysis, and the
finite-time Lyapunov exponent (FTLE), following Haller (2015),
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322).

**This package contains no Parcels code.** It sits on either side of an
advection run: it *emits* a particle set to release, and *ingests* the advected
positions to diagnose. You run Parcels (or anything else) in between.

## Install

The project is managed with [pixi](https://pixi.sh):

```console
$ pixi install
$ pixi run test          # run the test suite
$ pixi run check-example # execute the example against the current API
```

## Quickstart

```python
import numpy as np
from lcs_parcels import NeighborSeed

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
lcs = flowmap.lcs()
```

Lon/lat pairs are keyword-only throughout, so a transposed call raises instead of
quietly diagnosing the wrong ocean. Every returned array carries `long_name` and
`units`, so `ftle.plot(x="lon_grid", y="lat_grid")` labels itself.

`print(flowmap)` gives a one-line summary and `flowmap.ds` gives the dataset
itself — these objects *wrap* an `xr.Dataset` rather than subclass one:

```pycon
>>> print(flowmap)
<NeighborFlowMap 6x5 grid, lon -25.000..-20.000, lat 15.000..20.000, t0 2020-01-01T00:00:00, T +7.00 days (forward/repelling)>
```

Examples live under [`examples/`](examples/). Each is a jupytext triplet
(`.py`/`.md`/`.ipynb`) sharing one source; the rendered `.ipynb` is the one to
read:

- [`example_grid_pset`](examples/example_grid_pset.ipynb) — the package API
  exercised on its own, without Parcels.
- [`cabo_verde_ftle`](examples/cabo_verde_ftle.ipynb) — the Parcels v4 wiring:
  the FTLE from CMEMS currents. Needs CMEMS credentials.
- [`cabo_verde_lcs`](examples/cabo_verde_lcs.ipynb) — repelling and attracting
  LCS as strain tensor lines. Offline (bundled currents).
- [`cabo_verde_lcs_evolution`](examples/cabo_verde_lcs_evolution.ipynb) — evolve
  an extracted LCS as a material curve. Offline.

The Parcels examples need the `examples` pixi environment (`pixi install -e examples`).

## License

[MIT](LICENSE)
