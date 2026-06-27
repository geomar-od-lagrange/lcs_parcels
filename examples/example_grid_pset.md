---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: py:percent,md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Grid <-> particle-set round-trip

Seed a grid, emit it as a flat particle set, then ingest advected positions
back onto the grid -- the package's Parcels boundary, exercised here *without*
Parcels. We feed the emitted positions straight back, so the round trip is the
identity and we can check `to_parcels_pset` / `from_parcels_pset_lon_lat` are
lossless inverses.

```python
import numpy as np

from lcs_parcels import AuxiliaryGrid, NeighborGrid

t0 = np.datetime64("2020-01-01")
t1 = np.datetime64("2020-01-08")  # +7 day window; sign(T) > 0 -> repelling LCS
lon_axis = np.linspace(-25.0, -20.0, 6)
lat_axis = np.linspace(15.0, 20.0, 5)
```

## Seed a NeighborGrid

`from_axes` broadcasts the 1-D axes into curvilinear reference positions
`lon_0(i, j)` / `lat_0(i, j)` and records the release time `t0`.

```python
seed = NeighborGrid.from_axes(lon_axis, lat_axis, t0=t0)
seed.ds
```

## Emit a particle set

`to_parcels_pset` flattens the reference positions to plain `(lon, lat)` lists,
one entry per grid point -- ready to hand to a Parcels `ParticleSet`.

```python
lon, lat = seed.to_parcels_pset()
print(f"{len(lon)} particles ({lon_axis.size} x {lat_axis.size})")
print("first three lon:", lon[:3])
```

## Ingest advected positions

`from_parcels_pset_lon_lat` reattaches the flat positions and derives the
signed window `T = t1 - t0`. With Parcels these would be the advected outputs;
here they are the emitted positions, so the ingested grid equals the seed.

```python
advected = NeighborGrid.from_parcels_pset_lon_lat(seed, lon, lat, t1=t1)
advected.ds
```

Round-trip checks: the `(i, j)` grid is recovered, `T` is the signed window,
and the advected `lon`/`lat` match the reference positions (identity input).

```python
print("dims:", dict(advected.ds.sizes))
print("T:", advected.ds["T"].values)
print("lon matches reference:", np.allclose(advected.ds["lon"], seed.ds["lon_0"]))
print("lat matches reference:", np.allclose(advected.ds["lat"], seed.ds["lat_0"]))
```

## AuxiliaryGrid: four arms per point

The auxiliary stencil places four arms (`displacement = [east, north, west,
south]`) around each grid point. The arms are the *explicit* reference
positions `lon_0`/`lat_0` -- so the dataset is self-sufficient -- and the
grid-point centres (where the FTLE is reported) are kept separately as
`lon_c`/`lat_c`.

```python
aux_seed = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=t0)
aux_lon, aux_lat = aux_seed.to_parcels_pset()
print(f"{len(aux_lon)} particles = {lon_axis.size} x {lat_axis.size} x 4 arms")
```

```python
aux = AuxiliaryGrid.from_parcels_pset_lon_lat(aux_seed, aux_lon, aux_lat, t1=t1)
print("reference arms  lon_0:", aux.ds["lon_0"].dims)
print("advected arms   lon  :", aux.ds["lon"].dims)
print("diagnostic centres lon_c:", aux.ds["lon_c"].dims)
```

Re-emitting the ingested auxiliary grid reproduces the same particle set --
`to_parcels_pset` emits the explicit `lon_0`/`lat_0` arms directly.

```python
re_lon, re_lat = aux.to_parcels_pset()
print("re-emitted pset matches:", np.allclose(re_lon, aux_lon) and np.allclose(re_lat, aux_lat))
```
