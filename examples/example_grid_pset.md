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

# Seed <-> flow-map round-trip

Seed a grid, emit it as a flat particle set, then ingest advected positions
back -- the package's Parcels boundary, exercised here *without* Parcels. A
time-free `Seed` emits the particle set; ingest consumes the advected positions
and the window $T = t_1 - t_0$ to produce a `FlowMap`. We feed the emitted
positions straight back, so the flow map $F_{t_0}^{t_1}$ is the identity and we
can check that emit/ingest are lossless inverses.

```python
import numpy as np

from lcs_parcels import AuxiliarySeed, NeighborSeed
```

## One release window

The seed is time-free; the window enters only at ingest. We define a single
`t0` and thread the *same* variable into both the (mocked) release and the
`pset_to_flowmap` ingest -- the "`t0` must agree at both ends" contract.

```python
t0 = np.datetime64("2020-01-01")
t1 = np.datetime64("2020-01-08")  # +7 day window; sign(T) > 0 -> repelling LCS
lon_axis = np.linspace(-25.0, -20.0, 6)
lat_axis = np.linspace(15.0, 20.0, 5)
```

## Seed a NeighborSeed

`from_axes` broadcasts the 1-D axes into curvilinear reference positions
`lon_0(i, j)` / `lat_0(i, j)`. No time is recorded -- a seed carries no `t0`,
no `T`, and no data variables.

```python
seed = NeighborSeed.from_axes(lon_axis, lat_axis)
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

`pset_to_flowmap` reattaches the flat positions, records `t0`, and derives the
signed window $T = t_1 - t_0$, returning a `FlowMap`. With Parcels these would
be the advected outputs; here they are the emitted positions (released at the
same `t0`), so the flow map is the identity.

```python
fm = seed.pset_to_flowmap(lon, lat, t0=t0, t1=t1)
fm.ds
```

Round-trip checks: the `(i, j)` grid is recovered, `t0`/`T` are the scalar
release time and signed window, and the advected `lon`/`lat` match the
reference `lon_0`/`lat_0` (identity input).

```python
print("dims:", dict(fm.ds.sizes))
print("t0:", fm.ds["t0"].values)
print("T:", fm.ds["T"].values)
print("lon matches reference:", np.allclose(fm.ds["lon"], seed.ds["lon_0"]))
print("lat matches reference:", np.allclose(fm.ds["lat"], seed.ds["lat_0"]))
```

## Collapse back to a seed

`FlowMap.to_seed` drops the advected positions and the `t0`/`T` coords,
recovering a time-free seed. Re-emitting reproduces the same particle set
(losslessness), and the recovered seed carries neither `t0` nor `T`.

```python
recovered = fm.to_seed()
re_lon, re_lat = recovered.to_parcels_pset()
print("re-emitted pset matches:", np.allclose(re_lon, lon) and np.allclose(re_lat, lat))
print("recovered seed has t0:", "t0" in recovered.ds.coords)
print("recovered seed has T:", "T" in recovered.ds.coords)
```

## AuxiliarySeed: four arms per point

The auxiliary stencil places four arms (`displacement = [east, north, west,
south]`) around each grid point. The arms are the *explicit* reference
positions `lon_0`/`lat_0` on `(i, j, displacement)` -- so the dataset is
self-sufficient -- while the grid-point centres (where the FTLE is reported)
are kept separately as `lon_c`/`lat_c` on `(i, j)`.

```python
aux_seed = AuxiliarySeed.from_axes(lon_axis, lat_axis)
aux_lon, aux_lat = aux_seed.to_parcels_pset()
print(f"{len(aux_lon)} particles = {lon_axis.size} x {lat_axis.size} x 4 arms")
print("reference arms lon_0:", aux_seed.ds["lon_0"].dims)
print("diagnostic centres lon_c:", aux_seed.ds["lon_c"].dims)
```

Ingesting threads the same `t0` and `t1`. The advected `lon` lands on
`(i, j, displacement)` alongside the reference arms, while the centres stay on
`(i, j)`.

```python
aux_fm = aux_seed.pset_to_flowmap(aux_lon, aux_lat, t0=t0, t1=t1)
print("reference arms  lon_0:", aux_fm.ds["lon_0"].dims)
print("advected arms   lon  :", aux_fm.ds["lon"].dims)
print("diagnostic centres lon_c:", aux_fm.ds["lon_c"].dims)
```

`to_seed` rebuilds the auxiliary seed from the carried arms (no need for the
original axes or `aux_separation_m`); re-emitting reproduces the same arm
particle set.

```python
re_aux_lon, re_aux_lat = aux_fm.to_seed().to_parcels_pset()
print(
    "re-emitted arms match:",
    np.allclose(re_aux_lon, aux_lon) and np.allclose(re_aux_lat, aux_lat),
)
```
