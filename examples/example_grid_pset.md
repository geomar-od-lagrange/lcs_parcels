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

# Seed grids and flow maps

This notebook is about the structures and the API, not about advection.

A `SeedGrid` holds a grid of release positions. `SeedGrid.to_parcels_pset()`
emits them as flat `lon`, `lat` lists. `SeedGrid.pset_to_flowmap(...)` ingests the
advected positions and returns a `FlowMap`, which computes the deformation
gradient, the Cauchy–Green tensor and the FTLE. No Parcels is imported below;
the advection is replaced by particles that do not move.

There are two stencils, and each is a pair of classes. `NeighborSeedGrid` releases
one particle per grid point and differences against the neighbouring grid
points. `AuxiliarySeedGrid` releases four arms around each grid point and
differences across the arms.

```python
import numpy as np

from lcs_parcels import AuxiliarySeedGrid, NeighborSeedGrid
```

## Grid and window

The axes set the grid. The times are not attached to anything until ingest.
`t1 > t0` makes the window $T = t_1 - t_0$ positive, so the FTLE below is the
forward one.

```python
lon_axis = np.linspace(-25.0, -20.0, 6)
lat_axis = np.linspace(15.0, 20.0, 5)

t0 = np.datetime64("2020-01-01")
t1 = np.datetime64("2020-01-11")
```

## The neighbour stencil

One particle per grid point: the release positions `lon_0`/`lat_0` *are* the
diagnostic grid points `lon_grid`/`lat_grid`, stored twice rather than left to
be reconstructed. The outermost ring of grid points has no neighbour to
difference against on one side, so the FTLE is undefined there.

```python
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
print(seed)
seed.ds
```

## The auxiliary stencil

Four arms (`east, north, west, south`) at `aux_separation_m` around each
grid point, so `lon_0`/`lat_0` gain a `displacement` dim and there are four
particles per grid point. The finite-difference step is the arm separation
rather than the grid spacing, and the FTLE is defined at every grid point,
including the boundary.

```python
aux_seed = AuxiliarySeedGrid.from_axes(
    lon=lon_axis, lat=lat_axis, aux_separation_m=1_000.0
)
print(aux_seed)
aux_seed.ds
```

## Emit the particle sets

`to_parcels_pset()` flattens the release positions to two plain lists, one
entry per particle: 30 grid points for the neighbour stencil, 4 arms each for
the auxiliary one. The two lists are everything Parcels needs, and ingest
reattaches the advected positions to the grid by their order.

```python
lon_0, lat_0 = seed.to_parcels_pset()
aux_lon_0, aux_lat_0 = aux_seed.to_parcels_pset()
print("neighbour particles:", len(lon_0))
print("auxiliary particles:", len(aux_lon_0))
```

## Advect, without Parcels

The particles sit still: $u = v = 0$ for ten days. The flow map is the
identity, $\nabla F = I$, so the FTLE is zero wherever it is defined. A
non-zero window keeps $1/T$ finite.

```python
lon_1, lat_1 = lon_0, lat_0
aux_lon_1, aux_lat_1 = aux_lon_0, aux_lat_0
```

## Ingest the advected positions

`pset_to_flowmap` puts the flat positions back on the grid as `lon`/`lat`
beside the release positions `lon_0`/`lat_0`, and records `t0` and the signed
window `T` it derives from `t0` and `t1`. Both stencils ingest through the same
call; the datasets differ only where their release positions did.

```python
flowmap = seed.pset_to_flowmap(lon=lon_1, lat=lat_1, t0=t0, t1=t1)
print(flowmap)
flowmap.ds
```

```python
aux_flowmap = aux_seed.pset_to_flowmap(lon=aux_lon_1, lat=aux_lat_1, t0=t0, t1=t1)
print(aux_flowmap)
aux_flowmap.ds
```

`grid_image` is the one place the two datasets are reconciled: the advected
position *of the grid point*, on `(i, j)`. The neighbour flow map passes its
single advected position through; the auxiliary one takes the centroid of its
four advected arms.

```python
aux_flowmap.grid_image
```

## FTLE from both stencils

`ftle()` returns 1/s on the diagnostic grid. Both stencils give zero; they
differ in where the FTLE is defined.

```python
ftle = flowmap.ftle()
ftle
```

```python
aux_ftle = aux_flowmap.ftle()
aux_ftle
```

```python
print("neighbour: max |ftle| =", float(abs(ftle).max()))
print("auxiliary: max |ftle| =", float(abs(aux_ftle).max()))
print("neighbour NaN points:", int(ftle.isnull().sum()))
print("auxiliary NaN points:", int(aux_ftle.isnull().sum()))
```
