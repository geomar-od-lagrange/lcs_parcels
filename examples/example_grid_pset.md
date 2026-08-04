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

# Seed, flow map, and the two stencils

A tour of the structures the package is built from, run on a closed-form flow
map instead of a real advection: no currents, no download, and no Parcels
import anywhere below. The particle set still crosses the Parcels boundary in
both directions -- `Seed.to_parcels_pset()` emits flat `(lon, lat)`,
`Seed.pset_to_flowmap(...)` ingests the positions that come back -- so what
stands in for Parcels here is one line of arithmetic.

The same grid is seeded twice, once per stencil, and carried side by side to
the FTLE. On this smooth analytic flow the two agree on the *value* to roughly
machine precision; what differs is where the value exists at all, since the
neighbour stencil loses the outer ring of grid points and the auxiliary one
does not.

```python
import numpy as np

from lcs_parcels import AuxiliarySeed, NeighborSeed
```

## Grid and window

A seed is time-free: the axes fix *where*, and nothing fixes *when* until
ingest. `t1 > t0` makes the window $T = t_1 - t_0$ positive, so the FTLE below
is the forward, repelling one.

```python
lon_axis = np.linspace(-25.0, -20.0, 6)
lat_axis = np.linspace(15.0, 20.0, 5)

t0 = np.datetime64("2020-01-01")
t1 = np.datetime64("2020-01-08")
```

## The neighbour stencil

One particle per grid point: the release positions `lon_0`/`lat_0` *are* the
diagnostic grid points `lon_grid`/`lat_grid`, stored twice rather than left to
be reconstructed. $\nabla F$ is then differenced against the neighbouring grid
points, which ties the diagnostic scale to the seed spacing and leaves the
outermost ring of points without a neighbour to difference against.

```python
seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
print(seed)
seed.ds
```

## The auxiliary stencil

Four arms -- `east, north, west, south` -- at `aux_separation_m` around each
grid point, so `lon_0`/`lat_0` gain a `displacement` dim and there are four
particles per grid point. $\nabla F$ is differenced across the arms, so the
finite-difference step is the arm separation rather than the grid spacing, and
it is defined at every grid point including the boundary. 1 km is small against
the scale on which a mesoscale flow deforms, and large against the positional
error of an RK4 advection.

```python
aux_seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis, aux_separation_m=1_000.0)
print(aux_seed)
aux_seed.ds
```

## Emit the particle sets

`to_parcels_pset()` flattens the release positions to two plain lists, one
entry per particle: 30 grid points for the neighbour stencil, 4 arms each for
the auxiliary one. This is everything Parcels needs, and the order is what
ingest reattaches by.

```python
lon_0, lat_0 = seed.to_parcels_pset()
aux_lon_0, aux_lat_0 = aux_seed.to_parcels_pset()
print("neighbour particles:", len(lon_0))
print("auxiliary particles:", len(aux_lon_0))
```

## Advect, without Parcels

In place of an advection, a flow map with a known answer: a uniform strain that
stretches along longitude and squeezes along latitude about the grid centre,
by $e^{aT}$ over the window. That centre is where the package anchors its
equirectangular metres frame (the centroid of `lon_0`/`lat_0`), so the map is
an exactly linear strain in the frame $\nabla F$ is differenced in, and the
FTLE it implies is the strain rate $a$ at every grid point -- for either
stencil, whatever the spacing. Anything else the notebook prints below is
discretisation error, of which a linear map has none.

```python
strain_rate = 1e-6  # 1/s, about 0.086/day
window_s = (t1 - t0) / np.timedelta64(1, "s")
stretch = np.exp(strain_rate * window_s)

lon_ref, lat_ref = lon_axis.mean(), lat_axis.mean()


def strain(lon, lat):
    return (
        lon_ref + (np.asarray(lon) - lon_ref) * stretch,
        lat_ref + (np.asarray(lat) - lat_ref) / stretch,
    )
```

```python
lon_1, lat_1 = strain(lon_0, lat_0)
aux_lon_1, aux_lat_1 = strain(aux_lon_0, aux_lat_0)
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

`ftle()` returns 1/s on the diagnostic grid. Compare each field against the
strain rate the closed-form map was built with.

```python
ftle = flowmap.ftle()
ftle
```

```python
aux_ftle = aux_flowmap.ftle()
aux_ftle
```

```python
print("neighbour: max |ftle - a| =", float(abs(ftle - strain_rate).max()))
print("auxiliary: max |ftle - a| =", float(abs(aux_ftle - strain_rate).max()))
print("neighbour NaN points:", int(ftle.isnull().sum()))
print("auxiliary NaN points:", int(aux_ftle.isnull().sum()))
```

## One lost particle

A particle that beaches or leaves the domain comes back as NaN -- that is the
contract the recovery kernel of a real Parcels run has to honour. Nothing in
the package handles it specially; the NaN simply propagates through the
differences into the FTLE. How far it spreads is a property of the stencil, so
lose one particle at the interior grid point `(i=2, j=2)` in each set -- for
the auxiliary stencil, one of that point's four arms -- and count.

```python
# Find that particle in the flat order, by stacking the release positions the
# way `to_parcels_pset` does.
particles = seed.ds["lon_0"].reset_coords(drop=True).stack(particle=("i", "j"))
lost = int(np.flatnonzero((particles["i"] == 2) & (particles["j"] == 2))[0])

aux_particles = (
    aux_seed.ds["lon_0"]
    .reset_coords(drop=True)
    .stack(particle=("i", "j", "displacement"))
)
aux_lost = int(
    np.flatnonzero(
        (aux_particles["i"] == 2)
        & (aux_particles["j"] == 2)
        & (aux_particles["displacement"] == "east")
    )[0]
)
```

```python
lon_lost, lat_lost = np.asarray(lon_1).copy(), np.asarray(lat_1).copy()
lon_lost[lost] = lat_lost[lost] = np.nan

aux_lon_lost, aux_lat_lost = np.asarray(aux_lon_1).copy(), np.asarray(aux_lat_1).copy()
aux_lon_lost[aux_lost] = aux_lat_lost[aux_lost] = np.nan
```

```python
ftle_lost = seed.pset_to_flowmap(lon=lon_lost, lat=lat_lost, t0=t0, t1=t1).ftle()
aux_ftle_lost = aux_seed.pset_to_flowmap(
    lon=aux_lon_lost, lat=aux_lat_lost, t0=t0, t1=t1
).ftle()

print("neighbour NaN points:", int(ftle_lost.isnull().sum()))
print("auxiliary NaN points:", int(aux_ftle_lost.isnull().sum()))
```

```python
ftle_lost.isnull()
```
