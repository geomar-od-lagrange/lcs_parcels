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

# Evolving a hyperbolic LCS as a material curve

An extracted hyperbolic LCS (see `cabo_verde_lcs`) is a **material** curve:
once we have its vertices at $t_0$ — the `lon`/`lat` that `shrink_lines`
returns — its position at any other time is fixed by the flow map,
$\mathcal{M}(t) = F_{t_0}^{t}(\mathcal{M}(t_0))$ (Haller 2015, Eq. 5,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322)). We do
not re-diagnose the LCS at each time; we watch one fixed curve move.

The flow map $F_{t_0}^{t}$ *is* the advected-position field a `FlowMap`
stores, so evolving the curve is interpolating that field at the curve's
vertices with `FlowMap.image` — no re-integration of the curve, and
self-consistent with the map that diagnosed it.

Each family is evolved in its **coherent** direction, the one in which
perturbations shrink. So the attracting curve is carried by the forward maps
and the repelling curve by the backward maps.

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import NeighborSeed
```

## Currents

The local file that `get_data` writes.

```python
currents = xr.open_dataset("data/cabo_verde_currents_hourly.nc").load()
currents
```

## Parcels v4 field set

`copernicusmarine_to_sgrid` tags the CMEMS A-grid with SGRID metadata;
`from_sgrid_conventions` wraps it as a spherical `FieldSet`.

```python
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])
```

## Seed and horizons

A rectilinear `NeighborSeed` over the release box, anchored at the middle of
the window the local file covers so five days fit either side of $t_0$.

We evolve to **daily** horizons out to five days. Daily is the cadence at
which this flow moves a curve visibly while consecutive frames still overlap,
so the sequence reads as one curve deforming rather than as unrelated curves;
the longest horizon is also where we diagnose the LCS.

```python
t0 = np.datetime64("2025-08-06")
horizons = np.arange(1, 6) * np.timedelta64(1, "D")
resolution_deg = 1 / 25
seed_lon, seed_lat = (-27.0, -21.0), (13.5, 18.5)
```

```python
lon_axis = np.arange(seed_lon[0], seed_lon[1] + 1e-9, resolution_deg)
lat_axis = np.arange(seed_lat[0], seed_lat[1] + 1e-9, resolution_deg)
seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
seed.ds
```


## Recovery kernel

Particles that leave the domain or hit land are turned into `NaN` in place
(Parcels would otherwise abort the run), so losses propagate as `NaN` through
every diagnostic below. `StatusCode.EndofLoop` rather than
`StatusCode.Delete`: deleting shrinks the particle array and breaks the
alignment with the seed order.


```python
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)
```

## Advect to each horizon

Each horizon is its own integration from $t_0$, in each direction. Carrying a
single integration through all five horizons and reading the positions off
along the way would be cheaper, but a particle lost partway through then
re-enters the next leg with a stale clock, which perturbs the values its
surviving neighbours are interpolated from.

```python
forward_maps, backward_maps = [], []
for direction, maps in ((1, forward_maps), (-1, backward_maps)):
    for horizon in horizons:
        lon0, lat0 = seed.to_parcels_pset()
        pset = ParticleSet(
            fieldset,
            pclass=Particle,
            x=lon0,
            y=lat0,
            z=np.full(len(lon0), z_surface),
            t=t0,
        )
        pset.execute(
            [AdvectionRK4, set_lost_to_nan],
            dt=direction * np.timedelta64(1, "h"),
            runtime=horizon,
        )
        maps.append(
            seed.pset_to_flowmap(
                lon=pset.x, lat=pset.y, t0=t0, t1=t0 + direction * horizon
            )
        )
```

```python
forward_maps[-1].ds
```

## Extract the LCS at the longest window

The ridges are sharpest at the longest horizon, so we diagnose there:
repelling LCS from the forward flow, attracting from the backward one
(forward–backward duality, Haller & Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)), each seeded at
the local maxima of its own FTLE. `FlowMap.hyperbolic_lcs` runs that chain —
FTLE, ridge seeds, shrink lines — in one call and reads repelling or
attracting off the sign of its own window; *hyperbolic* because elliptic LCS
are a different family.

```python
repelling_lcs = forward_maps[-1].hyperbolic_lcs()
attracting_lcs = backward_maps[-1].hyperbolic_lcs()
attracting_lcs
```

```python
print(
    f"{repelling_lcs.sizes['line']} repelling, {attracting_lcs.sizes['line']} attracting lines"
)
```

## Evolve each family of material lines in its coherent direction

`FlowMap.image` interpolates a flow map at the curve's vertices and returns
`lon`/`lat` on the curve's own `(line, point)` dims, so an image stacks
directly with the curve it came from. Concatenating the curve at $t_0$ with
its image under each horizon map gives an evolution cube on
`(offset, line, point)`, `offset` being the signed offset from $t_0$ in days.

The scalar `T` on each cube points the *opposite* way to its `offset`, which
looks wrong and is not: `T` is inherited provenance, the window the curve was
diagnosed over, while `offset` is the direction it is being advected. An
attracting LCS is diagnosed backward ($T < 0$) and evolved forward.

```python
offset_days = np.concatenate([[0.0], horizons / np.timedelta64(1, "D")])
OFFSET_ATTRS = {"long_name": "signed offset from t0", "units": "days"}
```

```python
# The attracting curve rides the forward maps.
attracting_evo = xr.concat(
    [
        attracting_lcs[["lon", "lat"]],
        *(
            m.image(lon0=attracting_lcs["lon"], lat0=attracting_lcs["lat"])
            for m in forward_maps
        ),
    ],
    dim="offset",
).assign_coords(offset=("offset", offset_days, OFFSET_ATTRS))
attracting_evo
```

```python
# The repelling curve rides the backward maps.
repelling_evo = xr.concat(
    [
        repelling_lcs[["lon", "lat"]],
        *(
            m.image(lon0=repelling_lcs["lon"], lat0=repelling_lcs["lat"])
            for m in backward_maps
        ),
    ],
    dim="offset",
).assign_coords(offset=("offset", -offset_days, OFFSET_ATTRS))
repelling_evo
```

## Snapshots

Each family of material lines day by day, with its own $t_0$ position drawn
faintly in every
panel for reference. The axes are shared and left to autoscale, so a curve
that leaves the release box stays visible.

```python
fig, axes = plt.subplots(2, len(offset_days), figsize=(16, 6), sharex=True, sharey=True)
lcs_families = [
    (attracting_evo, "tab:blue", "attracting"),
    (repelling_evo, "tab:red", "repelling"),
]
for row, (evo, color, name) in zip(axes, lcs_families, strict=True):
    for k, ax in enumerate(row):
        ax.plot(
            evo["lon"].isel(offset=0).T,
            evo["lat"].isel(offset=0).T,
            color="0.7",
            lw=0.6,
        )
        ax.plot(
            evo["lon"].isel(offset=k).T,
            evo["lat"].isel(offset=k).T,
            color=color,
            lw=0.8,
        )
        ax.set_title(f"{name}, offset {evo['offset'].isel(offset=k).item():+.0f} d")
```
