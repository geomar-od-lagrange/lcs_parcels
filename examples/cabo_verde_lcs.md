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

# Cabo Verde hyperbolic LCS

The FTLE map (see `cabo_verde_ftle`) shows where the flow stretches. It does
not give the material curves along which the stretching happens. Those curves
are tensor lines of the strain tensor. Haller (2015, §5.1 / Table 1,
[doi:10.1146/annurev-fluid-010313-141322](https://doi.org/10.1146/annurev-fluid-010313-141322))
constructs those curves directly from the Cauchy–Green strain tensor
$C = (\nabla F)^\top \nabla F$, whose eigenpairs satisfy
$C\,\xi_i = \lambda_i\,\xi_i$ with $0 < \lambda_1 \le \lambda_2$ and
$\xi_1 \perp \xi_2$. A **repelling** LCS is a *shrink line* — a curve tangent
to $\xi_1$, i.e. orthogonal to the strong-stretch direction $\xi_2$ that the
FTLE ridge marks. It solves the ODE $\dot r = \xi_1(r)$.

Attracting LCS come from the forward–backward duality (Haller & Sapsis 2011,
[doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)): an **attracting**
LCS is a repelling LCS of the *backward* flow. The notebook therefore advects
the same seed grid both ways and takes the $\xi_1$ shrink lines of each flow
map: the forward map gives the repelling LCS, the backward map the attracting
ones.

```python
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import NeighborSeedGrid, ftle_ridge_seeds, shrink_lines
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

## Seed grid and window

$t_0$ sits at the middle of the window the local file covers, and the runs are
$\pm 5$ d, so the forward and the backward advection both stay inside the
data. A rectilinear `NeighborSeedGrid` puts one particle on every diagnostic grid
point; $\nabla F$ is then differenced against the grid neighbours.

```python
t0 = np.datetime64("2025-08-06")
T = np.timedelta64(5, "D")
resolution_deg = 1 / 25
seed_lon, seed_lat = (-27.0, -21.0), (13.5, 18.5)

lon_axis = np.arange(seed_lon[0], seed_lon[1] + 1e-9, resolution_deg)
lat_axis = np.arange(seed_lat[0], seed_lat[1] + 1e-9, resolution_deg)
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
seed
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

## Advect forward

`to_parcels_pset` emits the flat `(lon, lat)` pair; the finals go back in via
`pset_to_flowmap`, which takes both times and derives the signed window
$T = t_1 - t_0$.

```python
lon, lat = seed.to_parcels_pset()
z = np.full(len(lon), z_surface)
pset = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=False,
)
forward = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t0 + T)
forward
```

## Advect backward

The same seed grid, same $t_0$, negative `dt`, and $t_1 = t_0 - T$ so the
stored window is negative.

```python
lon, lat = seed.to_parcels_pset()
z = np.full(len(lon), z_surface)
pset = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=-np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=False,
)
backward = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t0 - T)
backward
```

## Repelling LCS, step by step

The forward FTLE (from $\lambda_2$ of the forward $C$) is the backdrop and its
ridges are where repelling LCS live.

```python
ftle_forward = forward.ftle()
ftle_forward
```

`ftle_ridge_seeds` picks seed points at the ridge tops: grid points that are
the maximum over a square window centred on them and lie in the top `quantile`
of the field. `window_m` is the full side of that window, so two seeds can be
about half of it apart. The target here is neighbouring filaments about
15 km apart in this eddy field, so the window is 30 km. The top decile keeps
the seeds on the pronounced ridges of *this* field. The alternative selector,
`ftle_min`, is an absolute floor in 1/s; use it when several regions or
windows have to be compared against one threshold.

```python
# Ridge selection.
window_m, quantile = 30_000.0, 0.90
```

```python
repelling_seeds = ftle_ridge_seeds(ftle_forward, window_m=window_m, quantile=quantile)
repelling_seeds
```

The seed dataset records how the window came out on this grid. The distance
below is the closest two seeds can be: half the window, not the window.

```python
repelling_seeds.attrs["min_seed_separation_m"]
```

`shrink_lines` integrates $\dot r = \xi_1(r)$ through each seed, half the
length either way. The 3 km step is below the seed-grid cell of about 4.4 km,
so the RK2 trace samples every cell it crosses. The 1500 km cap is longer than
the box diagonal, so a curve ends by leaving the grid, by hitting a NaN cell,
or by dropping below `min_anisotropy`, the floor on $\lambda_2 / \lambda_1$
below which $\xi_1$ is no longer a well-defined direction. That floor is a
ratio, so it does not depend on the window or on the stretching rate of the
flow, and the default of 1.15 also applies to a longer window or a faster
flow.

```python
# Line integration.
step_m, line_length_m, min_anisotropy = 3_000.0, 1_500_000.0, 1.15
```

```python
repelling_lcs = shrink_lines(
    forward,
    seed_lon=repelling_seeds["lon"],
    seed_lat=repelling_seeds["lat"],
    step_m=step_m,
    line_length_m=line_length_m,
    min_anisotropy=min_anisotropy,
)
repelling_lcs
```

The lines come back as a fixed `(line, point)` rectangle, NaN-padded past
termination: every line has the same number of columns, and the curves have
different lengths.

```python
repelling_lcs["lon"].isnull().sum("point")
```

Rows that are NaN in every column traced no curve. About a quarter of the
seeds here sit close to land, so the interpolated tensor is NaN at the seed
and no curve is produced.

```python
untraceable = repelling_lcs["lon"].isnull().all("point")
int(untraceable.sum()), repelling_lcs.sizes["line"]
```

## Attracting LCS, in one call

The backward flow map runs the same three steps, and `hyperbolic_lcs` packs
them into one call: it computes the FTLE once, hands it to the ridge finder,
and returns the curves together with the field the seeds were picked from.
*Hyperbolic* because elliptic LCS are a different family.

```python
attracting_lcs = backward.hyperbolic_lcs(
    window_m=window_m,
    quantile=quantile,
    step_m=step_m,
    line_length_m=line_length_m,
    min_anisotropy=min_anisotropy,
)
attracting_lcs
```

## Repelling LCS over the forward FTLE

The seed points are dotted, which shows whether they sit on the ridge tops and
stay separated. The FTLE gets a greyscale so the coloured curves stand out
against it. The dots with no curve through them are the untraceable seeds
counted above.

```python
fig, ax = plt.subplots()
ftle_forward.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(
    repelling_lcs["lon"].values.T,
    repelling_lcs["lat"].values.T,
    color="tab:red",
    lw=0.8,
)
ax.scatter(repelling_seeds["lon"], repelling_seeds["lat"], s=8, color="tab:red")
plt.show()
```
## Attracting LCS over the backward FTLE

Same picture for the backward flow map, straight out of the one dataset
`hyperbolic_lcs` returned.

```python
fig, ax = plt.subplots()
attracting_lcs["ftle"].plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(
    attracting_lcs["lon"].values.T,
    attracting_lcs["lat"].values.T,
    color="tab:blue",
    lw=0.8,
)
plt.show()
```
## Both families together

Both families over the forward FTLE in one frame. Where a repelling and an
attracting curve cross, the flow is locally hyperbolic.

```python
fig, ax = plt.subplots()
ftle_forward.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(
    repelling_lcs["lon"].values.T,
    repelling_lcs["lat"].values.T,
    color="tab:red",
    lw=0.8,
)
ax.plot(
    attracting_lcs["lon"].values.T,
    attracting_lcs["lat"].values.T,
    color="tab:blue",
    lw=0.8,
)
plt.show()
```
