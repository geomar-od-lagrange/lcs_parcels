# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,md,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Repelling and attracting LCS as strain tensor lines
#
# The FTLE map (see `cabo_verde_ftle`) shows *where* the flow stretches, but not
# the material curves themselves. Haller (2015, §5.1 / Table 1) constructs those
# curves directly from the Cauchy–Green strain tensor
# $C = (\nabla F)^\top \nabla F$, whose eigenpairs satisfy
# $C\,\xi_i = \lambda_i\,\xi_i$ with $0 < \lambda_1 \le \lambda_2$ and
# $\xi_1 \perp \xi_2$. A **repelling** LCS is a *shrink line* — a curve tangent
# to $\xi_1$, i.e. orthogonal to the strong-stretch direction $\xi_2$ that the
# FTLE ridge marks. It solves the ODE $\dot r = \xi_1(r)$.
#
# Attracting LCS come from the forward–backward duality (Haller & Sapsis 2011,
# [doi:10.1063/1.3579597](https://doi.org/10.1063/1.3579597)): an **attracting**
# LCS is a repelling LCS of the *backward* flow. So we advect the same seed grid
# both ways and take the $\xi_1$ shrink lines of each flow map — forward for
# repelling, backward for attracting.

# %%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import NeighborSeed, ftle_ridge_seeds, shrink_lines

# %% [markdown]
# ## Currents
#
# Hourly CMEMS surface velocity from the local file that `get_data` writes.

# %%
currents = xr.open_dataset("data/cabo_verde_currents_hourly.nc")
currents

# %% [markdown]
# `copernicusmarine_to_sgrid` tags the CMEMS A-grid with SGRID metadata;
# `from_sgrid_conventions` wraps it as a spherical `FieldSet`.

# %%
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])

# %% [markdown]
# ## Seed grid and window
#
# We anchor at the middle of the window the local file covers and run $\pm 5$ d,
# so the forward and the backward advection both stay inside the data. A
# rectilinear `NeighborSeed` puts one particle on every diagnostic grid point;
# $\nabla F$ is then differenced against the grid neighbours.

# %%
t0 = np.datetime64("2025-08-06")
T = np.timedelta64(5, "D")
resolution_deg = 1 / 25
seed_lon, seed_lat = (-27.0, -21.0), (13.5, 18.5)

lon_axis = np.arange(seed_lon[0], seed_lon[1] + 1e-9, resolution_deg)
lat_axis = np.arange(seed_lat[0], seed_lat[1] + 1e-9, resolution_deg)
seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
seed


# %% [markdown]
# ## Recovery kernel
#
# Particles that leave the domain or hit land are turned into `NaN` in place
# (Parcels would otherwise abort the run), so losses propagate as `NaN` through
# every diagnostic below.


# %%
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)


# %% [markdown]
# ## Advect forward
#
# `to_parcels_pset` emits the flat `(lon, lat)` pair; the finals go back in via
# `pset_to_flowmap`, which takes both times and derives the signed window
# $T = t_1 - t_0$.

# %%
lon, lat = seed.to_parcels_pset()
z = np.full(len(lon), z_surface)
pset = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=True,
)
forward = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t0 + T)
forward

# %% [markdown]
# ## Advect backward
#
# The same seed grid, same $t_0$, negative `dt`, and $t_1 = t_0 - T$ so the
# stored window is negative.

# %%
lon, lat = seed.to_parcels_pset()
z = np.full(len(lon), z_surface)
pset = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=-np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=True,
)
backward = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t0 - T)
backward

# %% [markdown]
# ## Repelling LCS, step by step
#
# The forward FTLE (from $\lambda_2$ of the forward $C$) is the backdrop and its
# ridges are where repelling LCS live.

# %%
ftle_forward = forward.ftle()
ftle_forward

# %% [markdown]
# `ftle_ridge_seeds` picks start points at the ridge tops: grid points that are
# the maximum over a `window_m`-wide neighbourhood and lie in the top `quantile`
# of the field. 30 km is the separation we want between neighbouring filaments in
# this eddy field — the window reaches half its side either way, so two seeds can
# end up about 15 km apart, still three to four cells of the 1/25-degree grid.
# The top decile keeps the seeds on the pronounced ridges of *this* field.

# %%
window_m, quantile = 30_000.0, 0.90
repelling_seed_lon, repelling_seed_lat = ftle_ridge_seeds(
    ftle_forward, window_m=window_m, quantile=quantile
)
repelling_seed_lon.shape, repelling_seed_lon[:5], repelling_seed_lat[:5]

# %% [markdown]
# `shrink_lines` integrates $\dot r = \xi_1(r)$ through each seed, half the
# length either way. The 3 km step is below the ~4 km grid cell, so the RK2 trace
# samples every cell it crosses; the 1500 km cap is longer than the box diagonal,
# so what actually ends a curve is leaving the grid, hitting a NaN cell, or
# `min_anisotropy` — the floor on $\lambda_2 / \lambda_1$ below which $\xi_1$ is
# no longer a well-defined direction. Being a ratio, it is free of the window and
# of the flow's own stretching rate, so the 1.15 default carries over unchanged
# from a longer window or a faster flow.

# %%
step_m, line_length_m, min_anisotropy = 3_000.0, 1_500_000.0, 1.15
repelling = shrink_lines(
    forward,
    seed_lon=repelling_seed_lon,
    seed_lat=repelling_seed_lat,
    step_m=step_m,
    line_length_m=line_length_m,
    min_anisotropy=min_anisotropy,
)
repelling

# %% [markdown]
# Every line is one row of a fixed `(line, point)` rectangle, NaN-padded past the
# point where it terminated — so the rows are all the same length and the curves
# are not.

# %%
repelling["lon"].isnull().sum("point")

# %% [markdown]
# ## Attracting LCS, in one call
#
# The backward flow map runs the identical three steps, so we let
# `hyperbolic_lcs` do it: it computes the FTLE once, hands it to the ridge
# finder, and returns the curves together with the field they were picked from.
# *Hyperbolic* because elliptic LCS are a different family.

# %%
attracting = backward.hyperbolic_lcs(
    window_m=window_m,
    quantile=quantile,
    step_m=step_m,
    line_length_m=line_length_m,
    min_anisotropy=min_anisotropy,
)
attracting

# %% [markdown]
# ## Repelling LCS over the forward FTLE
#
# The seed points are dotted so we can check they sit on the ridge tops and stay
# separated. The FTLE gets a greyscale so the curves read against it.

# %%
fig, ax = plt.subplots()
ftle_forward.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(repelling["lon"].values.T, repelling["lat"].values.T, color="tab:red", lw=0.8)
ax.scatter(repelling_seed_lon, repelling_seed_lat, s=8, color="tab:red")

# %% [markdown]
# ## Attracting LCS over the backward FTLE
#
# Same picture for the backward flow map, straight out of the one dataset
# `hyperbolic_lcs` returned.

# %%
fig, ax = plt.subplots()
attracting["ftle"].plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(
    attracting["lon"].values.T, attracting["lat"].values.T, color="tab:blue", lw=0.8
)

# %% [markdown]
# ## Both families together
#
# Both families over the forward FTLE, in one frame: where a repelling and an
# attracting curve cross, the flow is locally hyperbolic.

# %%
fig, ax = plt.subplots()
ftle_forward.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(repelling["lon"].values.T, repelling["lat"].values.T, color="tab:red", lw=0.8)
ax.plot(
    attracting["lon"].values.T, attracting["lat"].values.T, color="tab:blue", lw=0.8
)
