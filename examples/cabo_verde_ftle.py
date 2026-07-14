# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
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
# # Forward FTLE over Cabo Verde from CMEMS currents
#
# The minimal wiring between `lcs_parcels` and Parcels v4: seed a grid, advect it
# through CMEMS surface currents, ingest the final positions, map the forward
# FTLE. One stencil (`NeighborSeed`); nothing tuned for speed.
#
# Needs the `examples` pixi environment and CMEMS credentials: run with
# `pixi run -e examples jupyter ...`.

# %%
import numpy as np

import copernicusmarine as cm
from parcels import FieldSet, ParticleSet, Particle, StatusCode
from parcels.kernels import AdvectionRK4
from parcels.convert import copernicusmarine_to_sgrid

from lcs_parcels import NeighborSeed

# %% [markdown]
# ## Parameters

# %%
t0 = np.datetime64("2025-08-01")
T = np.timedelta64(10, "D")            # signed window; sign(T) sets the direction
t1 = t0 + T

resolution_deg = 1 / 25                # seed-grid spacing
seed_lon, seed_lat = (-27.0, -21.0), (13.5, 18.5)   # release box
data_lon, data_lat = (-30.5, -17.5), (10.0, 22.0)   # current field = seed box + margin

# %% [markdown]
# ## Currents: CMEMS hourly surface velocity

# %%
ds = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
    variables=["uo", "vo"],
    minimum_longitude=data_lon[0], maximum_longitude=data_lon[1],
    minimum_latitude=data_lat[0], maximum_latitude=data_lat[1],
    minimum_depth=0.0, maximum_depth=1.0,
    start_datetime=str((t0 - np.timedelta64(1, "D")).astype("datetime64[D]")),
    end_datetime=str((t1 + np.timedelta64(1, "D")).astype("datetime64[D]")),
).load()
ds

# %% [markdown]
# ## Parcels v4 field set
#
# `copernicusmarine_to_sgrid` tags the CMEMS A-grid with SGRID metadata;
# `from_sgrid_conventions` wraps it as a spherical `FieldSet`.

# %%
sgrid = copernicusmarine_to_sgrid(fields={"U": ds["uo"], "V": ds["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(ds["depth"].values[0])

# %% [markdown]
# ## Recovery kernel
#
# Particles that leave the domain or hit land are turned into `NaN` in place
# (Parcels would otherwise abort the run), so losses propagate as `NaN` through
# the FTLE.

# %%
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)


# %% [markdown]
# ## Seed, advect, FTLE
#
# A rectilinear `NeighborSeed` over the seed box (one particle per grid point,
# gradient differenced against grid neighbours) emits a flat particle set; we run
# RK4 forward for $T$ and ingest the finals back into a `FlowMap`.
# `FlowMap.ftle()` returns $1/\mathrm{s}$; we report $1/\mathrm{day}$.

# %%
# Create the Seed
lon_axis = np.arange(seed_lon[0], seed_lon[1] + 1e-9, resolution_deg)
lat_axis = np.arange(seed_lat[0], seed_lat[1] + 1e-9, resolution_deg)
seed = NeighborSeed.from_axes(lon_axis, lat_axis)

# %%
# Advect in Parcels
lon, lat = seed.to_parcels_pset()
lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
pset = ParticleSet(
    fieldset, pclass=Particle,
    x=lon, y=lat, z=np.full(lon.size, z_surface), t=np.full(lon.size, t0),
)
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"), runtime=T, verbose_progress=False,
)
fin_lon, fin_lat = np.asarray(pset.x, dtype=float), np.asarray(pset.y, dtype=float)

# %%
# Construct flowmap / calc FTLE
flowmap = seed.pset_to_flowmap(fin_lon, fin_lat, t0=t0, t1=t1)
ftle = (flowmap.ftle() * 86400.0).rename("FTLE")

# %% [markdown]
# ## Map

# %%
ftle.plot.pcolormesh(x="lon_0", y="lat_0")
