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
# # Cabo Verde FTLE
#
# The minimal wiring between `lcs_parcels` and Parcels v4: seed a grid, advect it
# through CMEMS surface currents, ingest the final positions, map the forward
# FTLE. One stencil (`NeighborSeedGrid`); nothing tuned for speed.
#
# The currents come from the local file `data/cabo_verde_currents_hourly.nc`;
# run `get_data.ipynb` once to produce it.

# %% tags=["remove-output"]
# Importing Parcels pulls in the holoviews/bokeh bootstrap and prints an
# alpha-version notice, neither of which belongs on the rendered page.
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import NeighborSeedGrid

# %% [markdown]
# ## Parameters
#
# Ten days is roughly one eddy turnover time at these latitudes: the FTLE map
# below comes out around 0.1/day over most of the box, an e-folding time of about
# ten days. That is long enough for the stretching to separate neighbouring
# particles by more than one seed-grid cell ($1/25^\circ$, about 4.4 km), and
# short enough that the seed grid still resolves where they went. The seed
# spacing is finer than the $1/12^\circ$ currents, so the differencing stencil is
# not what limits the FTLE.

# %%
t0 = np.datetime64("2025-08-01")
T = np.timedelta64(10, "D")  # signed window; sign(T) sets the direction
t1 = t0 + T

resolution_deg = 1 / 25  # seed-grid spacing
seed_lon, seed_lat = (-27.0, -21.0), (13.5, 18.5)  # release box

# %% [markdown]
# ## Currents
#
# The local file that `get_data` writes.

# %%
currents = xr.open_dataset("data/cabo_verde_currents_hourly.nc").load()
currents

# %% [markdown]
# ## Parcels v4 field set
#
# `copernicusmarine_to_sgrid` tags the CMEMS A-grid with SGRID metadata;
# `from_sgrid_conventions` wraps it as a spherical `FieldSet`.

# %%
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])


# %% [markdown]
# ## Recovery kernel
#
# Particles that leave the domain or hit land are turned into `NaN` in place
# (Parcels would otherwise abort the run), so losses propagate as `NaN` through
# the FTLE. `StatusCode.EndofLoop` rather than `StatusCode.Delete`: deleting
# shrinks the particle array and breaks the alignment with the seed order.


# %%
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)


# %% [markdown]
# ## Seed grid
#
# `NeighborSeedGrid` releases one particle per diagnostic grid point and differences
# the flow-map gradient against the grid neighbours, so the stencil costs nothing
# beyond the grid itself. The seed grid carries the release positions
# `lon_0`/`lat_0` and the diagnostic grid `lon_grid`/`lat_grid`, and no time.

# %%
lon_axis = np.arange(seed_lon[0], seed_lon[1] + 1e-9, resolution_deg)
lat_axis = np.arange(seed_lat[0], seed_lat[1] + 1e-9, resolution_deg)
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
seed.ds

# %% [markdown]
# ## Advect
#
# `to_parcels_pset()` flattens the release positions to `(lon, lat)`; Parcels
# calls those `x`/`y`.

# %%
lon, lat = seed.to_parcels_pset()
z = np.full(len(lon), z_surface)
pset = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)

# %%
pset.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=False,
)

# %% [markdown]
# ## Flow map
#
# The final positions go back in the seeding order. `pset_to_flowmap` takes both `t0` and
# `t1` and records the signed window `T = t1 - t0`.

# %%
flowmap = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t1)
flowmap.ds

# %% [markdown]
# ## FTLE

# %%
ftle = flowmap.ftle()
ftle

# %% [markdown]
# ## Map
#
# `ftle()` returns SI $1/\mathrm{s}$, which over a 10-day window is a field of
# numbers around $10^{-6}$; rescaling to $1/\mathrm{day}$ makes the map
# readable. The scaling carries the name and `long_name` over, so only `units`
# has to be corrected. The grid points are 2-D coords on the logical dims
# `(i, j)`, so `x` and `y` name the coords to use as axes; the labels come from
# the metadata.

# %%
ftle_per_day = (ftle * 86400.0).assign_attrs(units="1/day")

# %%
_ = ftle_per_day.plot.pcolormesh(x="lon_grid", y="lat_grid")
