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
# # Cape Cauldron coherent vortices
#
# A coherent Lagrangian vortex boundary is a *closed shear line*, a closed curve
# tangent to $\eta^\pm_\lambda$ and so stretched uniformly by $\lambda$ over the
# window (Haller & Beron-Vera 2013,
# [doi:10.1017/jfm.2013.391](https://doi.org/10.1017/jfm.2013.391)). This
# notebook finds them in the Cape Cauldron, the Agulhas-ring corridor south-west
# of South Africa, step by step with the elliptic API. The construction and
# every parameter are described in the
# [API guide](https://lcs-parcels.readthedocs.io/page/api.html), and
# `cabo_verde_ftle` explains the Parcels wiring.

# %% tags=["remove-output"]
# Importing Parcels pulls in the holoviews/bokeh bootstrap and prints an
# alpha-version notice, neither of which belongs on the rendered page.
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import (
    NeighborSeedGrid,
    closed_shear_lines,
    elliptic_centres,
    outermost_shear_lines,
    stretch_range,
)

# %% [markdown]
# ## Currents
#
# Daily-mean GLORYS12 reanalysis surface currents at 1/12 degree, the local file
# that `get_data` writes.

# %%
currents = xr.open_dataset("data/cape_cauldron_glorys_daily.nc").load()
currents

# %% [markdown]
# ## Parcels v4 field set

# %%
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])

# %% [markdown]
# ## Seed grid and window
#
# A box across the Cape Cauldron at 1/25 degree, released on 2018-06-01 and read
# at 15 days.

# %%
t0 = np.datetime64("2018-06-01")
T = np.timedelta64(15, "D")
lon_axis = np.arange(2.0, 22.0 + 1e-9, 1 / 25)
lat_axis = np.arange(-44.0, -28.0 + 1e-9, 1 / 25)
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
seed

# %% [markdown]
# ## Recovery kernel
#
# Particles that leave the domain or hit land are turned into `NaN` in place
# (Parcels would otherwise abort the run), so losses propagate as `NaN`
# through every diagnostic below. The kernel sets `StatusCode.EndofLoop`
# rather than `StatusCode.Delete`, because deleting shrinks the particle
# array and breaks the alignment with the seed order.


# %%
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)


# %% [markdown]
# ## Advect forward

# %%
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

# %% [markdown]
# A second run over the first day gives the rotation sense. The polar rotation
# angle is defined modulo $2\pi$, so it has to come from a window shorter than
# half a turn.

# %%
pset_day = ParticleSet(fieldset, pclass=Particle, x=lon, y=lat, z=z, t=t0)
pset_day.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=np.timedelta64(1, "D"),
    verbose_progress=False,
)
first_day = seed.pset_to_flowmap(
    lon=pset_day.x, lat=pset_day.y, t0=t0, t1=t0 + np.timedelta64(1, "D")
)
first_day

# %% [markdown]
# ## Candidate centres
#
# `elliptic_centres` takes any field. Here it is the Cauchy–Green eigenvalue
# ratio $\lambda_2 / \lambda_1$, whose windowed minima mark the vortex cores.

# %%
eigen = forward.cg_eigen()
anisotropy = (
    (eigen["lambda"].isel(eig=1, drop=True) / eigen["lambda"].isel(eig=0, drop=True))
    .rename("cg_anisotropy")
    .assign_attrs(
        long_name="Cauchy-Green eigenvalue ratio lambda_2 / lambda_1", units="1"
    )
)
# The ratio spans many decades, so it is drawn as its decimal logarithm.
log_anisotropy = (
    np.log10(anisotropy)
    .rename("log10_cg_anisotropy")
    .assign_attrs(long_name="log10 of the Cauchy-Green eigenvalue ratio", units="1")
)
log_anisotropy.plot.pcolormesh(x="lon_grid", y="lat_grid", robust=True)
plt.show()

# %%
centres = elliptic_centres(anisotropy, extremum="min", window_m=100_000.0)
centres

# %%
_, ax = plt.subplots()
log_anisotropy.plot.pcolormesh(
    x="lon_grid", y="lat_grid", ax=ax, cmap="Greys", robust=True
)
ax.scatter(centres["lon"], centres["lat"], marker="x", color="tab:orange")
plt.show()

# %% [markdown]
# ## The scan over $\lambda$
#
# `stretch_range` is the default scan, log-symmetric about 1 from about 0.68 to
# 1.48 in steps of 0.03 in $\ln \lambda$.

# %%
stretches = stretch_range(stretch_max=1.5, step=0.03)
stretches

# %% [markdown]
# ## Closed shear lines
#
# Every closed orbit at every centre, stretching factor and branch, from two
# sections running 150 km due east and due west of each centre.

# %%
orbits = closed_shear_lines(
    forward,
    centre_lon=centres["lon"],
    centre_lat=centres["lat"],
    stretches=stretches,
    max_radius_m=150_000.0,
)
orbits

# %% [markdown]
# Each orbit's equivalent radius against the $\lambda$ it closed at, all centres
# together.

# %%
orbits.plot.scatter(x="stretch", y="radius_m")
plt.show()

# %% [markdown]
# ## The outermost boundary per vortex
#
# `outermost_shear_lines` keeps the largest orbit per centre and merges centres
# that one boundary encloses. With the first day's polar rotation it adds
# `rotation_sense`, counter-clockwise positive. South of the equator a clockwise
# eddy is cyclonic.

# %%
eddies = outermost_shear_lines(orbits, rotation=first_day.polar_rotation())
eddies

# %%
_, ax = plt.subplots()
log_anisotropy.plot.pcolormesh(
    x="lon_grid", y="lat_grid", ax=ax, cmap="Greys", robust=True
)
ax.plot(orbits["lon"].T, orbits["lat"].T, color="tab:green", lw=0.5)
ax.plot(eddies["lon"].T, eddies["lat"].T, color="tab:red", lw=2.0)
plt.show()

# %% [markdown]
# ## The polar rotation angle
#
# The rotation over the first day, which $C$ discards, with the boundaries on
# top.

# %%
_, ax = plt.subplots()
first_day.polar_rotation().plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax)
ax.plot(eddies["lon"].T, eddies["lat"].T, color="black", lw=1.5)
plt.show()

# %% [markdown]
# ## The boundaries at the end of the window
#
# A boundary is a material curve, so `forward.image` carries it to $t_1$
# through the flow map already computed, with no second Parcels run.

# %%
evolved = forward.image(lon_0=eddies["lon"], lat_0=eddies["lat"])

# %%
_, ax = plt.subplots()
log_anisotropy.plot.pcolormesh(
    x="lon_grid", y="lat_grid", ax=ax, cmap="Greys", robust=True
)
ax.plot(eddies["lon"].T, eddies["lat"].T, color="tab:red", lw=1.5)
ax.plot(evolved["lon"].T, evolved["lat"].T, color="tab:red", lw=1.5, ls="--")
plt.show()

# %% [markdown]
# ## In one call
#
# `elliptic_lcs` runs the centres, the search and the selection at the package
# defaults. It has only the 15-day flow map, so its boundaries carry no
# rotation sense.

# %%
forward.elliptic_lcs()

# %% [markdown]
# ## Outcome
#
# `elliptic_centres` picks 241 candidate centres, and `closed_shear_lines`
# closes 192 orbits at 13 of them. `outermost_shear_lines` keeps 12 boundaries,
# six turning counter-clockwise over the first day and six clockwise.
# `elliptic_lcs` returns the same 12 boundaries in one call.
