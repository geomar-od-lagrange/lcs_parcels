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
# # One Cape Cauldron vortex
#
# This notebook takes one of the vortices that `cape_cauldron_vortices` finds,
# the counter-clockwise one near 5.2 degrees E and 30.6 degrees S, and looks at
# it closely. It shows the whole family of closed orbits found at its centre,
# why `outermost_shear_lines` picks the one it does, and how the boundary sits
# in the Cauchy–Green anisotropy, the FTLE and the polar rotation angle. The
# Parcels setup is the one of `cape_cauldron_vortices`.

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
    rotation_sense,
    stretch_range,
)

# %% [markdown]
# ## Currents and field set
#
# The daily-mean GLORYS12 surface currents of `cape_cauldron_vortices`.

# %%
currents = xr.open_dataset("data/cape_cauldron_glorys_daily.nc").load()
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])

# %% [markdown]
# ## Seed grid and window
#
# A box of 4 degrees centred on the eddy keeps its boundary well inside the
# flow map and clear of the 50 km edge margin of `elliptic_centres`. The
# spacing is 1/50 degree, twice as fine as in `cape_cauldron_vortices` with a
# fifth of its particles. `closed_shear_lines` launches at the grid spacing, so
# the finer grid also gives a denser family of orbits. Release and window are
# unchanged.

# %%
t0 = np.datetime64("2018-06-01")
T = np.timedelta64(15, "D")
lon_axis = np.arange(3.2, 7.2 + 1e-9, 1 / 50)
lat_axis = np.arange(-32.6, -28.6 + 1e-9, 1 / 50)
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
seed

# %% [markdown]
# ## Recovery kernel
#
# Lost particles become `NaN` in place, as in `cape_cauldron_vortices`.


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
# The first day, for the polar rotation angle.

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

# %% [markdown]
# ## The centre
#
# `elliptic_centres` on the box, then the candidate nearest 5.2 degrees E and
# 30.6 degrees S. `isel` with a list keeps the `centre` dim, so the one centre
# goes into `closed_shear_lines` like a list of them.

# %%
anisotropy = forward.cg_anisotropy()
centres = elliptic_centres(anisotropy, extremum="min", window_m=100_000.0)
centres

# %%
distance = np.hypot(centres["lon"] - 5.2, centres["lat"] + 30.6)
centre = centres.sortby(distance).isel(centre=[0])
centre

# %% [markdown]
# ## Every closed orbit at the centre

# %%
orbits = closed_shear_lines(
    forward,
    centre_lon=centre["lon"],
    centre_lat=centre["lat"],
    stretches=stretch_range(stretch_max=1.5, step=0.03),
)
orbits

# %% [markdown]
# The outermost orbit and its rotation sense over the first day,
# counter-clockwise positive.

# %%
eddy = outermost_shear_lines(orbits)
eddy = rotation_sense(eddy, rotation=first_day.polar_rotation())
eddy[["stretch", "branch", "radius_m", "residual_m", "at_scan_edge"]]

# %%
eddy[["rotation_rad", "rotation_sense"]]

# %% [markdown]
# ## The orbit family
#
# Each orbit coloured by its stretching factor $\lambda$, solid on the $+$
# branch and dashed on the $-$ branch, over the anisotropy. The outermost
# orbit is drawn in red.

# %%
log_anisotropy = (
    np.log10(anisotropy)
    .rename("log10_cg_anisotropy")
    .assign_attrs(long_name="log10 of the Cauchy-Green eigenvalue ratio", units="1")
)

# %%
norm = plt.Normalize(orbits["stretch"].min(), orbits["stretch"].max())
_, ax = plt.subplots()
log_anisotropy.plot.pcolormesh(
    x="lon_grid", y="lat_grid", ax=ax, cmap="Greys", robust=True, add_colorbar=False
)
for orbit in orbits["orbit"]:
    one = orbits.sel(orbit=orbit)
    ax.plot(
        one["lon"],
        one["lat"],
        color=plt.cm.viridis(norm(one["stretch"])),
        ls="-" if one["branch"] > 0 else "--",
        lw=0.7,
    )
ax.plot(eddy["lon"].T, eddy["lat"].T, color="tab:red", lw=1.5)
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="stretch")
plt.show()

# %% [markdown]
# Equivalent radius against $\lambda$, one point per orbit, with the outermost
# orbit circled. It is the orbit of largest area, wherever its $\lambda$ lies
# in the scan.

# %%
_, ax = plt.subplots()
for branch, group in orbits.groupby("branch"):
    ax.scatter(group["stretch"], group["radius_m"], s=12, label=f"branch {branch:+d}")
ax.scatter(
    eddy["stretch"],
    eddy["radius_m"],
    s=120,
    facecolors="none",
    edgecolors="tab:red",
    label="outermost",
)
ax.set_xlabel("stretch")
ax.set_ylabel("radius_m [m]")
ax.legend()
plt.show()

# %% [markdown]
# The closure residual of each orbit, against the same axis. The closure
# tolerance is half the grid spacing.

# %%
_, ax = plt.subplots()
for branch, group in orbits.groupby("branch"):
    ax.scatter(group["stretch"], group["residual_m"], s=12, label=f"branch {branch:+d}")
ax.scatter(
    eddy["stretch"],
    eddy["residual_m"],
    s=120,
    facecolors="none",
    edgecolors="tab:red",
    label="outermost",
)
ax.axhline(orbits.attrs["closure_tol_m"], color="grey", ls=":", label="closure_tol_m")
ax.set_xlabel("stretch")
ax.set_ylabel("residual_m [m]")
ax.legend()
plt.show()

# %% [markdown]
# ## The fields with the boundary
#
# The anisotropy and the FTLE over the 15-day window, and the polar rotation
# angle over the first day, each with the boundary in black.

# %%
_, ax = plt.subplots()
log_anisotropy.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, robust=True)
ax.plot(eddy["lon"].T, eddy["lat"].T, color="black", lw=1.5)
plt.show()

# %%
ftle = forward.ftle()
_, ax = plt.subplots()
ftle.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, robust=True)
ax.plot(eddy["lon"].T, eddy["lat"].T, color="black", lw=1.5)
plt.show()

# %%
rotation = first_day.polar_rotation()
_, ax = plt.subplots()
rotation.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax)
ax.plot(eddy["lon"].T, eddy["lat"].T, color="black", lw=1.5)
plt.show()

# %% [markdown]
# ## Profiles through the centre
#
# Each field along the grid row through the centre, with dotted lines where the
# boundary crosses that latitude. The crossings are the boundary points within
# 0.02 degree of the centre's latitude.

# %%
centre_lat = float(centre["lat"].squeeze())
j_centre = int(np.abs(forward.ds["lat_grid"].isel(i=0) - centre_lat).argmin("j"))
crossings = eddy["lon"].where(np.abs(eddy["lat"] - centre_lat) < 0.02)
west, east = float(crossings.min()), float(crossings.max())
west, east

# %%
_, ax = plt.subplots()
log_anisotropy.isel(j=j_centre).plot(x="lon_grid", ax=ax)
ax.axvline(west, color="black", ls=":")
ax.axvline(east, color="black", ls=":")
plt.show()

# %%
_, ax = plt.subplots()
ftle.isel(j=j_centre).plot(x="lon_grid", ax=ax)
ax.axvline(west, color="black", ls=":")
ax.axvline(east, color="black", ls=":")
plt.show()

# %%
_, ax = plt.subplots()
rotation.isel(j=j_centre).plot(x="lon_grid", ax=ax)
ax.axvline(west, color="black", ls=":")
ax.axvline(east, color="black", ls=":")
plt.show()

# %% [markdown]
# ## Outcome
#
# The candidate nearest 5.2 degrees E and 30.6 degrees S is the grid point at
# 5.08 degrees E and 30.82 degrees S, and `closed_shear_lines` closes 73 orbits
# around it, all on the $+$ branch. They span $\lambda$ from 0.887 to 1.030 and
# radii from 19 km to 87 km, and the larger orbits close at the smaller
# $\lambda$. The outermost orbit closes at $\lambda = 0.887$, the smallest
# $\lambda$ with an orbit, with a radius of 87.4 km and a closure residual of
# 613 m against a tolerance of 957 m. `at_scan_edge` is False. Its centroid is at
# 5.21 degrees E and 30.57 degrees S, and it crosses the centre's latitude at
# 4.45 and 6.15 degrees E. Over the first day it turns counter-clockwise, with a
# mean rotation angle of 0.41 rad and `rotation_sense` $+1$, which south of the
# equator is anticyclonic.
