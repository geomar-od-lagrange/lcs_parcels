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

# %%

# %% [markdown]
# # Cape Cauldron coherent vortices
#
# The Cape Cauldron is the Agulhas-ring corridor south-west of South Africa.
# Haller & Beron-Vera (2013,
# [doi:10.1017/jfm.2013.391](https://doi.org/10.1017/jfm.2013.391)) found
# coherent Lagrangian vortices there with the construction this notebook uses.
#
# The FTLE ridges of `cabo_verde_ftle` mark hyperbolic stretching. A coherent
# Lagrangian vortex is a region that stays together as it advects, bounded by a
# material loop that stretches uniformly. From the same Cauchy–Green tensor
# $C = (\nabla F)^\top \nabla F$, with eigenpairs $C\xi_i = \lambda_i \xi_i$ and
# $0 < \lambda_1 \le \lambda_2$, Haller & Beron-Vera (2013, Eq. 14) build the
# direction fields
#
# $$
# \eta_\lambda^{\pm} = \sqrt{\frac{\lambda_2 - \lambda^2}{\lambda_2 - \lambda_1}}\,\xi_1
#   \;\pm\; \sqrt{\frac{\lambda^2 - \lambda_1}{\lambda_2 - \lambda_1}}\,\xi_2 \,,
# $$
#
# defined where $\lambda_1 < \lambda^2 < \lambda_2$. They satisfy
# $|\nabla F\,\eta_\lambda| = \lambda$ at every point, so a curve tangent to one
# has its arc length multiplied by $\lambda$ over the window. A closed curve
# tangent to one is a *closed shear line*. Each vortex holds a nested family of
# them over $\lambda$, and the outermost member is its boundary.
#
# `closed_shear_lines` finds them with the return map of Haller & Beron-Vera
# (2013). Lines are launched along a short section from a candidate centre and
# traced until they cross it again, and a launch that comes back to itself is a
# closed orbit.

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
# Hourly CMEMS surface currents at 1/12 degree, the local file that `get_data`
# writes.

# %%
currents = xr.open_dataset("data/cape_cauldron_currents_hourly.nc").load()
currents

# %% [markdown]
# ## Parcels v4 field set
#
# `copernicusmarine_to_sgrid` tags the CMEMS A-grid with SGRID metadata, and
# `from_sgrid_conventions` wraps it as a spherical `FieldSet`.

# %%
sgrid = copernicusmarine_to_sgrid(fields={"U": currents["uo"], "V": currents["vo"]})
fieldset = FieldSet.from_sgrid_conventions(sgrid, mesh="spherical")
z_surface = float(currents["depth"].values[0])

# %% [markdown]
# ## Seed grid and window
#
# A box across the Cape Cauldron at 1/25 degree, released on 2025-06-01 and read
# at 30 days. An Agulhas ring turns once in roughly 7 to 12 days, so the window
# covers a few revolutions. A vortex boundary belongs to one window, so only the
# forward flow map is needed.

# %%
t0 = np.datetime64("2025-06-01")
T = np.timedelta64(30, "D")
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

# %%
forward = seed.pset_to_flowmap(lon=pset.x, lat=pset.y, t0=t0, t1=t0 + T)
forward

# %% [markdown]
# A second run over the first day gives the rotation of the flow. The polar
# rotation angle is defined modulo $2\pi$, and an Agulhas ring turns several
# times in 30 days, so its sign gives the rotation sense only over a short
# window.

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
# Particles lost to land or to the domain edge over the 30 days.

# %%
int(np.isnan(np.asarray(pset.x)).sum())

# %% [markdown]
# ## Eigendecomposition and FTLE
#
# The FTLE is the backdrop the centres and the boundaries are drawn on below.

# %%
eigen = forward.cg_eigen()
eigen

# %%
ftle = forward.ftle()
ftle.plot.pcolormesh(x="lon_grid", y="lat_grid")

# %% [markdown]
# ## Where $\eta_\lambda$ exists, at $\lambda = 1$
#
# $\eta_1$ needs $\lambda_1 < 1 < \lambda_2$, a neighbourhood stretched along one
# eigendirection and contracted along the other. Where that fails, no direction
# has a stretching factor of 1, and a traced line ends.

# %%
lambda1 = eigen["lambda"].isel(eig=0)
lambda2 = eigen["lambda"].isel(eig=1)
((lambda1 < 1.0) & (1.0 < lambda2)).plot.pcolormesh(x="lon_grid", y="lat_grid")

# %% [markdown]
# ## How far from area-preserving
#
# $|\det \nabla F| = \sqrt{\lambda_1\lambda_2}$ is the area change of a grid cell
# over the window. A loop whose interior changes area by $J$ and keeps its shape
# changes its length by $\sqrt{J}$, so a boundary closes near
# $\lambda = \sqrt{J}$ rather than at 1, and the search scans a range of
# $\lambda$.

# %%
np.sqrt(lambda1 * lambda2).quantile([0.05, 0.25, 0.5, 0.75, 0.95])

# %% [markdown]
# ## Candidate centres
#
# Windowed minima of $\lambda_2 / \lambda_1$, the points where the
# eigendirections come closest to undefined, over a 100 km window and at least
# 50 km from the domain edge and from land.

# %%
anisotropy = (lambda2 / lambda1).assign_attrs(
    long_name="Cauchy-Green eigenvalue ratio", units="1"
)
centres = elliptic_centres(anisotropy, extremum="min", window_m=100_000.0)
centres

# %%
_, ax = plt.subplots()
ftle.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.scatter(centres["lon"], centres["lat"], marker="x", color="tab:orange")
plt.show()

# %% [markdown]
# ## The scan over $\lambda$
#
# `stretch_range` is the default scan, log-symmetric about 1 from $1/1.5$ to
# $1.5$ in steps of 0.03 in $\ln \lambda$, so divergence and convergence are
# bracketed alike.

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
# Each orbit's equivalent radius against the $\lambda$ it closed at. One vortex
# shows up as a run of orbits growing with $\lambda$.

# %%
_, ax = plt.subplots()
ax.scatter(orbits["stretch"], orbits["radius_m"] / 1000)
ax.set_xlabel("stretching factor lambda")
ax.set_ylabel("equivalent radius (km)")
plt.show()

# %% [markdown]
# ## The outermost boundary per vortex
#
# `outermost_shear_lines` keeps the largest orbit per centre and merges centres
# that one boundary encloses. `rotation_sense` is the sign of the first day's
# polar rotation angle inside the boundary, counter-clockwise positive. South of
# the equator a clockwise eddy is cyclonic.

# %%
eddies = outermost_shear_lines(orbits, rotation=first_day.polar_rotation())
eddies

# %%
_, ax = plt.subplots()
ftle.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
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
# ## Material check
#
# A boundary stretches every tangent element by its $\lambda$, so its arc length
# over the window should grow by that factor. `forward.image` carries a curve
# through the flow map already computed, with no second Parcels run. A circle of
# the same equivalent radius about the same centre is not tangent to
# $\eta_\lambda$, so nothing constrains its ratio.


# %%
def arc_length_m(curve):
    lon, lat = curve["lon"], curve["lat"]
    lat_mid = np.deg2rad(0.5 * (lat + lat.shift(point=-1)))
    dx = 6_371_000.0 * np.cos(lat_mid) * np.deg2rad(lon.shift(point=-1) - lon)
    dy = 6_371_000.0 * np.deg2rad(lat.shift(point=-1) - lat)
    return np.hypot(dx, dy).sum("point")


boundary = eddies.isel(eddy=0)
evolved = forward.image(lon_0=boundary["lon"], lat_0=boundary["lat"])

angle = xr.DataArray(np.linspace(0.0, 2 * np.pi, 361), dims="point")
radius_deg = boundary["radius_m"] / 111_195.0
circle = xr.Dataset(
    {
        "lon": boundary["centroid_lon"]
        + radius_deg * np.cos(angle) / np.cos(np.deg2rad(boundary["centroid_lat"])),
        "lat": boundary["centroid_lat"] + radius_deg * np.sin(angle),
    }
)
evolved_circle = forward.image(lon_0=circle["lon"], lat_0=circle["lat"])

# %% [markdown]
# The boundary's $\lambda$, its arc-length ratio, and the circle's.

# %%
(
    float(boundary["stretch"]),
    float(arc_length_m(evolved) / arc_length_m(boundary)),
    float(arc_length_m(evolved_circle) / arc_length_m(circle)),
)

# %%
_, ax = plt.subplots()
ftle.plot.pcolormesh(x="lon_grid", y="lat_grid", ax=ax, cmap="Greys")
ax.plot(boundary["lon"].T, boundary["lat"].T, color="tab:red", label="boundary, t0")
ax.plot(evolved["lon"].T, evolved["lat"].T, color="tab:red", ls="--", label="t1")
ax.plot(circle["lon"].T, circle["lat"].T, color="tab:blue", label="circle, t0")
ax.plot(
    evolved_circle["lon"].T,
    evolved_circle["lat"].T,
    color="tab:blue",
    ls="--",
    label="t1",
)
ax.legend()
plt.show()


# %% [markdown]
# ## Outcome
#
# Of the 501 by 401 seeded particles, 24500 are lost to land or the domain edge.
# $|\det \nabla F|$ runs from 0.172 at the 5 percent quantile to 675 at the 95
# percent, with a median of 4.96.
#
# The search over 227 candidate centres closes 25 orbits, all at one centre,
# 13.32 E 35.64 S. The outermost one is the single boundary, at
# $\lambda = 1.350$ with an equivalent radius of 41.1 km and its centroid at
# 13.28 E 35.64 S. Over the first day it turns counter-clockwise, which south of
# the equator is anticyclonic.
#
# Carried through the flow map, the boundary's arc length grows by 1.298
# against its $\lambda$ of 1.350. A circle of the same equivalent radius about
# the same centroid grows by 1.550.
