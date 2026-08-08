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
# # Cabo Verde LCS on grid points that are not a grid
#
# `cabo_verde_lcs` and `cabo_verde_lcs_evolution` release one particle per cell
# of a rectilinear grid. This notebook does the same diagnostics over a set of
# grid points with no mesh behind them: dense where the eddies are, sparse
# elsewhere.
#
# What makes that possible is the auxiliary stencil. Four arms are released
# `aux_separation_m` around each grid point, and $\nabla F$ is differenced
# across a grid point's *own* arms, never against another grid point. The
# gradient at a point therefore does not depend on where the other points are.
# `UnstructuredAuxiliarySeedGrid.from_points` takes them as two flat arrays.
#
# Two consequences worth watching for below. The FTLE is as accurate in the
# sparse half as in the dense half, because the arms are 1 km apart in both, and
# the point density sets how finely the field is sampled. Reading the field
# *between* grid points, which `shrink_lines` and `FlowMap.image` both do, goes
# over the Delaunay triangulation of the points instead of along two axes.

# %% tags=["remove-output"]
# Importing Parcels pulls in the holoviews/bokeh bootstrap and prints an
# alpha-version notice, neither of which belongs on the rendered page.
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import UnstructuredAuxiliarySeedGrid

# %% [markdown]
# ## Currents
#
# The local file that `get_data` writes.

# %%
currents = xr.open_dataset("data/cabo_verde_currents_hourly.nc").load()

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
# ## A point set with two densities
#
# The same release box as the other Cabo Verde notebooks, sampled twice: a
# coarse cloud over all of it, and a fine cloud over the two degrees square in
# the middle where the eddy field is busiest. Both are drawn at random from a
# seeded generator, so the layout is reproducible without being a lattice.

# %%
t0 = np.datetime64("2025-08-06")
horizons = np.arange(1, 6) * np.timedelta64(1, "D")
box_lon, box_lat = (-27.0, -21.0), (13.5, 18.5)
patch_lon, patch_lat = (-25.5, -23.5), (15.0, 17.0)

# %%
rng = np.random.default_rng(20)
coarse_lon = rng.uniform(*box_lon, 4_000)
coarse_lat = rng.uniform(*box_lat, 4_000)
fine_lon = rng.uniform(*patch_lon, 6_000)
fine_lat = rng.uniform(*patch_lat, 6_000)

point_lon = np.concatenate([coarse_lon, fine_lon])
point_lat = np.concatenate([coarse_lat, fine_lat])

# %%
seed = UnstructuredAuxiliarySeedGrid.from_points(lon=point_lon, lat=point_lat)
print(seed)
seed.ds

# %% [markdown]
# The mean spacing comes out near 9 km in the coarse cloud and near 3 km in the
# patch, against the 1 km arms that measure the gradient at each point.

# %%
for name, lon, lat in (
    ("coarse", coarse_lon, coarse_lat),
    ("patch", fine_lon, fine_lat),
):
    area_m2 = (
        np.ptp(lon) * np.ptp(lat) * (111_195.0**2) * np.cos(np.deg2rad(np.mean(lat)))
    )
    print(
        f"{name}: {lon.size} points, mean spacing {np.sqrt(area_m2 / lon.size):.0f} m"
    )


# %% [markdown]
# ## Recovery kernel
#
# Particles that leave the domain or hit land are turned into `NaN` in place
# (Parcels would otherwise abort the run), so losses propagate as `NaN` through
# every diagnostic below. `StatusCode.EndofLoop` rather than
# `StatusCode.Delete`: deleting shrinks the particle array and breaks the
# alignment with the seed order.


# %%
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)


# %% [markdown]
# ## Advect to each horizon
#
# Four arms per grid point, so the particle set is four times the point count.
# The horizons are reached one leg at a time, each leg starting a new
# `ParticleSet` from where the previous one ended, because Parcels interpolates
# a particle set on the assumption that every particle shares one clock and a
# beached particle's clock stays behind.

# %%
forward_maps, backward_maps = [], []
for direction, maps in ((1, forward_maps), (-1, backward_maps)):
    lon, lat = (np.asarray(a) for a in seed.to_parcels_pset())
    start = np.timedelta64(0, "s")
    for horizon in horizons:
        pset = ParticleSet(
            fieldset,
            pclass=Particle,
            x=lon,
            y=lat,
            z=np.full(lon.size, z_surface),
            t=t0 + direction * start,
        )
        pset.execute(
            [AdvectionRK4, set_lost_to_nan],
            dt=direction * np.timedelta64(1, "h"),
            runtime=horizon - start,
            verbose_progress=False,
        )
        lon, lat = np.asarray(pset.x).copy(), np.asarray(pset.y).copy()
        start = horizon
        maps.append(
            seed.pset_to_flowmap(lon=lon, lat=lat, t0=t0, t1=t0 + direction * horizon)
        )

# %%
forward = forward_maps[-1]
backward = backward_maps[-1]
forward

# %% [markdown]
# ## The FTLE on a point set
#
# One value per grid point, on the `grid_point` dim the flat arrays became.

# %%
ftle_forward = forward.ftle()
ftle_forward

# %% [markdown]
# There are no axes to shade the field over, so it is shaded over the Delaunay
# triangulation of the grid points instead: the same triangulation the package
# interpolates $C$ on when it traces the tensor lines below. The patch samples
# the same filaments about three times more finely than the surrounding cloud.
# The gradient at every one of these points was measured over the same 1 km arms.
#
# Triangles with a lost grid point at a corner are masked out, which is where
# the islands come from. The interpolator returns NaN over the same triangles,
# so a traced line stops at the white.

# %%
triangulation = mtri.Triangulation(
    ftle_forward["lon_grid"].values, ftle_forward["lat_grid"].values
)
triangulation.set_mask(
    np.isnan(ftle_forward.values)[triangulation.triangles].any(axis=1)
)

# %%
fig, ax = plt.subplots(figsize=(7, 6))
shading = ax.tripcolor(
    triangulation,
    ftle_forward.values * 86_400.0,
    shading="gouraud",
    cmap="magma",
)
ax.set_xlabel("longitude [degrees_east]")
ax.set_ylabel("latitude [degrees_north]")
fig.colorbar(shading, ax=ax, label="finite-time Lyapunov exponent [1/day]")
plt.show()

# %% [markdown]
# ## LCS from the same point set
#
# `hyperbolic_lcs` runs the FTLE, the ridge seeding and the tensor lines in one
# call, and none of the three needs a mesh. Seeds are picked over a
# neighbourhood measured on the sphere, so `window_m` means the same 30 km in
# the sparse half as in the dense half. The lines are traced through $C$ read
# off the triangulation, and stop where they leave its convex hull.

# %%
repelling_lcs = forward.hyperbolic_lcs()
attracting_lcs = backward.hyperbolic_lcs()
print(
    f"{repelling_lcs.sizes['line']} repelling, "
    f"{attracting_lcs.sizes['line']} attracting lines"
)

# %%
fig, ax = plt.subplots(figsize=(7, 6))
shading = ax.tripcolor(
    triangulation,
    ftle_forward.values * 86_400.0,
    shading="gouraud",
    cmap="Greys",
)
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
ax.set_xlabel("longitude [degrees_east]")
ax.set_ylabel("latitude [degrees_north]")
fig.colorbar(shading, ax=ax, label="finite-time Lyapunov exponent [1/day]")
plt.show()

# %% [markdown]
# ## Evolving the curves
#
# An extracted LCS is a material curve, so `FlowMap.image` carries it to each
# horizon without re-diagnosing it (see `cabo_verde_lcs_evolution`). On a point
# set `image` reads the advected-position field off the triangulation, which is
# the one thing in the chain that the layout changes: a curve leaving the convex
# hull of the grid points terminates there rather than at a box edge.
#
# Each family is evolved in its coherent direction, the attracting curve by the
# forward maps and the repelling curve by the backward ones.

# %%
offset_days = np.concatenate([[0.0], horizons / np.timedelta64(1, "D")])
OFFSET_ATTRS = {"long_name": "signed offset from t0", "units": "days"}

# %%
attracting_evo = xr.concat(
    [
        attracting_lcs[["lon", "lat"]],
        *(
            m.image(lon_0=attracting_lcs["lon"], lat_0=attracting_lcs["lat"])
            for m in forward_maps
        ),
    ],
    dim="offset",
).assign_coords(offset=("offset", offset_days, OFFSET_ATTRS))

# %%
repelling_evo = xr.concat(
    [
        repelling_lcs[["lon", "lat"]],
        *(
            m.image(lon_0=repelling_lcs["lon"], lat_0=repelling_lcs["lat"])
            for m in backward_maps
        ),
    ],
    dim="offset",
).assign_coords(offset=("offset", -offset_days, OFFSET_ATTRS))
repelling_evo

# %% [markdown]
# ## Snapshots
#
# Each family day by day, with its own $t_0$ position drawn faintly in every
# panel for reference.

# %%
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
