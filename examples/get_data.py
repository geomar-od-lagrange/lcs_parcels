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
# # Download CMEMS data
#
# Run this once. It fetches the CMEMS subsets the example notebooks open from
# disk: `data/cabo_verde_currents_hourly.nc` for the Cabo Verde notebooks and
# `data/cape_cauldron_currents_hourly.nc` for `cape_cauldron_vortices`.
# Re-running it is cheap: if a file is already there and opens, nothing is
# downloaded for it.
#
# `examples/data/` is gitignored, so neither file is committed and a fresh
# clone has to run this notebook before any of the other examples.
#
# This assumes
# [`copernicusmarine`](https://help.marine.copernicus.eu/en/collections/9080063-copernicus-marine-toolbox)
# has credentials.

# %%
from pathlib import Path

import copernicusmarine as cm
import xarray as xr

# %% [markdown]
# ## The Cabo Verde subset
#
# Hourly surface velocity (`uo`, `vo`) from Global Ocean Physics Analysis and
# Forecast, `GLOBAL_ANALYSISFORECAST_PHY_001_024`,
# [doi:10.48670/moi-00016](https://doi.org/10.48670/moi-00016), dataset
# `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`. A box around Cabo Verde, first depth
# level only. The examples release particles between 2025-08-01 and 2025-08-11
# inside a smaller box; the extra day at each end and the margin in longitude and
# latitude are there so trajectories stay inside the data.

# %%
target = Path("data/cabo_verde_currents_hourly.nc")
target.parent.mkdir(parents=True, exist_ok=True)

# %%
try:
    xr.open_dataset(target).close()
    have_file = True
except (FileNotFoundError, OSError):
    have_file = False

print(
    f"{target}: already here, skipping the download"
    if have_file
    else f"{target}: missing, downloading it"
)

# %%
if not have_file:
    ds = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        variables=["uo", "vo"],
        minimum_longitude=-30.5,
        maximum_longitude=-17.5,
        minimum_latitude=10.0,
        maximum_latitude=22.0,
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime="2025-07-31",
        end_datetime="2025-08-12",
    ).load()
    ds.to_netcdf(target)

print(f"{target}: {target.stat().st_size / 1e6:.0f} MB on disk")

# %% [markdown]
# ## What landed on disk for Cabo Verde

# %%
currents = xr.open_dataset(target)
currents

# %% [markdown]
# ## The Cape Cauldron subset
#
# Same dataset and variables, a box around the Cape Cauldron, the Agulhas-ring
# corridor south-west of South Africa. `cape_cauldron_vortices` releases
# particles on 2025-06-01 and advects them to 2025-06-16 inside a smaller box.
# The margins here again keep trajectories inside the data.

# %%
target_cape = Path("data/cape_cauldron_currents_hourly.nc")
target_cape.parent.mkdir(parents=True, exist_ok=True)

# %%
try:
    xr.open_dataset(target_cape).close()
    have_file_cape = True
except (FileNotFoundError, OSError):
    have_file_cape = False

print(
    f"{target_cape}: already here, skipping the download"
    if have_file_cape
    else f"{target_cape}: missing, downloading it"
)

# %%
if not have_file_cape:
    ds_cape = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
        variables=["uo", "vo"],
        minimum_longitude=3.0,
        maximum_longitude=21.0,
        minimum_latitude=-43.0,
        maximum_latitude=-29.0,
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime="2025-05-31",
        end_datetime="2025-06-17",
    ).load()
    ds_cape.to_netcdf(target_cape)

print(f"{target_cape}: {target_cape.stat().st_size / 1e6:.0f} MB on disk")

# %% [markdown]
# ## What landed on disk for the Cape Cauldron

# %%
currents_cape = xr.open_dataset(target_cape)
currents_cape
