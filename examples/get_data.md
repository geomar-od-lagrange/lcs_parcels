---
jupyter:
  jupytext:
    cell_metadata_filter: tags,-all
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

# Download CMEMS data

Run this once. It fetches the subsets the example notebooks open from disk:
`data/cabo_verde_currents_hourly.nc` for the Cabo Verde notebooks,
`data/cape_cauldron_glorys_daily.nc` for both Cape Cauldron notebooks, and
`data/cape_cauldron_geostrophic_daily.nc` plus `data/gled/` for
`cape_cauldron_gled`. Re-running it is cheap, because nothing is downloaded for
a file that is already there and opens.

`examples/data/` is gitignored, so none of these are committed and a fresh
clone has to run this notebook before any of the other examples.

This assumes
[`copernicusmarine`](https://help.marine.copernicus.eu/en/collections/9080063-copernicus-marine-toolbox)
has credentials.

```python
import tarfile
import urllib.request
from pathlib import Path

import copernicusmarine as cm
import xarray as xr
```

## The Cabo Verde subset

Hourly surface velocity (`uo`, `vo`) from Global Ocean Physics Analysis and
Forecast, `GLOBAL_ANALYSISFORECAST_PHY_001_024`,
[doi:10.48670/moi-00016](https://doi.org/10.48670/moi-00016), dataset
`cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`. A box around Cabo Verde, first depth
level only. The examples release particles between 2025-08-01 and 2025-08-11
inside a smaller box; the extra day at each end and the margin in longitude and
latitude are there so trajectories stay inside the data.

```python
target = Path("data/cabo_verde_currents_hourly.nc")
target.parent.mkdir(parents=True, exist_ok=True)
```

```python
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
```

```python
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
```

## What landed on disk for Cabo Verde

```python
currents = xr.open_dataset(target)
currents
```

## The Cape Cauldron model subset

Daily-mean surface currents `uo`/`vo` and sea surface height `zos` from the
GLORYS12 reanalysis `GLOBAL_MULTIYEAR_PHY_001_030`,
[doi:10.48670/moi-00021](https://doi.org/10.48670/moi-00021), dataset
`cmems_mod_glo_phy_my_0.083deg_P1D-m`, over a box around the Cape Cauldron,
the Agulhas-ring corridor south-west of South Africa. The Cape Cauldron
notebooks release particles on 2018-06-01 in the box 2E to 22E, 44S to 28S
and advect them 30 days. This subset carries an 8 degree margin on every side
of that box so trajectories stay inside the data.

```python
target_cape = Path("data/cape_cauldron_glorys_daily.nc")
target_cape.parent.mkdir(parents=True, exist_ok=True)
```

```python
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
```

```python
if not have_file_cape:
    ds_cape = cm.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["uo", "vo", "zos"],
        minimum_longitude=-6.0,
        maximum_longitude=30.0,
        minimum_latitude=-52.0,
        maximum_latitude=-20.0,
        minimum_depth=0.0,
        maximum_depth=1.0,
        start_datetime="2018-05-31",
        end_datetime="2018-07-03",
    ).load()
    ds_cape.to_netcdf(target_cape)

print(f"{target_cape}: {target_cape.stat().st_size / 1e6:.0f} MB on disk")
```

## What landed on disk for the Cape Cauldron model subset

```python
currents_cape = xr.open_dataset(target_cape)
currents_cape
```

## The Cape Cauldron geostrophic subset

Daily altimetry-derived geostrophic velocity (`ugos`, `vgos`) from the
global sea level product `SEALEVEL_GLO_PHY_L4_MY_008_047`,
[doi:10.48670/moi-00148](https://doi.org/10.48670/moi-00148), dataset
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`. Same Cape
Cauldron box as the currents subset above, 2018-05-31 to 2018-07-03 to cover
the `cape_cauldron_gled` notebook's release window with margin at each end.

```python
target_geo = Path("data/cape_cauldron_geostrophic_daily.nc")
target_geo.parent.mkdir(parents=True, exist_ok=True)
```

```python
try:
    xr.open_dataset(target_geo).close()
    have_file_geo = True
except (FileNotFoundError, OSError):
    have_file_geo = False

print(
    f"{target_geo}: already here, skipping the download"
    if have_file_geo
    else f"{target_geo}: missing, downloading it"
)
```

```python
if not have_file_geo:
    ds_geo = cm.open_dataset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["ugos", "vgos"],
        minimum_longitude=-6.0,
        maximum_longitude=30.0,
        minimum_latitude=-52.0,
        maximum_latitude=-20.0,
        start_datetime="2018-05-31",
        end_datetime="2018-07-03",
    ).load()
    ds_geo.to_netcdf(target_geo)

print(f"{target_geo}: {target_geo.stat().st_size / 1e6:.0f} MB on disk")
```

## What landed on disk for the Cape Cauldron geostrophic subset

```python
geostrophic = xr.open_dataset(target_geo)
geostrophic
```

## The GLED coherent-eddy records

GLED v1.0 (Liu & Abernathey 2023,
[doi:10.5194/essd-15-1765-2023](https://doi.org/10.5194/essd-15-1765-2023)),
data archive at Zenodo record
[7349753](https://zenodo.org/records/7349753),
[doi:10.5281/zenodo.7349753](https://doi.org/10.5281/zenodo.7349753).

`eddyinfo.tar.gz` holds one JSON file per tracking duration, each mapping an
eddy id to `date_start`, `radius` (km), `cyc` (+1 anticyclonic, -1
cyclonic), and `center_lon`/`center_lat` (0-360) sampled every 10 days.

```python
gled_dir = Path("data/gled")
gled_json = gled_dir / "eddyinfo" / "eddy_info_30d.json"

if gled_json.exists():
    print(f"{gled_json}: already here, skipping the download")
else:
    print(f"{gled_json}: missing, downloading and extracting the archive")
    gled_dir.mkdir(parents=True, exist_ok=True)
    archive = gled_dir / "eddyinfo.tar.gz"
    urllib.request.urlretrieve(
        "https://zenodo.org/records/7349753/files/eddyinfo.tar.gz", archive
    )
    print(f"{archive}: {archive.stat().st_size / 1e6:.0f} MB downloaded")
    with tarfile.open(archive) as tar:
        tar.extractall(gled_dir, filter="data")
        extracted = tar.getnames()
    archive.unlink()
    print("extracted:", extracted)
```

## What landed on disk for GLED

```python
sorted(p.name for p in (gled_dir / "eddyinfo").iterdir())
```
