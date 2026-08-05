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

Run this once. It fetches the CMEMS subset that the three Cabo Verde notebooks
open from disk and writes it to `data/cabo_verde_currents_hourly.nc`. Re-running
it is cheap: if the file is already there and opens, nothing is downloaded.

`examples/data/` is gitignored, so the file is never committed and a fresh
clone has to run this notebook before the three Cabo Verde ones.

This assumes
[`copernicusmarine`](https://help.marine.copernicus.eu/en/collections/4060068-copernicus-marine-toolbox)
has credentials.

```python
from pathlib import Path

import copernicusmarine as cm
import xarray as xr
```

## The subset

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
```

## What landed on disk

```python
currents = xr.open_dataset(target)
currents
```
