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

# Cape Cauldron shear lines against the GLED eddy atlas

GLED v1.0 (Liu & Abernathey 2023,
[doi:10.5194/essd-15-1765-2023](https://doi.org/10.5194/essd-15-1765-2023),
data [doi:10.5281/zenodo.7349753](https://doi.org/10.5281/zenodo.7349753))
lists coherent Lagrangian eddies found from the Lagrangian averaged vorticity
deviation (Haller et al. 2016, J. Fluid Mech. 795,
[doi:10.1017/jfm.2016.151](https://doi.org/10.1017/jfm.2016.151)) on
trajectories advected with altimetry geostrophic currents. Its windows are
30 days long and start on the first of each month.

This notebook runs the closed-shear-line search of `cape_cauldron_vortices`
on the same kind of field over the same window, 30 days forward from
2018-06-01 at 00:00, and compares what the two methods find. GLED gives each
eddy's centre and radius at the release time, the time the Cauchy-Green field
refers to. GLED advects with the 1/4 degree DUACS product, so the search runs
on the 1/8 degree product and on its 1/4 degree block mean.

The comparison runs twice per resolution. Sweep A takes its candidate centres
from the Cauchy-Green field, so it tests the whole detection chain. Sweep B
takes the GLED centres, so it tests only whether a closed shear line exists
where GLED puts an eddy.

```python tags=["remove-output"]
# Importing Parcels pulls in the holoviews/bokeh bootstrap and prints an
# alpha-version notice, neither of which belongs on the rendered page.
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from parcels import FieldSet, Particle, ParticleSet, StatusCode
from parcels.convert import copernicusmarine_to_sgrid
from parcels.kernels import AdvectionRK4

from lcs_parcels import NeighborSeedGrid, closed_shear_lines, outermost_shear_lines
```

## Currents

Daily altimetry geostrophic velocity at 1/8 degree, the local file that
`get_data` writes.

```python
currents_eighth = xr.open_dataset("data/cape_cauldron_geostrophic_daily.nc").load()
currents_eighth
```

The 1/4 degree field is the 2 by 2 block mean of the 1/8 degree one, whose
cell centres fall on those of the 1/4 degree product. The mean skips land, so
a block that is part land takes the mean of its ocean cells.

GLED also removes the divergence that the latitude dependence of the Coriolis
parameter gives a geostrophic velocity. Neither field here has that
correction.

```python
currents_quarter = currents_eighth.coarsen(
    latitude=2, longitude=2, boundary="trim"
).mean()
currents_quarter
```

## Parcels v4 field sets

The product carries no depth coordinate, so the fields have no `Z` axis and
the particle sets below are released with no `z`.

```python
fieldset_eighth = FieldSet.from_sgrid_conventions(
    copernicusmarine_to_sgrid(
        fields={"U": currents_eighth["ugos"], "V": currents_eighth["vgos"]}
    ),
    mesh="spherical",
)
fieldset_quarter = FieldSet.from_sgrid_conventions(
    copernicusmarine_to_sgrid(
        fields={"U": currents_quarter["ugos"], "V": currents_quarter["vgos"]}
    ),
    mesh="spherical",
)
```

## Seed grid and window

The GLED box at 1/25 degree for both fields, released on 2018-06-01 and read
at 30 days, the window of the `eddy_info_30d` product.

```python
t0 = np.datetime64("2018-06-01")
T = np.timedelta64(30, "D")
lon_axis = np.arange(2.0, 22.0 + 1e-9, 1 / 25)
lat_axis = np.arange(-44.0, -28.0 + 1e-9, 1 / 25)
seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
seed
```

## Recovery kernel

Particles that leave the domain or hit land are turned into `NaN` in place
(Parcels would otherwise abort the run), so losses propagate as `NaN`
through every diagnostic below. The kernel sets `StatusCode.EndofLoop`
rather than `StatusCode.Delete`, because deleting shrinks the particle
array and breaks the alignment with the seed order.


```python
def set_lost_to_nan(particles, fieldset):
    lost = particles.state >= StatusCode.Error
    particles.x = np.where(lost, np.nan, particles.x)
    particles.y = np.where(lost, np.nan, particles.y)
    particles.state = np.where(lost, StatusCode.EndofLoop, particles.state)
```

## Advect forward

```python
lon, lat = seed.to_parcels_pset()
```

```python
# 1/8 degree
pset_eighth = ParticleSet(fieldset_eighth, pclass=Particle, x=lon, y=lat, t=t0)
pset_eighth.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=False,
)
forward_eighth = seed.pset_to_flowmap(
    lon=pset_eighth.x, lat=pset_eighth.y, t0=t0, t1=t0 + T
)
forward_eighth
```

```python
# 1/4 degree
pset_quarter = ParticleSet(fieldset_quarter, pclass=Particle, x=lon, y=lat, t=t0)
pset_quarter.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=T,
    verbose_progress=False,
)
forward_quarter = seed.pset_to_flowmap(
    lon=pset_quarter.x, lat=pset_quarter.y, t0=t0, t1=t0 + T
)
forward_quarter
```

A second run over the first day at each resolution gives the rotation of the
flow. The polar rotation angle is defined modulo $2\pi$, and an eddy turns
several times in 30 days, so its sign gives the rotation sense only over a
short window.

```python
one_day = np.timedelta64(1, "D")
```

```python
# 1/8 degree
pset_day = ParticleSet(fieldset_eighth, pclass=Particle, x=lon, y=lat, t=t0)
pset_day.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=one_day,
    verbose_progress=False,
)
first_day_eighth = seed.pset_to_flowmap(
    lon=pset_day.x, lat=pset_day.y, t0=t0, t1=t0 + one_day
)
```

```python
# 1/4 degree
pset_day = ParticleSet(fieldset_quarter, pclass=Particle, x=lon, y=lat, t=t0)
pset_day.execute(
    [AdvectionRK4, set_lost_to_nan],
    dt=np.timedelta64(1, "h"),
    runtime=one_day,
    verbose_progress=False,
)
first_day_quarter = seed.pset_to_flowmap(
    lon=pset_day.x, lat=pset_day.y, t0=t0, t1=t0 + one_day
)
```

Particles lost to land or to the domain edge over the 30 days, at 1/8 and at
1/4 degree.

```python
(
    int(np.isnan(np.asarray(pset_eighth.x)).sum()),
    int(np.isnan(np.asarray(pset_quarter.x)).sum()),
)
```

## The GLED records

`eddy_info_30d.json` and `eddy_info_90d.json` each map an eddy id to its
`date_start`, `radius` in km, polarity `cyc`, and centre position sampled
every 10 days. Index 0 of `center_lon`/`center_lat` is the position at the
start date, which is what selects an eddy into the box here.


```python
def load_gled(path, *, date_start, lon_bounds, lat_bounds):
    """GLED eddies starting on `date_start` whose start position is in the box."""
    with open(path) as f:
        raw = json.load(f)
    ids, gled_lon, gled_lat, radius_km, polarity = [], [], [], [], []
    for key, start in raw["date_start"].items():
        if start != date_start:
            continue
        # GLED stores 0-360 longitudes, and this box is east of Greenwich.
        lon_0 = ((raw["center_lon"][key][0] + 180.0) % 360.0) - 180.0
        lat_0 = raw["center_lat"][key][0]
        if not (lon_bounds[0] <= lon_0 <= lon_bounds[1]):
            continue
        if not (lat_bounds[0] <= lat_0 <= lat_bounds[1]):
            continue
        ids.append(raw["id"][key])
        gled_lon.append(lon_0)
        gled_lat.append(lat_0)
        radius_km.append(raw["radius"][key])
        polarity.append(raw["cyc"][key])
    return xr.Dataset(
        {
            "lon": (
                "gled",
                np.array(gled_lon, dtype=float),
                {"long_name": "GLED eddy centre longitude", "units": "degrees_east"},
            ),
            "lat": (
                "gled",
                np.array(gled_lat, dtype=float),
                {"long_name": "GLED eddy centre latitude", "units": "degrees_north"},
            ),
            "radius_m": (
                "gled",
                np.array(radius_km, dtype=float) * 1000.0,
                {"long_name": "GLED eddy radius", "units": "m"},
            ),
            "polarity": (
                "gled",
                np.array(polarity, dtype=int),
                {
                    "long_name": "GLED eddy polarity, 1 anticyclonic and -1 cyclonic",
                    "units": "1",
                },
            ),
        },
        coords={"gled": ("gled", ids, {"long_name": "GLED eddy identifier"})},
    )
```

```python
gled_30 = load_gled(
    "data/gled/eddyinfo/eddy_info_30d.json",
    date_start="2018-06-01",
    lon_bounds=(2.0, 22.0),
    lat_bounds=(-44.0, -28.0),
)
gled_90 = load_gled(
    "data/gled/eddyinfo/eddy_info_90d.json",
    date_start="2018-06-01",
    lon_bounds=(2.0, 22.0),
    lat_bounds=(-44.0, -28.0),
)
gled_30
```

## Sweep A, the whole detection chain

`elliptic_lcs` picks candidate centres at windowed minima of
$\lambda_2 / \lambda_1$, searches them for closed shear lines, and keeps the
outermost boundary per vortex, all at the package defaults. It has only the
30-day flow map, so its boundaries carry no rotation sense.

```python
eddies_a_eighth = forward_eighth.elliptic_lcs()
eddies_a_eighth
```

```python
eddies_a_quarter = forward_quarter.elliptic_lcs()
eddies_a_quarter
```

## Sweep B, from the GLED centres

The same search launched from the GLED 30-day eddy centres. A centre GLED
found cannot be missed here, so this separates centre selection from the
existence of a closed shear line. The boundaries take their rotation sense
from the first day.

```python
eddies_b_eighth = outermost_shear_lines(
    closed_shear_lines(
        forward_eighth, centre_lon=gled_30["lon"], centre_lat=gled_30["lat"]
    ),
    rotation=first_day_eighth.polar_rotation(),
)
eddies_b_quarter = outermost_shear_lines(
    closed_shear_lines(
        forward_quarter, centre_lon=gled_30["lon"], centre_lat=gled_30["lat"]
    ),
    rotation=first_day_quarter.polar_rotation(),
)
eddies_b_quarter
```

## Matching

A GLED eddy is matched when the centroid of a boundary lies within its
radius. In the southern hemisphere a counter-clockwise eddy is anticyclonic,
so `rotation_sense` is GLED's polarity there.


```python
def distance_m(*, lon_a, lat_a, lon_b, lat_b):
    lat_mid = np.deg2rad(0.5 * (lat_a + lat_b))
    dx = 6_371_000.0 * np.cos(lat_mid) * np.deg2rad(lon_b - lon_a)
    dy = 6_371_000.0 * np.deg2rad(lat_b - lat_a)
    return np.hypot(dx, dy)


def match_to_gled(eddies, *, gled):
    """One row per GLED eddy with the nearest boundary, distances and radii in km."""
    distance = distance_m(
        lon_a=gled["lon"],
        lat_a=gled["lat"],
        lon_b=eddies["centroid_lon"],
        lat_b=eddies["centroid_lat"],
    )
    nearest = eddies.isel(eddy=distance.argmin("eddy"))
    return pd.DataFrame(
        {
            "gled_lon": gled["lon"].values,
            "gled_lat": gled["lat"].values,
            "gled_radius_km": gled["radius_m"].values / 1000,
            "distance_km": distance.min("eddy").values / 1000,
            "radius_km": nearest["radius_m"].values / 1000,
            "stretch": nearest["stretch"].values,
            "matched": (distance.min("eddy") <= gled["radius_m"]).values,
            "same_polarity": (
                (nearest["rotation_sense"] == gled["polarity"]).values
                if "rotation_sense" in nearest
                else np.nan
            ),
        },
        index=gled["gled"].values,
    )
```

Each GLED eddy against the nearest sweep-A boundary at 1/4 degree.

```python
match_to_gled(eddies_a_quarter, gled=gled_30).round(2)
```

The same at 1/8 degree.

```python
match_to_gled(eddies_a_eighth, gled=gled_30).round(2)
```

## The two resolutions side by side

The radius ratio is the boundary's equivalent radius over the GLED radius. The
polarity agreement is over the matched GLED eddies, for sweep B, whose
boundaries carry a rotation sense.


```python
def summary(eddies, *, gled):
    rows = match_to_gled(eddies, gled=gled)
    matched = rows[rows["matched"]]
    distance = distance_m(
        lon_a=gled["lon"],
        lat_a=gled["lat"],
        lon_b=eddies["centroid_lon"],
        lat_b=eddies["centroid_lat"],
    )
    return {
        "boundaries": eddies.sizes["eddy"],
        "boundaries inside a GLED eddy": int(
            (distance <= gled["radius_m"]).any("gled").sum()
        ),
        "GLED eddies matched": len(matched),
        "radius ratio, median": float(
            (matched["radius_km"] / matched["gled_radius_km"]).median()
        ),
        "same polarity": matched["same_polarity"].sum(min_count=1),
    }
```

```python
pd.DataFrame(
    {
        "A, 1/8 degree": summary(eddies_a_eighth, gled=gled_30),
        "A, 1/4 degree": summary(eddies_a_quarter, gled=gled_30),
        "B, 1/8 degree": summary(eddies_b_eighth, gled=gled_30),
        "B, 1/4 degree": summary(eddies_b_quarter, gled=gled_30),
    }
).round(3)
```

## GLED and the closed shear lines

The GLED eddies as circles at their radius, blue for cyclonic and red for
anticyclonic, with a thick outline where the 90-day product carries the eddy
too. A 90-day eddy is coherent over three times the window here, so its
outline marks persistence and not a 30-day boundary. The boundaries of sweep A
are drawn solid and those of sweep B dashed, over the eigenvalue ratio the
sweep-A centres were picked from.


```python
def plot_against_gled(eddies_a, eddies_b):
    angle = np.linspace(0.0, 2 * np.pi, 200)
    _, ax = plt.subplots(figsize=(11, 8))
    np.log10(eddies_a["cg_anisotropy"]).plot.pcolormesh(
        x="lon_grid", y="lat_grid", ax=ax, cmap="Greys"
    )
    for records, width in ((gled_30, 1.2), (gled_90, 3.0)):
        for e in range(records.sizes["gled"]):
            eddy = records.isel(gled=e)
            radius_deg = float(eddy["radius_m"]) / 111_195.0
            ax.plot(
                float(eddy["lon"])
                + radius_deg * np.cos(angle) / np.cos(np.deg2rad(float(eddy["lat"]))),
                float(eddy["lat"]) + radius_deg * np.sin(angle),
                color="tab:red" if int(eddy["polarity"]) > 0 else "tab:blue",
                lw=width,
            )
    ax.plot(eddies_a["lon"].T, eddies_a["lat"].T, color="tab:green", lw=2.0)
    ax.plot(eddies_b["lon"].T, eddies_b["lat"].T, color="tab:green", lw=2.0, ls="--")
    plt.show()
```

```python
# 1/8 degree
plot_against_gled(eddies_a_eighth, eddies_b_eighth)
```

```python
# 1/4 degree
plot_against_gled(eddies_a_quarter, eddies_b_quarter)
```

## Outcome

The 30-day GLED product places 23 eddies in the box on 1 June 2018. Advection
lost 23401 of the 200901 particles at 1/8 degree and 23319 at 1/4 degree.

Sweep A returns 11 boundaries at each resolution. At 1/8 degree 7 of them lie
inside a GLED eddy and 7 GLED eddies are matched. At 1/4 degree it is 8 and 8.
Sweep B, from the GLED centres, returns 9 boundaries at each resolution, all
inside a GLED eddy, so 9 of the 23 GLED eddies have a closed shear line at
their centre. Over the first day every one of those 9 turns in the sense
GLED's polarity gives, at both resolutions.

The median boundary radius over the GLED radius is 0.539 and 0.592 for
sweep A at 1/8 and 1/4 degree, and 0.551 and 0.555 for sweep B.
