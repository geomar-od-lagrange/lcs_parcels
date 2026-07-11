"""Shape / dim / coord contracts for the Seed and FlowMap families.

A ``Seed`` is time-free: ``from_axes`` takes no ``t0``, and the dataset is
all-coordinates (no ``t0``/``T``, no advected ``lon``/``lat``). Time and the
advected positions enter at ingest, ``seed.pset_to_flowmap(lon, lat, *, t0,
t1)``, which returns a ``FlowMap`` carrying scalar ``t0``/``T`` coords and
``lon``/``lat`` data vars.
"""

import numpy as np

from lcs_parcels import AuxiliarySeed, NeighborSeed
from lcs_parcels.grids import _lonlat_to_meters

# Release/end times supplied only at ingest (the seed itself is time-free).
T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")


# --- seed shape ------------------------------------------------------------


def test_neighbor_seed_from_axes_dims(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    # curvilinear reference lon_0/lat_0 on logical dims (i, j)
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0

    # NeighborSeed carries no displacement stencil dim
    assert "displacement" not in ds.dims


def test_neighbor_seed_lon_lat_values(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    # 2D reference lon_0/lat_0 broadcast from the 1D axes (label-based access).
    lon_iv = lon_axis[2]
    lat_jv = lat_axis[3]
    assert float(ds["lon_0"].isel(i=2, j=3)) == lon_iv
    assert float(ds["lat_0"].isel(i=2, j=3)) == lat_jv
    # lon_0 varies along i and is constant along j; lat_0 the reverse.
    assert float(ds["lon_0"].isel(i=2, j=0)) == float(ds["lon_0"].isel(i=2, j=4))
    assert float(ds["lat_0"].isel(i=0, j=3)) == float(ds["lat_0"].isel(i=3, j=3))


def test_auxiliary_seed_from_axes_dims(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # AuxiliarySeed adds a four-arm stencil on the `displacement` dim (no centre,
    # no diagonals).
    assert ds.sizes["displacement"] == 4
    assert list(ds["displacement"].values) == ["east", "north", "west", "south"]

    # The reference positions x_0 are the explicit per-arm positions: lon_0/lat_0
    # carry the displacement dim.
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat_0"].dims) == {"i", "j", "displacement"}

    # The diagnostic grid-point centres are kept on (i, j).
    assert set(ds["lon_c"].dims) == {"i", "j"}
    assert set(ds["lat_c"].dims) == {"i", "j"}

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference + centre coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ns = NeighborSeed.from_axes(lon_axis, lat_axis)
    aus = AuxiliarySeed.from_axes(lon_axis, lat_axis)

    # NeighborSeed has no auxiliary stencil: no displacement dim, no centres, and
    # its reference positions are the grid points themselves on (i, j).
    assert "displacement" not in ns.ds.dims
    assert "lon_c" not in ns.ds.coords and "lat_c" not in ns.ds.coords
    assert set(ns.ds["lon_0"].dims) == {"i", "j"}

    # AuxiliarySeed carries the four-arm displacement stencil and the separate
    # diagnostic centres.
    assert aus.ds.sizes["displacement"] == 4
    assert set(aus.ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert "lon_c" in aus.ds.coords and "lat_c" in aus.ds.coords


def test_auxiliary_arm_geometry_and_separation(lon_axis, lat_axis):
    """AuxiliarySeed places its four arms at exactly +/- ``aux_separation_m``.

    Read the arm positions back into the meters frame
    (:func:`_lonlat_to_meters`) with a non-default separation and assert: the
    east-west and north-south spans are each exactly ``2s``, opposing arms share
    the centre on the other axis (no cross-offset), and the geographic direction
    of each labelled arm is correct.
    """
    s = 2_500.0  # non-default aux_separation_m
    seed = AuxiliarySeed.from_axes(lon_axis, lat_axis, aux_separation_m=s)
    lon0, lat0 = seed.ds["lon_0"], seed.ds["lat_0"]
    lon_c, lat_c = seed.ds["lon_c"], seed.ds["lat_c"]
    lon_ref, lat_ref = float(lon_c.mean()), float(lat_c.mean())
    x, y = _lonlat_to_meters(lon0, lat0, lon_ref, lat_ref)  # dims (i, j, displacement)

    ew = x.sel(displacement="east") - x.sel(displacement="west")
    ns = y.sel(displacement="north") - y.sel(displacement="south")
    assert float(abs(ew - 2 * s).max()) < 1e-6  # east-west span == 2s
    assert float(abs(ns - 2 * s).max()) < 1e-6  # north-south span == 2s

    # opposing arms share the centre on the OTHER axis (no cross-offset)
    assert float(abs(y.sel(displacement="east") - y.sel(displacement="west")).max()) < 1e-6
    assert float(abs(x.sel(displacement="north") - x.sel(displacement="south")).max()) < 1e-6

    # geographic direction is correct
    assert float((lon0.sel(displacement="east") - lon0.sel(displacement="west")).min()) > 0
    assert float((lat0.sel(displacement="north") - lat0.sel(displacement="south")).min()) > 0


# --- flow-map shape --------------------------------------------------------


def test_neighbor_flowmap_shape(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t0=T0, t1=T1)
    ds = fm.ds

    # The flow map adds the advected positions as its only data vars, on (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j"}
    assert set(ds["lat"].dims) == {"i", "j"}

    # Reference positions are carried through unchanged on (i, j).
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}

    # Scalar t0 (release time) and signed window T = t1 - t0 land as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == T0
    assert ds["T"] == (T1 - T0)


def test_auxiliary_flowmap_shape(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon_axis, lat_axis)
    fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t0=T0, t1=T1)
    ds = fm.ds

    # Advected arm positions carry the displacement dim on top of (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat"].dims) == {"i", "j", "displacement"}

    # Reference arms + centres carried through unchanged.
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lon_c"].dims) == {"i", "j"}
    assert set(ds["lat_c"].dims) == {"i", "j"}

    # Scalar t0 / signed T as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == T0
    assert ds["T"] == (T1 - T0)
