"""Seeding (`from_axes`) shape / dim / coord contracts."""

import numpy as np
import xarray as xr

from lcs_parcels import AuxiliaryGrid, NeighborGrid

# Release time recorded on the seed grid (see plans/timing-design.md).
T0 = np.datetime64("2020-01-01")


def test_neighbor_grid_from_axes_dims(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    ds = g.ds

    # curvilinear reference lon_0/lat_0 on logical dims (i, j)
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # the grid owns its release time t0
    assert ds["t0"] == T0

    # a seed is the identity sample F_{t0}^{t0} = x_0: the advected lon/lat equal
    # the reference lon_0/lat_0 and the signed window is zero.
    xr.testing.assert_allclose(ds["lon"], ds["lon_0"])
    xr.testing.assert_allclose(ds["lat"], ds["lat_0"])
    assert ds["T"] == (T0 - T0)

    # NeighborGrid carries no displacement stencil dim
    assert "displacement" not in ds.dims


def test_neighbor_grid_lon_lat_values(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    ds = g.ds

    # 2D reference lon_0/lat_0 broadcast from the 1D axes (label-based access).
    lon_iv = lon_axis[2]
    lat_jv = lat_axis[3]
    assert float(ds["lon_0"].isel(i=2, j=3)) == lon_iv
    assert float(ds["lat_0"].isel(i=2, j=3)) == lat_jv
    # lon_0 varies along i and is constant along j; lat_0 the reverse.
    assert float(ds["lon_0"].isel(i=2, j=0)) == float(ds["lon_0"].isel(i=2, j=4))
    assert float(ds["lat_0"].isel(i=0, j=3)) == float(ds["lat_0"].isel(i=3, j=3))


def test_auxiliary_grid_from_axes_dims(lon_axis, lat_axis):
    g = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=T0)
    ds = g.ds

    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size
    assert ds["t0"] == T0

    # AuxiliaryGrid adds a fixed four-arm stencil on a single `displacement` dim
    # (no center, no diagonals; the shape is enforced, not arbitrary).
    assert ds.sizes["displacement"] == 4
    assert list(ds["displacement"].values) == ["east", "north", "west", "south"]

    # dx, dy displacement offsets live on `displacement` and are in meters.
    assert set(ds["dx"].dims) == {"displacement"}
    assert set(ds["dy"].dims) == {"displacement"}


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ng = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    ag = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=T0)

    assert "dx" not in ng.ds and "dy" not in ng.ds
    assert "dx" in ag.ds and "dy" in ag.ds
