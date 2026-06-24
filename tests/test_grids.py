"""Seeding (`from_axes`) shape / dim / coord contracts."""

import numpy as np

from lcs_parcels import AuxiliaryGrid, NeighborGrid


def test_neighbor_grid_from_axes_dims(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis)
    ds = g.ds

    # curvilinear lon/lat on logical dims (i, j)
    assert set(ds["lon"].dims) == {"i", "j"}
    assert set(ds["lat"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # NeighborGrid carries no displacement stencil dims
    assert "di" not in ds.dims
    assert "dj" not in ds.dims


def test_neighbor_grid_lon_lat_values(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis)
    ds = g.ds

    # 2D lon/lat broadcast from the 1D axes (label-based access only).
    lon_iv = lon_axis[2]
    lat_jv = lat_axis[3]
    assert float(ds["lon"].isel(i=2, j=3)) == lon_iv
    assert float(ds["lat"].isel(i=2, j=3)) == lat_jv
    # lon varies along i and is constant along j; lat the reverse.
    assert float(ds["lon"].isel(i=2, j=0)) == float(ds["lon"].isel(i=2, j=4))
    assert float(ds["lat"].isel(i=0, j=3)) == float(ds["lat"].isel(i=3, j=3))


def test_auxiliary_grid_from_axes_dims(lon_axis, lat_axis):
    g = AuxiliaryGrid.from_axes(lon_axis, lat_axis)
    ds = g.ds

    assert set(ds["lon"].dims) == {"i", "j"}
    assert set(ds["lat"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # AuxiliaryGrid adds a per-point displacement stencil on (di, dj).
    assert "di" in ds.dims
    assert "dj" in ds.dims
    assert ds.sizes["di"] >= 2
    assert ds.sizes["dj"] >= 2

    # dx, dy displacement fields live on (di, dj) and are in meters.
    assert set(ds["dx"].dims) == {"di", "dj"}
    assert set(ds["dy"].dims) == {"di", "dj"}


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ng = NeighborGrid.from_axes(lon_axis, lat_axis)
    ag = AuxiliaryGrid.from_axes(lon_axis, lat_axis)

    assert "dx" not in ng.ds and "dy" not in ng.ds
    assert "dx" in ag.ds and "dy" in ag.ds
