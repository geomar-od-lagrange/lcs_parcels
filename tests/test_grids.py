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

    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size
    assert ds["t0"] == T0

    # AuxiliaryGrid adds a fixed four-arm stencil on a single `displacement` dim
    # (no center, no diagonals; the shape is enforced, not arbitrary).
    assert ds.sizes["displacement"] == 4
    assert list(ds["displacement"].values) == ["east", "north", "west", "south"]

    # The reference release positions x_0 are the explicit per-arm positions:
    # lon_0/lat_0 carry the displacement dim (this is what to_parcels_pset emits,
    # so the dataset is self-sufficient -- no metric needed to recover them).
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat_0"].dims) == {"i", "j", "displacement"}

    # The grid-point centres on which diagnostics are reported are kept explicitly
    # on (i, j) (needed for downstream LCS work).
    assert set(ds["lon_c"].dims) == {"i", "j"}
    assert set(ds["lat_c"].dims) == {"i", "j"}

    # A seed is the identity sample: advected arms equal the reference arms, the
    # centre is the mean of its four arms, and the signed window is zero.
    xr.testing.assert_allclose(ds["lon"], ds["lon_0"])
    xr.testing.assert_allclose(ds["lat"], ds["lat_0"])
    xr.testing.assert_allclose(ds["lon_c"], ds["lon_0"].mean("displacement"))
    xr.testing.assert_allclose(ds["lat_c"], ds["lat_0"].mean("displacement"))
    assert ds["T"] == (T0 - T0)


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ng = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    ag = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=T0)

    # NeighborGrid has no auxiliary stencil: no displacement dim, no centres, and
    # its reference positions are the grid points themselves on (i, j).
    assert "displacement" not in ng.ds.dims
    assert "lon_c" not in ng.ds.coords and "lat_c" not in ng.ds.coords
    assert set(ng.ds["lon_0"].dims) == {"i", "j"}

    # AuxiliaryGrid carries the four-arm displacement stencil (in its explicit
    # reference positions) and the separate diagnostic centres.
    assert ag.ds.sizes["displacement"] == 4
    assert set(ag.ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert "lon_c" in ag.ds.coords and "lat_c" in ag.ds.coords
