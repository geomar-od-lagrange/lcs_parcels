"""Emit / ingest round-trip: to_parcels_pset <-> from_parcels_pset_lon_lat."""

import numpy as np
import xarray as xr

from lcs_parcels import AuxiliaryGrid, NeighborGrid

# Release time and signed integration window for ingest (see plans/timing-design.md).
T0 = np.datetime64("2020-01-01")
T = np.timedelta64(1, "D")


def test_neighbor_pset_length(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis)
    lon, lat = g.to_parcels_pset()

    expected = lon_axis.size * lat_axis.size
    assert isinstance(lon, list) and isinstance(lat, list)
    assert len(lon) == expected
    assert len(lat) == expected


def test_auxiliary_pset_length(lon_axis, lat_axis):
    g = AuxiliaryGrid.from_axes(lon_axis, lat_axis)
    lon, lat = g.to_parcels_pset()

    ds = g.ds
    expected = ds.sizes["i"] * ds.sizes["j"] * ds.sizes["di"] * ds.sizes["dj"]
    assert len(lon) == expected
    assert len(lat) == expected


def test_neighbor_roundtrip_identity(lon_axis, lat_axis):
    seed = NeighborGrid.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()

    # Reattach the SAME positions: ingest must reproduce the seed grid exactly.
    advected = NeighborGrid.from_parcels_pset_lon_lat(seed, lon, lat, t0=T0, T=T)

    xr.testing.assert_allclose(advected.ds["lon"], seed.ds["lon"])
    xr.testing.assert_allclose(advected.ds["lat"], seed.ds["lat"])


def test_auxiliary_roundtrip_identity(lon_axis, lat_axis):
    seed = AuxiliaryGrid.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()

    advected = AuxiliaryGrid.from_parcels_pset_lon_lat(seed, lon, lat, t0=T0, T=T)

    xr.testing.assert_allclose(advected.ds["lon"], seed.ds["lon"])
    xr.testing.assert_allclose(advected.ds["lat"], seed.ds["lat"])


def test_roundtrip_preserves_grid_dims(lon_axis, lat_axis):
    seed = NeighborGrid.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()
    advected = NeighborGrid.from_parcels_pset_lon_lat(seed, lon, lat, t0=T0, T=T)

    # Ingest unstacks back to the logical (i, j) grid.
    assert advected.ds.sizes["i"] == lon_axis.size
    assert advected.ds.sizes["j"] == lat_axis.size
    assert "particle" not in advected.ds.dims
