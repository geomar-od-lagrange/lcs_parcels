"""Emit / ingest round-trip: to_parcels_pset <-> from_parcels_pset_lon_lat."""

import numpy as np

from lcs_parcels import AuxiliaryGrid, NeighborGrid

# Release time (recorded on the seed) and integration end time supplied at
# ingest; the signed window T = T1 - T0 is derived. See plans/timing-design.md.
T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")


def test_neighbor_pset_length(lon_axis, lat_axis):
    g = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    lon, lat = g.to_parcels_pset()

    expected = lon_axis.size * lat_axis.size
    assert isinstance(lon, list) and isinstance(lat, list)
    assert len(lon) == expected
    assert len(lat) == expected


def test_auxiliary_pset_length(lon_axis, lat_axis):
    g = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=T0)
    lon, lat = g.to_parcels_pset()

    ds = g.ds
    expected = ds.sizes["i"] * ds.sizes["j"] * ds.sizes["displacement"]
    assert len(lon) == expected
    assert len(lat) == expected


def test_neighbor_roundtrip_identity(lon_axis, lat_axis):
    seed = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    lon, lat = seed.to_parcels_pset()

    # Reattach the SAME positions: ingest must reproduce the seed grid exactly.
    # This pins the lossless stack -> unstack inverse; the positions are the
    # identity flow map regardless of t1, so a non-zero window still exercises T.
    advected = NeighborGrid.from_parcels_pset_lon_lat(seed, lon, lat, t1=T1)

    # Advected positions land in lon/lat; with identity input they equal the
    # reference seed positions lon_0/lat_0, which are carried through unchanged.
    # (compare values; the seed and advected grids differ only in the T coord.)
    assert np.allclose(advected.ds["lon"], seed.ds["lon_0"])
    assert np.allclose(advected.ds["lat"], seed.ds["lat_0"])
    assert np.allclose(advected.ds["lon_0"], seed.ds["lon_0"])
    assert np.allclose(advected.ds["lat_0"], seed.ds["lat_0"])

    # t0 carried from the seed; signed window T = t1 - t0 derived and stored.
    assert advected.ds["t0"] == T0
    assert advected.ds["T"] == (T1 - T0)


def test_auxiliary_roundtrip_identity(lon_axis, lat_axis):
    seed = AuxiliaryGrid.from_axes(lon_axis, lat_axis, t0=T0)
    lon, lat = seed.to_parcels_pset()

    advected = AuxiliaryGrid.from_parcels_pset_lon_lat(seed, lon, lat, t1=T1)

    # Re-emitting the ingested grid reproduces the same flat particle set
    # (to_parcels_pset and from_parcels_pset_lon_lat are lossless inverses).
    lon_again, lat_again = advected.to_parcels_pset()
    assert np.allclose(lon_again, lon)
    assert np.allclose(lat_again, lat)

    # The ingested arm positions carry the `displacement` dim on top of (i, j).
    assert "displacement" in advected.ds["lon"].dims
    assert advected.ds["t0"] == T0
    assert advected.ds["T"] == (T1 - T0)


def test_roundtrip_preserves_grid_dims(lon_axis, lat_axis):
    seed = NeighborGrid.from_axes(lon_axis, lat_axis, t0=T0)
    lon, lat = seed.to_parcels_pset()
    advected = NeighborGrid.from_parcels_pset_lon_lat(seed, lon, lat, t1=T1)

    # Ingest unstacks back to the logical (i, j) grid.
    assert advected.ds.sizes["i"] == lon_axis.size
    assert advected.ds.sizes["j"] == lat_axis.size
    assert "particle" not in advected.ds.dims
