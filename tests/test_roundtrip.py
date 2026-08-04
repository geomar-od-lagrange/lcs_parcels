"""Emit / ingest round-trip: Seed.to_parcels_pset <-> Seed.pset_to_flowmap.

The seed is time-free; the window enters only at ingest as ``t0`` (release
time) and ``t1`` (end time), from which the signed ``T = t1 - t0`` is derived.
``FlowMap.to_seed`` is the lossless inverse that drops the advected positions
and the time coords, yielding a time-free seed again.
"""

import numpy as np
import pytest

from lcs_parcels import AuxiliarySeed, NeighborSeed

# Release time and integration end time supplied at ingest; the signed window
# T = END_TIME - RELEASE_TIME is derived.
RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")


def test_neighbor_pset_length(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    expected = lon_axis.size * lat_axis.size
    assert isinstance(lon, list) and isinstance(lat, list)
    assert len(lon) == expected
    assert len(lat) == expected


def test_auxiliary_pset_length(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    ds = seed.ds
    expected = ds.sizes["i"] * ds.sizes["j"] * ds.sizes["displacement"]
    assert len(lon) == expected
    assert len(lat) == expected


def test_neighbor_roundtrip_identity(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    # Reattach the SAME positions (identity flow map): ingest must reproduce the
    # seed positions exactly, pinning the lossless stack -> unstack inverse.
    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)

    # Advected positions land in lon/lat; with identity input they equal the
    # reference seed positions lon_0/lat_0, which are carried through unchanged.
    assert np.allclose(fm.ds["lon"], seed.ds["lon_0"])
    assert np.allclose(fm.ds["lat"], seed.ds["lat_0"])
    assert np.allclose(fm.ds["lon_0"], seed.ds["lon_0"])
    assert np.allclose(fm.ds["lat_0"], seed.ds["lat_0"])

    # t0 (release time) recorded and signed window T = t1 - t0 derived and stored.
    assert fm.ds["t0"] == RELEASE_TIME
    assert fm.ds["T"] == (END_TIME - RELEASE_TIME)


def test_auxiliary_emit_ingest_emit_lossless(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)

    # to_seed drops the advected lon/lat and the time coords, yielding a
    # time-free seed; re-emitting it reproduces the same flat particle set
    # (to_parcels_pset and pset_to_flowmap are lossless inverses).
    reseed = fm.to_seed()
    lon_again, lat_again = reseed.to_parcels_pset()
    assert np.allclose(lon_again, lon)
    assert np.allclose(lat_again, lat)

    # The re-derived seed is time-free and carries no advected positions.
    assert "t0" not in reseed.ds.coords
    assert "T" not in reseed.ds.coords
    assert "lon" not in reseed.ds.variables
    assert "lat" not in reseed.ds.variables

    # The ingested arm positions carry the `displacement` dim on top of (i, j).
    assert "displacement" in fm.ds["lon"].dims
    assert fm.ds["t0"] == RELEASE_TIME
    assert fm.ds["T"] == (END_TIME - RELEASE_TIME)


def test_roundtrip_preserves_grid_dims(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)

    # Ingest unstacks back to the logical (i, j) grid.
    assert fm.ds.sizes["i"] == lon_axis.size
    assert fm.ds.sizes["j"] == lat_axis.size
    assert "particle" not in fm.ds.dims


def test_zero_window_raises(lon_axis, lat_axis):
    """Ingesting with t1 == t0 (zero window) must raise -- FTLE's 1/|T| would
    otherwise divide by zero."""
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    with pytest.raises(ValueError):
        seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=RELEASE_TIME)
