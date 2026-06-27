"""Shared fixtures and helpers for the lcs_parcels test suite.

All helpers use only the high-level, label-based xarray API (``.isel`` /
``.sel`` / named dims), never positional numpy-style indexing on xarray
objects, per AGENTS.md.
"""

import numpy as np
import pytest


# --- synthetic axis inputs -------------------------------------------------

NI = 4
NJ = 5


@pytest.fixture
def lon_axis():
    """1D longitude axis of length NI (degrees east)."""
    return np.linspace(-2.0, 1.0, NI)


@pytest.fixture
def lat_axis():
    """1D latitude axis of length NJ (degrees north)."""
    return np.linspace(10.0, 14.0, NJ)


# --- linear-flow-map helpers ----------------------------------------------


def apply_linear_map_to_pset(lon, lat, M, origin):
    """Advect a flat (lon, lat) particle set through a constant linear map.

    The map acts in a local meters tangent frame:
    ``displacement_out = M @ displacement_in`` where ``displacement_in`` is the
    seed position measured (in meters) from ``origin = (lon0, lat0)``.  The
    advected positions are returned as flat lon/lat lists in the SAME order as
    the input, mimicking what a Parcels integration would hand back.

    A constant ``M`` makes the deformation gradient exactly ``M`` everywhere,
    independent of the meters<->degrees conventions, because both the input and
    output separations are transformed by the same (locally affine) frame.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon0, lat0 = origin

    R = 6_371_000.0
    deg = np.pi / 180.0
    coslat = np.cos(lat0 * deg)

    # seed separations from origin, in meters (flat-tangent convention)
    dx = R * coslat * (lon - lon0) * deg
    dy = R * (lat - lat0) * deg

    out = np.stack([dx, dy], axis=0)  # shape (2, N)
    out = M @ out  # apply constant linear map
    dxo, dyo = out[0], out[1]

    lon_out = lon0 + dxo / (R * coslat * deg)
    lat_out = lat0 + dyo / (R * deg)
    return list(lon_out), list(lat_out)


def advected_flowmap(seed_cls, lon_axis, lat_axis, M, t0, t1):
    """Build an advected ``FlowMap`` from a constant linear map ``M``.

    Seed ``seed_cls`` from the 1-D axes (the seed is TIME-FREE, so ``from_axes``
    takes no ``t0``), emit its particle set, advect every flat position through
    the constant linear map ``M`` about the seed CENTROID, then ingest the
    advected positions via ``seed.pset_to_flowmap`` -- where the window enters as
    ``t0`` (release time) and ``t1`` (end time), from which the signed
    ``T = t1 - t0`` is derived -- and return the resulting ``FlowMap``.

    The advection ``origin`` is the seed centroid
    ``(float(seed.ds['lon_0'].mean()), float(seed.ds['lat_0'].mean()))`` so it
    coincides with the implementation's single reference longitude/latitude (the
    local-tangent meters frame is anchored at the same point). Because both the
    seed separations and the advected separations are then measured in that one
    frame, the deformation gradient recovers ``M`` exactly -- the off-diagonal
    terms of ``M`` survive only when the read-back frame matches the map frame.
    """
    seed = seed_cls.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()
    origin = (float(seed.ds["lon_0"].mean()), float(seed.ds["lat_0"].mean()))
    lon_out, lat_out = apply_linear_map_to_pset(lon, lat, M, origin)
    return seed.pset_to_flowmap(lon_out, lat_out, t0=t0, t1=t1)
