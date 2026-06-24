"""Shared fixtures and helpers for the lcs_parcels test suite.

All helpers use only the high-level, label-based xarray API (``.isel`` /
``.sel`` / named dims), never positional numpy-style indexing on xarray
objects, per AGENTS.md.
"""

import numpy as np
import pytest
import xarray as xr


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
