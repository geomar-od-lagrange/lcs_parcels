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


# --- flow-map helpers ------------------------------------------------------


def apply_map_to_pset(lon, lat, f, origin):
    """Advect a flat (lon, lat) particle set through a general map ``f``.

    The map acts in a local meters tangent frame: the callable
    ``f(dx, dy) -> (dx_out, dy_out)`` transforms the seed separations
    ``(dx, dy)`` -- measured (in meters) from ``origin = (lon0, lat0)`` -- into
    advected separations, which are converted back to lon/lat. The advected
    positions are returned as flat lon/lat lists in the SAME order as the input,
    mimicking what a Parcels integration would hand back.

    ``f`` need not be linear: a spatially-varying (e.g. quadratic) ``f`` makes the
    deformation gradient vary across the grid, so it exercises the auxiliary
    stencil's per-point differencing (unlike a constant linear map, whose metric
    scale cancels top-and-bottom in the meters/meters ratio).
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

    dxo, dyo = f(dx, dy)  # apply the (possibly nonlinear) map

    lon_out = lon0 + dxo / (R * coslat * deg)
    lat_out = lat0 + dyo / (R * deg)
    return list(lon_out), list(lat_out)


def apply_linear_map_to_pset(lon, lat, M, origin):
    """Advect a flat (lon, lat) particle set through a constant linear map.

    Thin wrapper over :func:`apply_map_to_pset` that builds the linear callable
    ``displacement_out = M @ displacement_in`` from the matrix ``M`` (the map acts
    on the seed separation measured in meters from ``origin = (lon0, lat0)``).

    A constant ``M`` makes the deformation gradient exactly ``M`` everywhere,
    independent of the meters<->degrees conventions, because both the input and
    output separations are transformed by the same (locally affine) frame.
    """

    def linear(dx, dy):
        out = M @ np.stack([dx, dy], axis=0)
        return out[0], out[1]

    return apply_map_to_pset(lon, lat, linear, origin)


def advected_flowmap_f(seed_cls, lon_axis, lat_axis, f, t0, t1):
    """Build an advected ``FlowMap`` from a general meters-frame map ``f``.

    Seed ``seed_cls`` from the 1-D axes (the seed is TIME-FREE, so ``from_axes``
    takes no ``t0``), emit its particle set, advect every flat position through
    the callable ``f(dx, dy) -> (dx_out, dy_out)`` about the seed CENTROID, then
    ingest the advected positions via ``seed.pset_to_flowmap`` -- where the window
    enters as ``t0`` (release time) and ``t1`` (end time), from which the signed
    ``T = t1 - t0`` is derived -- and return the resulting ``FlowMap``.

    The advection ``origin`` is the seed centroid
    ``(float(seed.ds['lon_0'].mean()), float(seed.ds['lat_0'].mean()))`` so it
    coincides with the implementation's single reference longitude/latitude (the
    local-tangent meters frame is anchored at the same point). Because both the
    seed separations and the advected separations are then measured in that one
    frame, the recovered deformation gradient is the Jacobian of ``f`` -- for a
    spatially-varying ``f`` this genuinely varies from grid point to grid point.
    """
    seed = seed_cls.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()
    origin = (float(seed.ds["lon_0"].mean()), float(seed.ds["lat_0"].mean()))
    lon_out, lat_out = apply_map_to_pset(lon, lat, f, origin)
    return seed.pset_to_flowmap(lon_out, lat_out, t0=t0, t1=t1)


def advected_flowmap(seed_cls, lon_axis, lat_axis, M, t0, t1):
    """Build an advected ``FlowMap`` from a constant linear map ``M``.

    Thin wrapper over :func:`advected_flowmap_f` that builds the linear callable
    ``displacement_out = M @ displacement_in`` from the matrix ``M``; see
    :func:`advected_flowmap_f` for the seed/advect/ingest flow and the anchoring
    of the advection origin at the seed centroid.

    Because both the seed separations and the advected separations are measured in
    that one frame, the deformation gradient recovers ``M`` exactly -- the
    off-diagonal terms of ``M`` survive only when the read-back frame matches the
    map frame.
    """

    def linear(dx, dy):
        out = M @ np.stack([dx, dy], axis=0)
        return out[0], out[1]

    return advected_flowmap_f(seed_cls, lon_axis, lat_axis, linear, t0, t1)
