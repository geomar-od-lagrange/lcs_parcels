"""Shared fixtures and helpers for the lcs_parcels test suite."""

import numpy as np
import pytest
import xarray as xr

from lcs_parcels import UnstructuredAuxiliarySeedGrid

EARTH_RADIUS_M = 6_371_000.0
DEG = np.pi / 180.0

# --- synthetic axis inputs -------------------------------------------------

NI = 4
NJ = 5

#: ``region -> (lon_axis, lat_axis)``, all of shape ``(NI,)`` / ``(NJ,)``.
#:
#: ``reference`` is a mid-latitude band away from the branch cut. ``antimeridian``
#: runs up to 180 so that advected longitudes cross it. The axis itself stays
#: monotonic, since a wrapped axis (177, 179, -179) is not a valid interpolation
#: axis. ``high_latitude`` spans a band, over which ``cos(lat)`` changes by a
#: factor of about 1.5.
REGIONS = {
    "reference": (np.linspace(-2.0, 1.0, NI), np.linspace(10.0, 14.0, NJ)),
    "antimeridian": (np.linspace(177.0, 180.0, NI), np.linspace(10.0, 14.0, NJ)),
    "high_latitude": (np.linspace(-2.0, 1.0, NI), np.linspace(68.0, 76.0, NJ)),
}


@pytest.fixture(params=list(REGIONS), ids=list(REGIONS))
def region(request):
    """Return the (lon, lat) axis pair every axis-taking test is run over."""
    return REGIONS[request.param]


@pytest.fixture
def lon_axis(region):
    """Return the 1-D longitude axis of length NI (degrees east)."""
    return region[0]


@pytest.fixture
def lat_axis(region):
    """Return the 1-D latitude axis of length NJ (degrees north)."""
    return region[1]


# --- flow-map helpers ------------------------------------------------------


def apply_map_to_pset(lon, lat, f, origin):
    """Advect a flat (lon, lat) particle set through a general map ``f``.

    The map acts in a local meters tangent frame. ``f(dx, dy) -> (dx_out,
    dy_out)`` transforms the seed separations ``(dx, dy)``, measured in meters
    from ``origin = (lon_0, lat_0)``, into advected separations, converted back
    to lon/lat. The advected positions are returned as flat lon/lat lists in the
    same order as the input. ``f`` need not be linear.
    """
    lon_out, lat_out = apply_map_to_lonlat(lon, lat, f, origin)
    return list(lon_out), list(lat_out)


def apply_map_to_lonlat(lon, lat, f, origin):
    """Apply the same map as ``apply_map_to_pset``, on arrays of any shape, and
    return arrays.

    Kept separate so the analytic expectations below can push the *grid points*
    through the same map the particles went through, without flattening.

    The advected longitudes come back wrapped to ``[-180, 180)``, which is what
    Parcels and CMEMS hand back. The ``antimeridian`` region puts the advected
    positions on the other side of the branch cut from the seed. Any test that
    takes ``lon_axis``/``lat_axis`` differences therefore takes them across it.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_0, lat_0 = origin
    coslat = np.cos(lat_0 * DEG)

    # seed separations from origin, in meters (flat-tangent convention)
    dx = EARTH_RADIUS_M * coslat * (lon - lon_0) * DEG
    dy = EARTH_RADIUS_M * (lat - lat_0) * DEG

    dxo, dyo = f(dx, dy)  # apply the (possibly nonlinear) map

    lon_out = lon_0 + dxo / (EARTH_RADIUS_M * coslat * DEG)
    lat_out = lat_0 + dyo / (EARTH_RADIUS_M * DEG)
    return (lon_out + 180.0) % 360.0 - 180.0, lat_out


def local_frame_gradient(flat_jacobian, *, lat_grid, lat_advected, lat_origin):
    """Compute the deformation gradient the package measures, from a flat-frame
    Jacobian.

    ``apply_map_to_pset`` acts in the *one* tangent frame at ``origin``, so its
    Jacobian ``flat_jacobian`` is expressed in that frame. The package measures
    each separation in the local east/north frame of the pair it connects: the
    denominator at the release latitude ``lat_grid``, the numerator at the
    arrival latitude ``lat_advected``. Converting between the two is a diagonal
    rescaling on each side::

        grad F = diag(cos(Phi) / c, 1) . flat_jacobian . diag(c / cos(phi), 1)

    with ``c = cos(lat_origin)``, ``phi = lat_grid``, and ``Phi = lat_advected``.
    Only the east components are touched, because north is ``R d phi`` in either
    frame.

    On a sphere no map has a constant tangent Jacobian, so this expectation
    varies over the grid even when ``flat_jacobian`` does not. That variation is
    the content of the rescaling, not an artefact of it.

    Parameters
    ----------
    flat_jacobian : xr.DataArray
        The map's Jacobian in the ``origin`` tangent frame, dims ``(row, col)``
        valued ``['x', 'y']``, optionally also varying over the grid points.
    lat_grid, lat_advected : xr.DataArray
        Release and arrival latitudes of the grid points (degrees), on the
        grid points' own dims.
    lat_origin : float
        Latitude (degrees) of the tangent frame the map was applied in.

    Returns
    -------
    xr.DataArray
        ``grad F`` with the grid points' dims plus ``row``/``col``, valued
        ``['x', 'y']``.
    """
    c = np.cos(lat_origin * DEG)
    one = xr.ones_like(lat_grid)
    left = xr.concat([np.cos(lat_advected * DEG) / c, one], dim="row")
    right = xr.concat([c / np.cos(lat_grid * DEG), one], dim="col")
    scaled = left * flat_jacobian * right
    return scaled.assign_coords(row=["x", "y"], col=["x", "y"]).transpose(
        *lat_grid.dims, "row", "col"
    )


def analytic_gradient(M, *, flowmap, origin):
    """Compute :func:`local_frame_gradient` for the constant linear map ``M``.

    ``M`` is the flat-frame Jacobian of :func:`apply_linear_map_to_pset`, and the
    grid points are pushed through the same map to get their arrival latitudes.
    """
    release = ["lon_0", "lat_0"]
    lon_grid = flowmap.ds["lon_grid"].drop_vars(release, errors="ignore")
    lat_grid = flowmap.ds["lat_grid"].drop_vars(release, errors="ignore")
    _, lat_advected = apply_map_to_lonlat(
        lon_grid.values, lat_grid.values, _linear(M), origin
    )
    return local_frame_gradient(
        xr.DataArray(M, dims=("row", "col")),
        lat_grid=lat_grid,
        lat_advected=lat_grid.copy(data=lat_advected),
        lat_origin=origin[1],
    )


def seed_origin(flowmap_or_seed):
    """Return the tangent-frame origin the ``advected_flowmap*`` helpers advected
    about."""
    ds = flowmap_or_seed.ds
    return float(ds["lon_0"].mean()), float(ds["lat_0"].mean())


def _linear(M):
    """Wrap ``M`` as a flat-frame callable ``f(dx, dy) -> (dx_out, dy_out)``."""

    def linear(dx, dy):
        out = M @ np.stack([np.ravel(dx), np.ravel(dy)], axis=0)
        return out[0].reshape(np.shape(dx)), out[1].reshape(np.shape(dy))

    return linear


def apply_linear_map_to_pset(lon, lat, M, origin):
    """Advect a flat (lon, lat) particle set through a constant linear map.

    Wrap :func:`apply_map_to_pset` and build the linear callable
    ``displacement_out = M @ displacement_in`` from the matrix ``M``, acting on
    the seed separation in meters from ``origin = (lon_0, lat_0)``. The
    deformation gradient the package then measures is
    :func:`analytic_gradient`, not ``M`` itself.
    """
    return apply_map_to_pset(lon, lat, _linear(M), origin)


def advected_flowmap_f(seed_cls, lon_axis, lat_axis, f, t0, t1):
    """Build an advected ``FlowMap`` from a general meters-frame map ``f``.

    Seed ``seed_cls`` from the 1-D axes, emit its particle set, and advect every
    flat position through ``f(dx, dy) -> (dx_out, dy_out)`` about the seed
    centroid. Then ingest via ``seed.pset_to_flowmap(lon=..., lat=..., t0=...,
    t1=...)`` and return the ``FlowMap``. The advection ``origin`` is the seed
    centroid (:func:`seed_origin`). ``f``'s Jacobian is in that one tangent
    frame, so the deformation gradient the package recovers is that Jacobian
    rescaled into the local frames. See :func:`local_frame_gradient`.
    """
    seed = seed_cls.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    lon_out, lat_out = apply_map_to_pset(lon, lat, f, seed_origin(seed))
    return seed.pset_to_flowmap(lon=lon_out, lat=lat_out, t0=t0, t1=t1)


def scattered_points(lon_axis, lat_axis):
    """Return the outer product of two 1-D axes, as a flat ``(lon, lat)`` point
    pair.

    These are the points a structured grid over the same axes would have, in
    the same order, so one region can be put through either layout.
    """
    lon_grid, lat_grid = xr.broadcast(
        xr.DataArray(np.asarray(lon_axis, dtype=float), dims="i"),
        xr.DataArray(np.asarray(lat_axis, dtype=float), dims="j"),
    )
    return lon_grid.values.ravel(), lat_grid.values.ravel()


def advected_scattered_flowmap(lon_axis, lat_axis, M, t0, t1):
    """Build the same flow map as :func:`advected_flowmap`, with the grid points
    as a set rather than axes.

    Seed an :class:`UnstructuredAuxiliarySeedGrid` through
    :meth:`~lcs_parcels.UnstructuredAuxiliarySeedGrid.from_points` over
    :func:`scattered_points`, so its diagnostics are the structured ones
    flattened.
    """
    lon_points, lat_points = scattered_points(lon_axis, lat_axis)
    seed = UnstructuredAuxiliarySeedGrid.from_points(lon=lon_points, lat=lat_points)
    lon, lat = seed.to_parcels_pset()
    lon_out, lat_out = apply_linear_map_to_pset(lon, lat, M, seed_origin(seed))
    return seed.pset_to_flowmap(lon=lon_out, lat=lat_out, t0=t0, t1=t1)


def advected_flowmap(seed_cls, lon_axis, lat_axis, M, t0, t1):
    """Build an advected ``FlowMap`` from a constant linear map ``M``.

    Wrap :func:`advected_flowmap_f` and build the linear callable
    ``displacement_out = M @ displacement_in`` from the matrix ``M``. See
    :func:`advected_flowmap_f` for the seed/advect/ingest flow. The deformation
    gradient the package recovers is :func:`analytic_gradient`.
    """
    return advected_flowmap_f(seed_cls, lon_axis, lat_axis, _linear(M), t0, t1)
