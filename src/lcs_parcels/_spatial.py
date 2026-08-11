"""Queries over the diagnostic grid points, taken on the sphere.

This module provides two interpolator builders, one reading the grid points as
two monotonic axes and one as a point set with no structure, and a
neighbourhood maximum.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.spatial import KDTree

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in metres. All distances are taken on a sphere of this radius."""

_M_PER_DEG = EARTH_RADIUS_M * (np.pi / 180.0)
"""Metres per degree of latitude, and of longitude at the equator."""


def _sphere_xyz(lon: xr.DataArray, lat: xr.DataArray) -> np.ndarray:
    """Cartesian coordinates of the points on the sphere, one row each.

    The embedding has no branch cut and no pole, so a distance taken in it is
    the chord of the great circle whatever the longitude convention.
    """
    lon_rad = np.deg2rad(np.ravel(lon.values))
    lat_rad = np.deg2rad(np.ravel(lat.values))
    cos_lat = np.cos(lat_rad)
    return EARTH_RADIUS_M * np.column_stack(
        [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)]
    )


def _flat_values(field: xr.DataArray, dims: tuple[str, ...]) -> np.ndarray:
    """``field`` with ``dims`` collapsed into one leading axis of samples."""
    values = field.transpose(*dims, ...).values
    return values.reshape(-1, *values.shape[len(dims) :])


def rectilinear_interpolator(field: xr.DataArray, *, lon_grid, lat_grid):
    """Interpolator for ``field`` on the axis-aligned ``lon_grid``/``lat_grid`` axes.

    The axes are read off row ``j = 0`` and column ``i = 0``, so the grid has to
    be rectilinear and the ``lon_grid`` axis monotonic.

    Parameters
    ----------
    field : xr.DataArray
        A quantity sampled at the grid points, on dims ``(i, j)`` plus any
        component dims.
    lon_grid, lat_grid : xr.DataArray
        The grid points, on ``(i, j)``.

    Returns
    -------
    callable
        Takes an ``(n, 2)`` array of ``(lon, lat)`` and returns the interpolated
        values with a leading axis of length ``n``, NaN outside the axes.
    """
    return RegularGridInterpolator(
        (lon_grid.isel(j=0, drop=True).values, lat_grid.isel(i=0, drop=True).values),
        field.transpose("i", "j", ...).values,
        bounds_error=False,
        fill_value=np.nan,
    )


def scattered_interpolator(field: xr.DataArray, *, lon_grid, lat_grid):
    """Interpolator for ``field`` sampled at grid points of any layout.

    It is linear on the Delaunay triangulation of the points, taken in degrees.
    The points therefore have to sit on one longitude branch, and three of them
    may not be collinear.

    Parameters
    ----------
    field : xr.DataArray
        A quantity sampled at the grid points, on the grid points' own dims plus
        any component dims.
    lon_grid, lat_grid : xr.DataArray
        The grid points, on any dims.

    Returns
    -------
    callable
        Takes an ``(n, 2)`` array of ``(lon, lat)`` and returns the interpolated
        values with a leading axis of length ``n``, NaN outside the convex hull
        of the points.
    """
    return LinearNDInterpolator(
        np.column_stack([np.ravel(lon_grid.values), np.ravel(lat_grid.values)]),
        _flat_values(field, lon_grid.dims),
    )


def neighborhood_maximum(
    values: xr.DataArray, *, lon_grid, lat_grid, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum of ``values`` over the grid points within ``radius_m``, per point.

    Returns the maxima and the neighbourhood sizes, both flat over the grid
    points in the order ``lon_grid`` carries them. NaN values are skipped, and a
    point whose whole neighbourhood is NaN comes back NaN.

    Parameters
    ----------
    values : xr.DataArray
        The field to take maxima of, on the grid points' own dims.
    lon_grid, lat_grid : xr.DataArray
        The grid points, on the same dims.
    radius_m : float
        Great-circle radius of the neighbourhood, in metres.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(maximum, count)``, each of length the number of grid points.
    """
    tree = KDTree(_sphere_xyz(lon_grid, lat_grid))
    # A great-circle radius becomes the chord of the same arc in the embedding.
    # Half the sphere's circumference is the widest arc the chord distinguishes.
    half_angle = min(0.5 * radius_m / EARTH_RADIUS_M, 0.5 * np.pi)
    neighbors = tree.query_ball_point(
        tree.data,
        2.0 * EARTH_RADIUS_M * np.sin(half_angle),
        return_sorted=False,
        workers=-1,
    )
    count = np.fromiter(map(len, neighbors), dtype=int, count=len(neighbors))
    # No neighbourhood is empty, so the groups of the concatenated indices start
    # at the exclusive cumulative count and `reduceat` needs no empty-group case.
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    flat = np.concatenate(neighbors)
    # fmax rather than maximum, so a NaN neighbour loses instead of propagating.
    return np.fmax.reduceat(np.ravel(values.values)[flat], start), count
