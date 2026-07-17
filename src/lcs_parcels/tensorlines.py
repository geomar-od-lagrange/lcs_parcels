"""Hyperbolic LCS as strain tensor lines. Haller (2015) §5.1 / Table 1 (n=2).

A repelling LCS is a *shrink line* -- a curve tangent to the weak-stretch
eigenvector ``xi_1`` of the Cauchy-Green tensor ``C`` (equivalently, normal to
the strong-stretch ``xi_2`` that the FTLE ridge marks). It solves the tensor-line
ODE ``dr/ds = xi_1(r)``. Attracting LCS need no separate machinery: by the
forward-backward duality (Haller & Sapsis 2011) they are the shrink lines of the
*backward* flow, so :func:`shrink_lines` of a backward :class:`~lcs_parcels.FlowMap`
gives them.

Two functions compose the workflow: :func:`ftle_ridge_seeds` picks start points,
:func:`shrink_lines` integrates the tensor lines through them. Both take the
gridded xarray outputs of a :class:`~lcs_parcels.FlowMap`; the tight ODE loop
drops to NumPy/SciPy (a :class:`scipy.interpolate.RegularGridInterpolator` on the
tensor field), the one place we leave the label-based xarray API.

Rectilinear grids only: like :class:`~lcs_parcels.NeighborFlowMap`, the tensor is
interpolated on axis-aligned ``lon_0``/``lat_0`` axes (``lon_0`` varying along
``i``, ``lat_0`` along ``j``).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Earth radius (m); the metres frame in which C's eigenvectors live (see grids._to_meters).
_R = 6_371_000.0
_DEG = np.pi / 180.0


def _grid_lonlat(obj: xr.DataArray | xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """The ``(i, j)`` diagnostic-grid lon/lat coords: centres if present, else x_0."""
    lon = obj["lon_c"] if "lon_c" in obj.coords else obj["lon_0"]
    lat = obj["lat_c"] if "lat_c" in obj.coords else obj["lat_0"]
    return lon, lat


def ftle_ridge_seeds(
    ftle: xr.DataArray, *, window: int = 7, quantile: float = 0.90
) -> tuple[np.ndarray, np.ndarray]:
    """Seed points at strong local maxima of an FTLE field.

    A grid point is a seed when its FTLE is the maximum over a
    ``window x window`` neighbourhood (a windowed local maximum on the raw
    value) *and* is at or above the ``quantile`` of the field -- an absolute
    magnitude floor, not a local-contrast test. Well separated (spacing set by
    ``window``) so the tensor lines through them do not bundle. NaN cells (e.g.
    the :class:`~lcs_parcels.NeighborFlowMap` edge) never qualify.

    Parameters
    ----------
    ftle : xr.DataArray
        FTLE field with dims ``(i, j)`` and ``lon_0``/``lat_0`` (or
        ``lon_c``/``lat_c``) coordinates, e.g. from :meth:`FlowMap.ftle`.
    window : int, optional
        Side of the square neighbourhood for the local-maximum test (default 7).
    quantile : float, optional
        Global magnitude floor in ``[0, 1]`` (default 0.90 = top decile).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lon, lat)`` 1-D arrays of the seed positions (degrees).
    """
    peak = ftle.rolling(i=window, j=window, center=True, min_periods=1).max()
    is_seed = (ftle >= peak) & (ftle >= ftle.quantile(quantile))
    lon, lat = _grid_lonlat(ftle)
    mask = is_seed.transpose("i", "j").values
    return lon.transpose("i", "j").values[mask], lat.transpose("i", "j").values[mask]


def shrink_lines(
    flowmap,
    seed_lon,
    seed_lat,
    *,
    lambda_max_min: float = 1.1,
    step_m: float = 3_000.0,
    n_steps: int = 250,
) -> xr.Dataset:
    """Integrate shrink lines (``xi_1`` tensor lines) through the seed points.

    Traces the tensor-line ODE ``dr/ds = xi_1(r)`` both ways from each seed,
    where ``xi_1`` is the weak-stretch eigenvector of ``flowmap.cauchy_green()``.
    A *forward* flow map yields repelling LCS; a *backward* one yields attracting
    LCS (Haller-Sapsis duality). The integrator:

    - interpolates the tensor ``C`` (not the eigenvector) and re-diagonalises at
      each point, so it stays smooth through the near-degenerate
      ``lambda_1 ~ lambda_2`` spots where ``xi_1`` is otherwise sign-ambiguous;
    - orients each step to the running heading (an eigenvector has no intrinsic
      sign);
    - stops a line where ``lambda_2 < lambda_max_min`` (a low guard against the
      rare degenerate points), or where it leaves the grid / hits a NaN cell.

    Marches all seeds together with a midpoint (arc-length) step. Lines are a
    fixed ``2 * n_steps + 1`` points long, NaN-filled past termination.

    Parameters
    ----------
    flowmap : FlowMap
        Advected flow map on a rectilinear grid; supplies ``cauchy_green()`` and
        the ``lon_0``/``lat_0`` axes.
    seed_lon, seed_lat : array_like
        Seed positions (degrees), e.g. from :func:`ftle_ridge_seeds`.
    lambda_max_min : float, optional
        Stop a line where the larger eigenvalue ``lambda_2`` falls below this
        (default 1.1). Over long windows the flow is hyperbolic almost
        everywhere, so this is a degeneracy guard, not an LCS selector.
    step_m : float, optional
        Arc-length step in metres (default 3000).
    n_steps : int, optional
        Steps per direction (default 250); full line spans ``2 * n_steps * step_m``.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on dims ``(line, point)``, one ``line`` per
        seed, ordered along the curve. Terminated points are ``NaN``.
    """
    lon_da, lat_da = _grid_lonlat(flowmap.ds)
    lon_axis = lon_da.isel(j=0).values
    lat_axis = lat_da.isel(i=0).values
    C_grid = flowmap.cauchy_green().transpose("i", "j", "row", "col").values
    interp = RegularGridInterpolator(
        (lon_axis, lat_axis), C_grid, bounds_error=False, fill_value=np.nan
    )

    def xi1(lon, lat, heading):
        """Unit xi_1 at (lon, lat), oriented to `heading`; NaN where the line stops."""
        C = interp(np.column_stack([lon, lat]))
        bad = ~np.isfinite(C).all(axis=(1, 2))
        lam, vec = np.linalg.eigh(np.where(bad[:, None, None], np.eye(2), C))
        bad = bad | (lam[:, 1] < lambda_max_min)
        d = vec[:, :, 0]
        d[np.sum(d * heading, axis=1) < 0] *= -1
        d[bad] = np.nan
        return d

    def step(lon, lat, d):
        m_per_deg_lat = _R * _DEG
        m_per_deg_lon = m_per_deg_lat * np.cos(lat * _DEG)
        return lon + d[:, 0] / m_per_deg_lon * step_m, lat + d[:, 1] / m_per_deg_lat * step_m

    def half(sign):
        lon = np.asarray(seed_lon, dtype=float).ravel().copy()
        lat = np.asarray(seed_lat, dtype=float).ravel().copy()
        heading = sign * xi1(lon, lat, np.ones((lon.size, 2)))  # pick the initial branch
        # A seed we cannot trace from (off-grid, NaN cell, or below the guard)
        # makes an all-NaN line rather than a dangling seed point.
        untraceable = ~np.isfinite(heading).all(axis=1)
        lon[untraceable] = np.nan
        lat[untraceable] = np.nan
        track = [(lon.copy(), lat.copy())]
        for _ in range(n_steps):
            d1 = xi1(lon, lat, heading)
            mid_lon, mid_lat = step(lon, lat, 0.5 * d1)
            d2 = xi1(mid_lon, mid_lat, d1)
            lon, lat = step(lon, lat, d2)
            heading = d2
            track.append((lon.copy(), lat.copy()))
        return track

    points = half(-1)[::-1] + half(+1)[1:]  # backward half reversed, then forward
    lon_lines = np.array([p[0] for p in points]).T  # (line, point)
    lat_lines = np.array([p[1] for p in points]).T
    return xr.Dataset(
        {"lon": (("line", "point"), lon_lines), "lat": (("line", "point"), lat_lines)},
        coords={"line": np.arange(lon_lines.shape[0]), "point": np.arange(lon_lines.shape[1])},
    )
