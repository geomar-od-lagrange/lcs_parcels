"""Numerical primitives: distances on the sphere, windowed extrema, line tracing.

Positions are lon/lat in degrees, distances are metres on a sphere of radius
:data:`EARTH_RADIUS_M`, and directions are local ``(east, north)`` vectors.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# --- sphere ----------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in metres. All distances are taken on a sphere of this radius."""

_M_PER_DEG = EARTH_RADIUS_M * (np.pi / 180.0)
"""Metres per degree of latitude, and of longitude at the equator."""

_LONLAT_ATTRS = {
    "lon": {"units": "degrees_east"},
    "lat": {"units": "degrees_north"},
}


def _wrap_lon(dlon):
    """Wrap a longitude *difference* (degrees) into ``[-180, 180]``.

    Takes plain arrays or xarray objects. Applied to differences and to the
    offsets a circular mean is built from, never to a stored position.
    """
    # Subtracting the nearest multiple of 360, rather than shifting and taking a
    # modulo, returns an argument already inside the range bit-for-bit.
    return dlon - 360.0 * np.round(dlon / 360.0)


def _separation_m(*, lon_a, lat_a, lon_b, lat_b):
    """East/north separation of point ``b`` from point ``a``, in metres.

    Takes plain arrays or xarray objects. The longitude difference is wrapped and
    scaled by the cosine of the pair's mid-latitude, and the latitude difference
    by the Earth radius alone, so each pair is differenced in its own frame.

    Returns
    -------
    tuple
        ``(dx, dy)``, the eastward and northward components in metres.
    """
    dlon = _wrap_lon(lon_b - lon_a)
    lat_mid = 0.5 * (lat_a + lat_b)
    dx = _M_PER_DEG * np.cos(np.deg2rad(lat_mid)) * dlon
    # A position is a pair, so `0.0 * dlon` carries a lost longitude into the
    # north component, which otherwise never touches lon.
    dy = _M_PER_DEG * ((lat_b - lat_a) + 0.0 * dlon)
    return dx, dy


def _circular_mean_lon(lon: xr.DataArray, dim: str) -> xr.DataArray:
    """Mean longitude over ``dim``, taken on the circle.

    Anchored on the first element along ``dim`` and averaged over wrapped offsets
    from it, so the result stays on the anchor's branch. ``skipna=False`` makes
    the mean NaN if any member is, as one lost stencil point invalidates it.
    """
    anchor = lon.isel({dim: 0}, drop=True)
    return anchor + _wrap_lon(lon - anchor).mean(dim, skipna=False)


def _step_lonlat_by_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    direction: np.ndarray,
    *,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance ``(lon, lat)`` by ``step_m`` metres along ``direction``.

    ``direction`` is a local east/north vector at ``(lon, lat)``. The step is the
    exact inverse of :func:`_separation_m`, so measuring it afterwards returns
    ``step_m * direction``. Latitude is not folded at the pole.

    Parameters
    ----------
    lon, lat : np.ndarray
        Positions (degrees), shape ``(n,)``.
    direction : np.ndarray
        Shape ``(n, 2)``, local ``(east, north)`` components.
    step_m : float
        Arc length in metres for a unit ``direction``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stepped ``(lon, lat)`` in degrees.
    """
    dlat = direction[:, 1] * step_m / _M_PER_DEG
    lat_mid = lat + 0.5 * dlat
    return (
        lon + direction[:, 0] * step_m / (_M_PER_DEG * np.cos(np.deg2rad(lat_mid))),
        lat + dlat,
    )


# --- windowed extrema --------------------------------------------------------


def _odd_cells(window_m: float, spacing_m: float) -> int:
    """Number of cells covering ``window_m`` at spacing ``spacing_m``, made odd.

    An odd count has a middle cell, so the rolling window sits centred on its own
    grid point. The count is rounded to nearest, dropped by one if even, and at
    least 1.
    """
    cells = round(window_m / spacing_m)
    if cells % 2 == 0:
        cells -= 1
    return max(1, cells)


def _grid_spacing_m(field: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Absolute east separation of adjacent points along ``i`` and north
    separation along ``j``, in metres, from ``lon_grid``/``lat_grid``.

    Both are on ``(i, j)`` and NaN at the first index of their dim.
    """
    lon_grid, lat_grid = field["lon_grid"], field["lat_grid"]
    dx, _ = _separation_m(
        lon_a=lon_grid.shift(i=1),
        lat_a=lat_grid.shift(i=1),
        lon_b=lon_grid,
        lat_b=lat_grid,
    )
    _, dy = _separation_m(
        lon_a=lon_grid.shift(j=1),
        lat_a=lat_grid.shift(j=1),
        lon_b=lon_grid,
        lat_b=lat_grid,
    )
    return np.abs(dx), np.abs(dy)


def _window_geometry(field: xr.DataArray, window_m: float) -> dict[str, float]:
    """Convert a window in metres into cell counts on this field's grid.

    The cell counts use the **median** grid spacing, the size most of the grid
    has. ``min_seed_separation_m`` uses the **minimum**, since that is where two
    windowed extrema get closest.

    A window of ``cells`` reaches ``(cells - 1) // 2`` cells either side, so the
    nearest competing *strict* extremum is one cell beyond that. Every cell of a
    plateau of equal values ties, so adjacent cells can all qualify.
    """
    dx, dy = _grid_spacing_m(field)
    spacing_i = float(dx.median())
    spacing_j = float(dy.median())
    cells_i = _odd_cells(window_m, spacing_i)
    cells_j = _odd_cells(window_m, spacing_j)
    return {
        "window_m": float(window_m),
        "window_cells_i": cells_i,
        "window_cells_j": cells_j,
        "grid_spacing_i_m": spacing_i,
        "grid_spacing_j_m": spacing_j,
        "min_seed_separation_m": min(
            ((cells_i - 1) // 2 + 1) * float(dx.min()),
            ((cells_j - 1) // 2 + 1) * float(dy.min()),
        ),
    }


def _windowed_extrema(
    field: xr.DataArray, *, window_m: float, extremum: str
) -> tuple[xr.DataArray, dict[str, float]]:
    """Where ``field`` is the ``extremum`` (``"min"`` or ``"max"``) of a square
    window of side ``window_m`` centred on the point.

    Returns the boolean mask on ``(i, j)``, False at NaN cells, and the
    :func:`_window_geometry` of ``window_m``.

    Raises
    ------
    ValueError
        If ``extremum`` is neither ``"min"`` nor ``"max"``.
    """
    if extremum not in ("min", "max"):
        raise ValueError(f'extremum must be "min" or "max", got {extremum!r}')
    geometry = _window_geometry(field, window_m)
    rolling = field.rolling(
        i=geometry["window_cells_i"],
        j=geometry["window_cells_j"],
        center=True,
        min_periods=1,
    )
    mask = field <= rolling.min() if extremum == "min" else field >= rolling.max()
    return mask, geometry


# --- tracing -----------------------------------------------------------------


def _rk2_step(
    lon: np.ndarray,
    lat: np.ndarray,
    heading: np.ndarray,
    *,
    tangent,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One midpoint (RK2) step of ``dr/ds = tangent(r)`` along a direction field.

    ``tangent(lon, lat, heading)`` returns unit ``(east, north)`` vectors
    oriented to ``heading``, or NaN rows where the field is not well defined.

    The step evaluates ``tangent`` at the current point and at the half-step
    midpoint, and takes the full step along the midpoint direction.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The stepped ``lon``, ``lat`` and the midpoint direction, which is the
        heading for the next step.
    """
    direction = tangent(lon, lat, heading)
    mid_lon, mid_lat = _step_lonlat_by_meters(lon, lat, 0.5 * direction, step_m=step_m)
    # The tangent is evaluated at the RK2 midpoint too, so a step with sound
    # endpoints still terminates on an ill-defined field halfway along.
    mid_direction = tangent(mid_lon, mid_lat, direction)
    lon, lat = _step_lonlat_by_meters(lon, lat, mid_direction, step_m=step_m)
    return lon, lat, mid_direction


def _tensor_interp(cauchy_green: xr.DataArray) -> RegularGridInterpolator:
    """Interpolator over a ``(i, j, row, col)`` tensor on its rectilinear axes.

    Returns ``(n, 2, 2)`` tensors for ``(n, 2)`` lon/lat points, and ``NaN`` off
    the grid.
    """
    lon_axis = cauchy_green["lon_grid"].isel(j=0).values
    lat_axis = cauchy_green["lat_grid"].isel(i=0).values
    # CG_grid is the Cauchy-Green tensor on the diagnostic grid, not an Arakawa
    # C-grid.
    CG_grid = cauchy_green.transpose("i", "j", "row", "col").values
    # The tracers leave the label-based xarray API here, because the ODE loop
    # would otherwise build an xarray object per step.
    return RegularGridInterpolator(
        (lon_axis, lat_axis), CG_grid, bounds_error=False, fill_value=np.nan
    )
