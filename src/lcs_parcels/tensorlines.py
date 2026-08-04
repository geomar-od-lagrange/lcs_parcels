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
tensor field).

Rectilinear grids only: like :class:`~lcs_parcels.NeighborFlowMap`, the tensor is
interpolated on axis-aligned ``lon_grid``/``lat_grid`` axes (``lon_grid`` varying
along ``i``, ``lat_grid`` along ``j``).
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels.grids import _DEG, EARTH_RADIUS_M, _lonlat_to_meters

SECONDS_PER_DAY = 86_400.0


def _odd_cells(window_m: float, spacing_m: float) -> int:
    """Cells spanning ``window_m`` at grid spacing ``spacing_m``, odd and at least 1.

    Odd keeps the rolling window centred on its own grid point; rounding down to
    the nearest odd count keeps it from reaching past ``window_m / 2``.
    """
    cells = round(window_m / spacing_m)
    if cells % 2 == 0:
        cells -= 1
    return max(1, cells)


def _window_cells(ftle: xr.DataArray, window_m: float) -> tuple[int, int]:
    """The ``(i, j)`` cell counts spanning ``window_m`` on the field's own grid.

    Spacing comes from the ``lon_grid``/``lat_grid`` coordinates projected into
    the single-reference-latitude metres frame the package works in
    (:func:`~lcs_parcels.grids._lonlat_to_meters`, one ``cos(phi_ref)``), taken
    as the median cell size along each dimension.
    """
    lon_grid, lat_grid = ftle["lon_grid"], ftle["lat_grid"]
    x, y = _lonlat_to_meters(
        lon_grid, lat_grid, float(lon_grid.mean()), float(lat_grid.mean())
    )
    dx = float(np.abs(x.diff("i")).median())
    dy = float(np.abs(y.diff("j")).median())
    return _odd_cells(window_m, dx), _odd_cells(window_m, dy)


def ftle_ridge_seeds(
    ftle: xr.DataArray, *, window_m: float = 30_000.0, quantile: float = 0.90
) -> tuple[np.ndarray, np.ndarray]:
    """Seed points at strong local maxima of an FTLE field.

    A grid point is a seed when its FTLE is the maximum over a neighbourhood
    spanning ``window_m`` in each direction (a windowed local maximum on the raw
    value) *and* is at or above the ``quantile`` of the field -- an absolute
    magnitude floor, not a local-contrast test. Well separated (spacing set by
    ``window_m``) so the tensor lines through them do not bundle. NaN cells (e.g.
    the :class:`~lcs_parcels.NeighborFlowMap` edge) never qualify.

    Parameters
    ----------
    ftle : xr.DataArray
        FTLE field with dims ``(i, j)`` and ``lon_grid``/``lat_grid``
        coordinates, e.g. from :meth:`FlowMap.ftle`.
    window_m : float, optional
        Side of the square neighbourhood in metres (default 30 km). Converted to
        an odd cell count per dimension from the field's own grid spacing, so the
        separation between seeds is a physical distance and does not change with
        grid resolution. On a 1/25-degree grid at 20 N (about 4.2 km cells) the
        default is 7 cells.
    quantile : float, optional
        Global magnitude floor in ``[0, 1]`` (default 0.90 = top decile).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lon, lat)`` 1-D arrays of the seed positions (degrees).
    """
    cells_i, cells_j = _window_cells(ftle, window_m)
    peak = ftle.rolling(i=cells_i, j=cells_j, center=True, min_periods=1).max()
    is_seed = (ftle >= peak) & (ftle >= ftle.quantile(quantile))
    lon = ftle["lon_grid"].transpose("i", "j").values
    lat = ftle["lat_grid"].transpose("i", "j").values
    mask = is_seed.transpose("i", "j").values
    return lon[mask], lat[mask]


def _shrink_direction(
    lon: np.ndarray,
    lat: np.ndarray,
    heading: np.ndarray,
    *,
    tensor_interp: RegularGridInterpolator,
    lambda_max_min: float,
) -> np.ndarray:
    """Unit ``xi_1`` at each ``(lon, lat)``, oriented to ``heading``.

    Parameters
    ----------
    lon, lat : np.ndarray
        Positions (degrees), shape ``(n,)``.
    heading : np.ndarray
        Shape ``(n, 2)``; the running direction each returned vector is aligned
        with. Need not be a unit vector -- only its sign against ``xi_1``
        matters.
    tensor_interp : RegularGridInterpolator
        Interpolator over the Cauchy-Green tensor field, returning ``(n, 2, 2)``
        and ``NaN`` off-grid.
    lambda_max_min : float
        Degeneracy guard: points whose ``lambda_2`` falls below this return NaN.

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` unit vectors in the metres frame; ``NaN`` rows where the
        point is off-grid, in a NaN cell, or below ``lambda_max_min``.
    """
    cauchy_green = tensor_interp(np.column_stack([lon, lat]))
    terminated = ~np.isfinite(cauchy_green).all(axis=(1, 2))
    # eigh returns eigenvalues ascending: eigenvalues[:, 0] = lambda_1 (the
    # *smaller*, weak-stretch eigenvalue) with eigenvector eigenvectors[:, :, 0]
    # = xi_1, the shrink-line tangent; eigenvalues[:, 1] = lambda_2 = lambda_max.
    # Off-grid points are diagonalised as the identity purely to keep eigh from
    # raising; their rows are overwritten with NaN below.
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.where(terminated[:, None, None], np.eye(2), cauchy_green)
    )
    # Stop at near-degenerate points: where lambda_1 ~ lambda_2 the tensor is
    # close to isotropic and xi_1 is an arbitrary direction in the plane, so
    # continuing would trace numerical noise.
    terminated = terminated | (eigenvalues[:, 1] < lambda_max_min)
    direction = eigenvectors[:, :, 0]
    # An eigenvector has no intrinsic sign, so eigh's choice flips arbitrarily
    # between neighbouring points. Flip each one to the acute side of the running
    # heading, which is what makes the marched sequence a continuous curve rather
    # than a zig-zag.
    direction[np.sum(direction * heading, axis=1) < 0] *= -1
    direction[terminated] = np.nan
    return direction


def _step_lonlat_by_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    direction: np.ndarray,
    *,
    step_m: float,
    lat_ref: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance ``(lon, lat)`` by ``step_m`` metres along ``direction``.

    ``direction`` is a vector in the single-reference-latitude metres frame the
    Cauchy-Green tensor lives in (:func:`~lcs_parcels.grids._to_meters`), so the
    conversion back to degrees uses the one ``lat_ref``, not a per-point
    ``cos(lat)``. A unit ``direction`` moves exactly ``step_m``; a shorter one
    (the half-step of the midpoint scheme) moves proportionally less.

    Parameters
    ----------
    lon, lat : np.ndarray
        Positions (degrees), shape ``(n,)``.
    direction : np.ndarray
        Shape ``(n, 2)``, metres-frame ``(x, y)`` components.
    step_m : float
        Arc length in metres for a unit ``direction``.
    lat_ref : float
        The reference latitude (degrees) of the metres frame.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The stepped ``(lon, lat)`` in degrees.
    """
    m_per_deg_lat = EARTH_RADIUS_M * _DEG
    m_per_deg_lon = m_per_deg_lat * np.cos(lat_ref * _DEG)
    return (
        lon + direction[:, 0] / m_per_deg_lon * step_m,
        lat + direction[:, 1] / m_per_deg_lat * step_m,
    )


def _trace_half_line(
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    sign: int,
    *,
    tensor_interp: RegularGridInterpolator,
    lambda_max_min: float,
    step_m: float,
    lat_ref: float,
    n_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """March every seed ``n_steps`` steps in one of the two ``xi_1`` directions.

    Integrates ``dr/ds = xi_1(r)`` with RK2 (the midpoint / modified-Euler
    scheme): evaluate ``xi_1`` at the current point, half-step along it, evaluate
    again at that midpoint, and take the full step along the midpoint direction.

    Parameters
    ----------
    seed_lon, seed_lat : np.ndarray
        Seed positions (degrees), shape ``(n,)``.
    sign : int
        ``+1`` or ``-1``, selecting which of the two opposite ``xi_1`` branches
        this half follows away from the seeds.
    tensor_interp, lambda_max_min, step_m, lat_ref
        As for :func:`_shrink_direction` and :func:`_step_lonlat_by_meters`.
    n_steps : int
        Number of steps taken; the returned track has ``n_steps + 1`` entries,
        the first being the seeds themselves.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        ``n_steps + 1`` ``(lon, lat)`` pairs, each of shape ``(n,)``, ordered
        along the curve away from the seed. Terminated lines carry ``NaN`` from
        the step that terminated them onward.
    """
    lon = np.asarray(seed_lon, dtype=float).ravel().copy()
    lat = np.asarray(seed_lat, dtype=float).ravel().copy()
    # Pick the initial branch by dotting xi_1 against the 45-degree direction --
    # an arbitrary tie-break, since at the seed there is no running heading yet.
    # A seed whose xi_1 lies near the anti-diagonal therefore flips branch on
    # numerical noise. The two halves make this choice independently, so a curve
    # can kink at the seed point they share.
    heading = sign * _shrink_direction(
        lon,
        lat,
        np.ones((lon.size, 2)),
        tensor_interp=tensor_interp,
        lambda_max_min=lambda_max_min,
    )
    # A seed we cannot trace from (off-grid, NaN cell, or below the guard)
    # makes an all-NaN line rather than a dangling seed point.
    untraceable = ~np.isfinite(heading).all(axis=1)
    lon[untraceable] = np.nan
    lat[untraceable] = np.nan
    track = [(lon.copy(), lat.copy())]
    for _ in range(n_steps):
        direction = _shrink_direction(
            lon,
            lat,
            heading,
            tensor_interp=tensor_interp,
            lambda_max_min=lambda_max_min,
        )
        mid_lon, mid_lat = _step_lonlat_by_meters(
            lon, lat, 0.5 * direction, step_m=step_m, lat_ref=lat_ref
        )
        # The degeneracy guard is evaluated at the RK2 midpoint too, so a step
        # whose two endpoints are both fine still terminates the line if the
        # tensor is degenerate halfway along it.
        mid_direction = _shrink_direction(
            mid_lon,
            mid_lat,
            direction,
            tensor_interp=tensor_interp,
            lambda_max_min=lambda_max_min,
        )
        lon, lat = _step_lonlat_by_meters(
            lon, lat, mid_direction, step_m=step_m, lat_ref=lat_ref
        )
        heading = mid_direction
        # Every line runs the full n_steps and is NaN-filled past termination,
        # rather than breaking out: the whole seed population marches together in
        # one array, so there is nothing to break out of, and the result is a
        # rectangular (line, point) block. That costs work on lines that died
        # early and pads the output; see #11.
        track.append((lon.copy(), lat.copy()))
    return track


def shrink_lines(
    flowmap,
    *,
    seed_lon,
    seed_lat,
    ftle_min_per_day: float = 0.005,
    step_m: float = 3_000.0,
    line_length_m: float = 1_500_000.0,
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
    - stops a line where the local stretching falls below ``ftle_min_per_day``
      (a low guard against the rare degenerate points), or where it leaves the
      grid / hits a NaN cell.

    Marches all seeds together with a midpoint (arc-length) step. Every line is
    the same length, NaN-filled past termination.

    Parameters
    ----------
    flowmap : FlowMap
        Advected flow map on a rectilinear grid; supplies ``cauchy_green()`` and
        the ``lon_grid``/``lat_grid`` axes.
    seed_lon, seed_lat : array_like
        Seed positions (degrees), e.g. from :func:`ftle_ridge_seeds`.
    ftle_min_per_day : float, optional
        Stop a line where the local FTLE falls below this, in 1/day (default
        0.005). Expressed as a stretching *rate* so the guard means the same
        thing whatever ``|T|`` the flow map spans; it converts to the eigenvalue
        floor ``lambda_min = exp(2 |T|_days Lambda_min)`` used against
        ``lambda_2``, which over a 7-day window is ``lambda_min = 1.07``. Over
        long windows the flow is hyperbolic almost everywhere, so this is a
        degeneracy guard, not an LCS selector.
    step_m : float, optional
        Arc-length step in metres (default 3000).
    line_length_m : float, optional
        Full length of each line in metres (default 1500 km), traced half in
        each direction from the seed; the step count per direction is
        ``line_length_m / (2 * step_m)``, at least 1.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on dims ``(line, point)``, one ``line`` per
        seed, ordered along the curve. Terminated points are ``NaN``.
    """
    n_steps = max(1, round(line_length_m / (2.0 * step_m)))
    # FTLE = (1 / |T|) * 0.5 * log(lambda_max), so a floor on the FTLE in 1/day
    # is a floor exp(2 |T|_days Lambda_min) on lambda_max.
    t_days = flowmap._integration_seconds() / SECONDS_PER_DAY
    lambda_max_min = float(np.exp(2.0 * t_days * ftle_min_per_day))
    lon_axis = flowmap.lon_grid.isel(j=0).values
    lat_axis = flowmap.lat_grid.isel(i=0).values
    # xi_1 is a direction in the single-reference-latitude metres frame C lives in
    # (grids._to_meters), so the arc-length step converts back to degrees with that
    # one reference latitude, not a per-point cos(lat).
    lat_ref = float(flowmap.ds["lat_0"].mean())
    # CG_grid (not "C-grid": no Arakawa staggering here) is the Cauchy-Green
    # tensor on the analysis grid, interpolated point-by-point during the trace.
    CG_grid = flowmap.cauchy_green().transpose("i", "j", "row", "col").values
    # The one place this package leaves the label-based xarray API: the ODE loop
    # below evaluates the tensor at millions of scattered points, which
    # ``.interp()`` cannot do without building an xarray object per step.
    tensor_interp = RegularGridInterpolator(
        (lon_axis, lat_axis), CG_grid, bounds_error=False, fill_value=np.nan
    )

    trace_kwargs = {
        "tensor_interp": tensor_interp,
        "lambda_max_min": lambda_max_min,
        "step_m": step_m,
        "lat_ref": lat_ref,
        "n_steps": n_steps,
    }
    # Trace both ways from each seed and stitch into one curve through it: the
    # backward half reversed (so it runs into the seed), then the forward half
    # with its first point (the seed, shared) dropped. Each half picks its
    # initial branch on its own, so the stitched curve may kink at the seed.
    points = _trace_half_line(seed_lon, seed_lat, -1, **trace_kwargs)[::-1]
    points += _trace_half_line(seed_lon, seed_lat, +1, **trace_kwargs)[1:]
    lon_lines = np.array([p[0] for p in points]).T  # (line, point)
    lat_lines = np.array([p[1] for p in points]).T
    return xr.Dataset(
        {
            "lon": xr.DataArray(
                lon_lines,
                dims=("line", "point"),
                attrs={
                    "long_name": "longitude along the shrink line",
                    "units": "degrees_east",
                },
            ),
            "lat": xr.DataArray(
                lat_lines,
                dims=("line", "point"),
                attrs={
                    "long_name": "latitude along the shrink line",
                    "units": "degrees_north",
                },
            ),
        },
        coords={
            "line": xr.DataArray(
                np.arange(lon_lines.shape[0]),
                dims="line",
                attrs={"long_name": "shrink line index, one per seed point"},
            ),
            "point": xr.DataArray(
                np.arange(lon_lines.shape[1]),
                dims="point",
                attrs={"long_name": "point index along the shrink line"},
            ),
        },
    )
