"""Hyperbolic LCS as strain tensor lines.

Haller (2015) §5.1 / Table 1 (n=2), doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).

A repelling LCS is a *shrink line* -- a curve tangent to the weak-stretch
eigenvector ``xi_1`` of the Cauchy-Green tensor ``C`` (equivalently, normal to
the strong-stretch ``xi_2`` that the FTLE ridge marks). It solves the tensor-line
ODE ``dr/ds = xi_1(r)``. Attracting LCS need no separate machinery. By the
forward-backward duality (Haller & Sapsis 2011, https://doi.org/10.1063/1.3579597)
they are the shrink lines of the *backward* flow, so :func:`shrink_lines` of a
backward :class:`~lcs_parcels.FlowMap` gives them.

Two functions compose the workflow. :func:`ftle_ridge_seeds` picks seed points
and :func:`shrink_lines` integrates the tensor lines through them. Both take the
gridded xarray outputs of a :class:`~lcs_parcels.FlowMap`, and
:meth:`~lcs_parcels.FlowMap.hyperbolic_lcs` runs the pair in one call.

Rectilinear grids only. The tensor is interpolated on axis-aligned
``lon_grid``/``lat_grid`` axes (``lon_grid`` varying along ``i``, ``lat_grid``
along ``j``), and the ``lon_grid`` axis must be monotonic. Only the axis is
constrained, and a traced line may run anywhere.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels.grids import _M_PER_DEG, _separation_m


def _odd_cells(window_m: float, spacing_m: float) -> int:
    """Number of cells covering ``window_m`` at spacing ``spacing_m``, made odd.

    An odd count has a middle cell, which is what lets the rolling window sit
    centred on its own grid point. Rounding down to odd shortens the window by at
    most one cell and changes nothing about the grid.
    """
    cells = round(window_m / spacing_m)
    if cells % 2 == 0:
        cells -= 1
    return max(1, cells)


def _window_geometry(ftle: xr.DataArray, window_m: float) -> dict[str, float]:
    """Convert a window in metres into cell counts on this field's grid.

    The rolling window is counted in cells, so ``window_m`` has to be divided by
    a cell size, and the two returned quantities need different ones.

    The cell counts use the **median** adjacent-point separation, the size most
    of the grid has. ``min_seed_separation_m``, the closest two seeds can be,
    uses the **minimum** instead, since that is where seeds get closest. A window
    of ``cells`` reaches ``(cells - 1) // 2`` cells either side, so the nearest
    competing maximum is one cell beyond that.

    The bound holds for strict maxima only. Selection is ``ftle >= rolling max``,
    so every cell of a plateau of equal values ties and adjacent cells can all be
    seeds.
    """
    lon_grid, lat_grid = ftle["lon_grid"], ftle["lat_grid"]
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
    dx, dy = np.abs(dx), np.abs(dy)
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


def ftle_ridge_seeds(
    ftle: xr.DataArray,
    *,
    window_m: float = 30_000.0,
    quantile: float | None = None,
    ftle_min: float | None = None,
) -> xr.Dataset:
    """Seed points at strong local maxima of an FTLE field.

    A grid point is a seed when its FTLE is the maximum over a window
    ``window_m`` wide in total, centred on it (a windowed local maximum on the
    raw value) *and* is at or above a magnitude floor. The floor is either a
    ``quantile`` of this field or an absolute ``ftle_min``, and it tests the
    magnitude of the value rather than its contrast against the surrounding
    cells. NaN cells (e.g., the :class:`~lcs_parcels.NeighborFlowMap` edge) never
    qualify.

    A window of side ``window_m`` reaches only ``window_m / 2`` to either side of
    its own grid point, so two seeds can sit about ``window_m / 2`` apart, not
    ``window_m``. The returned ``min_seed_separation_m`` attribute is that floor
    computed on this grid.

    Warns when ``window_m`` spans fewer than three cells in either dimension,
    because a one-cell window makes every point a windowed maximum, so the
    local-maximum test stops selecting and only the magnitude floor is left.

    Parameters
    ----------
    ftle : xr.DataArray
        FTLE field with dims ``(i, j)`` and ``lon_grid``/``lat_grid``
        coordinates, e.g., from :meth:`FlowMap.ftle`.
    window_m : float, optional
        Side of the square window in metres (default 30 km). Converted to an odd
        cell count per dimension from the field's own grid spacing, so the
        separation between seeds is a physical distance and does not change with
        grid resolution. On a 1/25-degree grid at 20 N (about 4.2 km cells) the
        default is 7 cells.
    quantile : float, optional
        Magnitude floor as a quantile over all finite cells of the field, in
        ``[0, 1]``. Defaults to 0.90 (the top decile) when neither selector is
        given.
    ftle_min : float, optional
        Magnitude floor as an absolute value, in the units of ``ftle`` (1/s for
        :meth:`FlowMap.ftle`). Mutually exclusive with ``quantile``.

        The quantile is the default because it means the same thing on any field.
        Use ``ftle_min`` to hold several windows or regions to one common
        threshold, which a quantile cannot express.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on a ``seed`` dim, one entry per seed point.
        The attributes record what ``window_m`` became on this grid: the cell
        counts used, the median grid spacing, and ``min_seed_separation_m``.

    Raises
    ------
    ValueError
        If both ``quantile`` and ``ftle_min`` are given.
    """
    if quantile is not None and ftle_min is not None:
        raise ValueError("give either quantile or ftle_min, not both")
    if quantile is None and ftle_min is None:
        quantile = 0.90
    threshold = float(ftle.quantile(quantile)) if ftle_min is None else float(ftle_min)

    geometry = _window_geometry(ftle, window_m)
    cells_i = geometry["window_cells_i"]
    cells_j = geometry["window_cells_j"]
    if cells_i < 3 or cells_j < 3:
        warnings.warn(
            f"window_m={window_m} spans {cells_i}x{cells_j} cells of this grid "
            f"({geometry['grid_spacing_i_m']:.0f} x "
            f"{geometry['grid_spacing_j_m']:.0f} m). A window under three cells "
            "makes every point a windowed maximum in that dimension, so the "
            "local-maximum test stops selecting. Consider wider window_m or finer "
            "grid.",
            UserWarning,
            stacklevel=2,
        )

    peak = ftle.rolling(i=cells_i, j=cells_j, center=True, min_periods=1).max()
    is_seed = (ftle >= peak) & (ftle >= threshold)
    picked = (
        xr.Dataset({"lon": ftle["lon_grid"], "lat": ftle["lat_grid"]})
        .stack(seed=("i", "j"))
        .where(is_seed.stack(seed=("i", "j")), drop=True)
    )
    lon = picked["lon"].values
    lat = picked["lat"].values
    return xr.Dataset(
        {
            "lon": xr.DataArray(
                lon,
                dims="seed",
                attrs={
                    "long_name": "longitude of the FTLE ridge seed",
                    "units": "degrees_east",
                },
            ),
            "lat": xr.DataArray(
                lat,
                dims="seed",
                attrs={
                    "long_name": "latitude of the FTLE ridge seed",
                    "units": "degrees_north",
                },
            ),
        },
        coords={
            "seed": xr.DataArray(
                np.arange(lon.size),
                dims="seed",
                attrs={"long_name": "FTLE ridge seed index"},
            )
        },
        attrs={
            "long_name": "seed points at strong local maxima of the FTLE field",
            "selector": "quantile" if ftle_min is None else "ftle_min",
            "ftle_threshold": threshold,
            **geometry,
        },
    )


def _shrink_line_tangent(
    lon: np.ndarray,
    lat: np.ndarray,
    heading: np.ndarray,
    *,
    tensor_interp: RegularGridInterpolator,
    min_anisotropy: float,
) -> np.ndarray:
    """Unit ``xi_1`` at each ``(lon, lat)``, oriented to ``heading``.

    ``xi_1`` belongs to the smaller eigenvalue of ``C``, so it is the direction
    material line elements shrink along, and the shrink line is the curve tangent
    to it everywhere.

    Returns ``NaN`` wherever the tangent is **not well defined**, which covers a
    point off the grid, a point in a NaN cell, and a tensor too close to
    isotropic for its eigenvectors to be resolved (``min_anisotropy``). Callers
    read a NaN row as "the line ends here" and do not need to know which of the
    three fired.

    Parameters
    ----------
    lon, lat : np.ndarray
        Positions (degrees), shape ``(n,)``.
    heading : np.ndarray
        Shape ``(n, 2)``; the running direction each returned vector is aligned
        with. Need not be a unit vector, since only its sign against ``xi_1``
        matters.
    tensor_interp : RegularGridInterpolator
        Interpolator over the Cauchy-Green tensor field, returning ``(n, 2, 2)``
        and ``NaN`` off-grid.
    min_anisotropy : float
        Well-definedness guard on ``lambda_2 / lambda_1``; see
        :func:`shrink_lines`.

    Returns
    -------
    np.ndarray
        Shape ``(n, 2)`` unit ``(east, north)`` vectors; ``NaN`` rows wherever
        the tangent is not well defined.
    """
    cauchy_green = tensor_interp(np.column_stack([lon, lat]))
    terminated = ~np.isfinite(cauchy_green).all(axis=(1, 2))
    # eigh returns eigenvalues ascending, so index 0 is xi_1. Off-grid points go
    # in as the identity to keep eigh from raising, and are NaN'd below.
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.where(terminated[:, None, None], np.eye(2), cauchy_green)
    )
    # Stop where the eigenvalues are too close to separate. Clamping a
    # round-off-negative lambda_1 to zero lets very anisotropic points through.
    terminated = terminated | (
        eigenvalues[:, 1] < min_anisotropy * np.maximum(eigenvalues[:, 0], 0.0)
    )
    direction = eigenvectors[:, :, 0]
    # An eigenvector has no intrinsic sign, so agreeing with the running heading
    # is what keeps the marched sequence continuous instead of zig-zag.
    direction[np.sum(direction * heading, axis=1) < 0] *= -1
    direction[terminated] = np.nan
    return direction


def _step_lonlat_by_meters(
    lon: np.ndarray,
    lat: np.ndarray,
    direction: np.ndarray,
    *,
    step_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance ``(lon, lat)`` by ``step_m`` metres along ``direction``.

    ``direction`` is a local east/north vector at ``(lon, lat)``, the frame ``C``
    is built in. The step is the exact inverse of the measurement that built it
    (:func:`~lcs_parcels.grids._separation_m`), so measuring it afterwards
    returns the vector asked for. A unit ``direction`` moves ``step_m``, and a
    shorter one moves proportionally less.

    Inverting the measurement, rather than solving the direct great-circle
    problem, is what keeps the traced curve on the direction field. Latitude is
    not folded at the pole, so a line stepped past 90 degrees runs off the chart
    and terminates at the next tangent lookup.

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


def _trace_half_line(
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    sign: int,
    *,
    tensor_interp: RegularGridInterpolator,
    min_anisotropy: float,
    step_m: float,
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
    tensor_interp, min_anisotropy, step_m
        As for :func:`_shrink_line_tangent` and :func:`_step_lonlat_by_meters`.
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
    # No running heading at the seed, so the branch comes from the 45-degree
    # direction. It ignores `sign`, so the halves start opposite.
    heading = sign * _shrink_line_tangent(
        lon,
        lat,
        np.ones((lon.size, 2)),
        tensor_interp=tensor_interp,
        min_anisotropy=min_anisotropy,
    )
    # A seed whose tangent is not well defined makes an all-NaN line rather than
    # a dangling seed point.
    untraceable = ~np.isfinite(heading).all(axis=1)
    lon[untraceable] = np.nan
    lat[untraceable] = np.nan
    track = [(lon.copy(), lat.copy())]
    for _ in range(n_steps):
        direction = _shrink_line_tangent(
            lon,
            lat,
            heading,
            tensor_interp=tensor_interp,
            min_anisotropy=min_anisotropy,
        )
        mid_lon, mid_lat = _step_lonlat_by_meters(
            lon, lat, 0.5 * direction, step_m=step_m
        )
        # The guard runs at the RK2 midpoint too, so a step with sound endpoints
        # still terminates on a degenerate tensor halfway along.
        mid_direction = _shrink_line_tangent(
            mid_lon,
            mid_lat,
            direction,
            tensor_interp=tensor_interp,
            min_anisotropy=min_anisotropy,
        )
        lon, lat = _step_lonlat_by_meters(lon, lat, mid_direction, step_m=step_m)
        heading = mid_direction
        # All seeds march in one array, so a terminated line is NaN-filled rather
        # than broken out of. That costs work on lines that died early, see #11.
        track.append((lon.copy(), lat.copy()))
    return track


def shrink_lines(
    flowmap,
    *,
    seed_lon,
    seed_lat,
    min_anisotropy: float = 1.15,
    step_m: float = 3_000.0,
    line_length_m: float = 1_500_000.0,
) -> xr.Dataset:
    """Integrate shrink lines (``xi_1`` tensor lines) through the seed points.

    Traces the tensor-line ODE ``dr/ds = xi_1(r)`` both ways from each seed,
    where ``xi_1`` is the weak-stretch eigenvector of ``flowmap.cauchy_green()``.
    A *forward* flow map yields repelling LCS and a *backward* one yields
    attracting LCS, by the forward-backward duality. The integrator:

    - interpolates the tensor ``C`` rather than the eigenvector and
      re-diagonalises at each point, so it stays smooth through the
      near-degenerate spots where ``xi_1`` is otherwise sign-ambiguous;
    - orients each step to the running heading, an eigenvector having no
      intrinsic sign;
    - stops a line where ``xi_1`` stops being well defined, whether through
      ``min_anisotropy``, running off the grid, or a NaN cell.

    All seeds march together as one array with a midpoint (arc-length) step.
    ``line_length_m`` is a cap, so a line that terminates early is shorter and
    the returned block is NaN-filled past termination to keep every row the same
    length.

    Parameters
    ----------
    flowmap : FlowMap
        Advected flow map on a rectilinear grid, supplying ``cauchy_green()`` and
        the ``lon_grid``/``lat_grid`` axes.
    seed_lon, seed_lat : array_like
        Seed positions (degrees), e.g., from :func:`ftle_ridge_seeds`.
    min_anisotropy : float, optional
        Stop a line where ``lambda_2 / lambda_1`` falls below this (default
        1.15). It is a well-definedness guard on whether ``xi_1`` is a direction
        or numerical noise, not a selector for which structures count as LCS,
        which is what ``quantile`` in :func:`ftle_ridge_seeds` sets. Being a
        ratio it retunes with neither the window nor the flow regime.
    step_m : float, optional
        Arc-length step in metres (default 3000).
    line_length_m : float, optional
        Maximum length of each line in metres (default 1500 km), traced half in
        each direction from the seed. The step count per direction is
        ``line_length_m / (2 * step_m)``, at least 1. Lines that terminate early
        are shorter.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on dims ``(line, point)``, one ``line`` per
        seed, ordered along the curve. Terminated points are ``NaN``.
    """
    n_steps = max(1, round(line_length_m / (2.0 * step_m)))
    lon_axis = flowmap.lon_grid.isel(j=0).values
    lat_axis = flowmap.lat_grid.isel(i=0).values
    # CG_grid is the Cauchy-Green tensor on the diagnostic grid, not an Arakawa
    # C-grid.
    CG_grid = flowmap.cauchy_green().transpose("i", "j", "row", "col").values
    # The one place this package leaves the label-based xarray API, because the
    # ODE loop would otherwise build an xarray object per step.
    tensor_interp = RegularGridInterpolator(
        (lon_axis, lat_axis), CG_grid, bounds_error=False, fill_value=np.nan
    )

    trace_kwargs = {
        "tensor_interp": tensor_interp,
        "min_anisotropy": min_anisotropy,
        "step_m": step_m,
        "n_steps": n_steps,
    }
    # Stitch the halves into one curve, reversing the backward one and dropping
    # the forward one's shared first point. They leave in opposite directions.
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
