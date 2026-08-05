"""Hyperbolic LCS as strain tensor lines.

Haller (2015) §5.1 / Table 1 (n=2), doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).

A repelling LCS is a *shrink line* -- a curve tangent to the weak-stretch
eigenvector ``xi_1`` of the Cauchy-Green tensor ``C`` (equivalently, normal to
the strong-stretch ``xi_2`` that the FTLE ridge marks). It solves the tensor-line
ODE ``dr/ds = xi_1(r)``. Attracting LCS need no separate machinery: by the
forward-backward duality (Haller & Sapsis 2011, https://doi.org/10.1063/1.3579597)
they are the shrink lines of the *backward* flow, so :func:`shrink_lines` of a
backward :class:`~lcs_parcels.FlowMap` gives them.

Two functions compose the workflow: :func:`ftle_ridge_seeds` picks seed points,
:func:`shrink_lines` integrates the tensor lines through them. Both take the
gridded xarray outputs of a :class:`~lcs_parcels.FlowMap`, and
:meth:`~lcs_parcels.FlowMap.hyperbolic_lcs` runs the pair in one call.

Rectilinear grids only: like :class:`~lcs_parcels.NeighborFlowMap`, the tensor is
interpolated on axis-aligned ``lon_grid``/``lat_grid`` axes (``lon_grid`` varying
along ``i``, ``lat_grid`` along ``j``). The ``lon_grid`` axis must also be
monotonic, so a domain crossing the antimeridian is seeded on ``170, 175, 180,
185`` rather than ``170, 175, 180, -175``. Only the axis is constrained; a traced
line may cross the antimeridian.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels.grids import _DEG, EARTH_RADIUS_M, _separation_m


def _odd_cells(window_m: float, spacing_m: float) -> int:
    """Cells spanning ``window_m`` at grid spacing ``spacing_m``, odd and at least 1.

    Odd keeps the rolling window centred on its own grid point; rounding down to
    the nearest odd count keeps it from reaching past ``window_m / 2``.
    """
    cells = round(window_m / spacing_m)
    if cells % 2 == 0:
        cells -= 1
    return max(1, cells)


def _window_geometry(ftle: xr.DataArray, window_m: float) -> dict[str, float]:
    """What ``window_m`` actually becomes on this field's grid.

    The rolling window is a count of *cells*, so the two reported quantities read
    the grid's cell sizes differently and both medians and minima are needed.

    ``grid_spacing_i_m`` / ``grid_spacing_j_m`` are the **median** local
    east/north separations of adjacent grid points
    (:func:`~lcs_parcels.grids._separation_m`). They set the cell counts: the
    size that most of the grid has is the right one to convert ``window_m`` with.

    ``min_seed_separation_m`` is the closest two seeds can be, and so is built on
    the **minimum** cell size instead: a window of ``cells`` reaches
    ``(cells - 1) // 2`` cells to either side of its own grid point, so the
    nearest point that can also be a windowed maximum is one cell beyond that,
    and the physical distance that buys is smallest where the cells are smallest.
    On a grid of near-uniform cells the value lands near ``window_m / 2``, which
    is the number a user choosing ``window_m`` needs; on a grid whose cells
    converge poleward it is smaller, and the median would overstate it by a
    factor of 3 over a 75-degree band, measured.

    The bound is on *strict* local maxima. Selection is ``ftle >= rolling max``,
    so every cell of a plateau of exactly equal values is a windowed maximum and
    adjacent cells can both be seeds. A masked, saturated or float-tied field can
    therefore return seeds closer than this. The cells of such a plateau are
    indistinguishable from one another, so all of them are kept.
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

    Warns when ``window_m`` spans fewer than three cells in either dimension: a
    one-cell window makes every point a windowed maximum, so the local-maximum
    test stops selecting anything and only the magnitude floor is left.

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

        A window of side ``window_m`` reaches ``window_m / 2`` to either side of
        its own grid point, so **the closest two seeds can be is about
        ``window_m / 2``**, not ``window_m``. The returned
        ``min_seed_separation_m`` attribute is that distance computed on this
        grid from the smallest cell rather than the typical one, so it is a floor
        on the separation. For a quick estimate halve ``window_m``; for the value
        on this grid read the attribute. The bound holds for *strict* local
        maxima: a plateau of exactly equal values makes every one of its cells a
        windowed maximum, and those can be adjacent.
    quantile : float, optional
        Magnitude floor as a quantile of this field, in ``[0, 1]``. Defaults to
        0.90 (the top decile) when neither selector is given.
    ftle_min : float, optional
        Magnitude floor as an absolute value, in the units of ``ftle`` (1/s for
        :meth:`FlowMap.ftle`). Mutually exclusive with ``quantile``.

        The default selector is the quantile, because a quantile means the same
        thing on any field, while a given rate marks a ridge in a fast flow and
        nothing in a slow one. Use ``ftle_min`` when several windows or regions
        have to be compared against one common threshold, which a quantile cannot
        provide.

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
        raise ValueError(
            "give either quantile or ftle_min, not both: they are two ways of "
            "setting the same magnitude floor"
        )
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
            "local-maximum test stops selecting. Widen window_m or seed a finer "
            "grid.",
            UserWarning,
            stacklevel=2,
        )

    peak = ftle.rolling(i=cells_i, j=cells_j, center=True, min_periods=1).max()
    is_seed = (ftle >= peak) & (ftle >= threshold)
    mask = is_seed.transpose("i", "j").values
    lon = ftle["lon_grid"].transpose("i", "j").values[mask]
    lat = ftle["lat_grid"].transpose("i", "j").values[mask]
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

    ``xi_1`` is the direction material line elements *shrink* along, and the
    shrink line is the curve tangent to it everywhere.

    Returns ``NaN`` wherever the tangent is **not well defined**, which is one
    condition with three causes: the point is off-grid, it sits in a NaN cell, or
    the tensor is too close to isotropic for its eigenvectors to be resolved
    (``min_anisotropy``). Callers read a NaN row as "the line ends here" and do
    not need to know which of the three fired.

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
    # eigh returns eigenvalues ascending: eigenvalues[:, 0] = lambda_1 (the
    # *smaller*, weak-stretch eigenvalue) with eigenvector eigenvectors[:, :, 0]
    # = xi_1, the shrink-line tangent; eigenvalues[:, 1] = lambda_2 = lambda_max.
    # Off-grid points are diagonalised as the identity purely to keep eigh from
    # raising; their rows are overwritten with NaN below.
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.where(terminated[:, None, None], np.eye(2), cauchy_green)
    )
    # Stop where the eigenvalues are too close together to separate: an
    # eigenvector's sensitivity to perturbation of C goes as the inverse of the
    # *relative* gap between the eigenvalues, so the test is on
    # lambda_2 / lambda_1 rather than on lambda_2 alone.
    # C is positive semi-definite, so a lambda_1 at or below zero is round-off on
    # an extremely anisotropic tensor; clamping it to zero lets those points
    # through, which is correct for them.
    terminated = terminated | (
        eigenvalues[:, 1] < min_anisotropy * np.maximum(eigenvalues[:, 0], 0.0)
    )
    direction = eigenvectors[:, :, 0]
    # An eigenvector has no intrinsic sign, so eigh's choice flips arbitrarily
    # between neighbouring points. Flipping each one to the acute side of the
    # running heading keeps the marched sequence a continuous curve instead of a
    # zig-zag.
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
    is built in. This is the exact inverse of the measurement that built it
    (:func:`~lcs_parcels.grids._separation_m`): the northward component is
    ``R d(phi)``, the eastward one ``R cos(phi_mid) d(lambda)`` about the same
    mid-latitude. A unit ``direction`` moves ``step_m``; a shorter one (the
    half-step of the midpoint scheme) moves proportionally less.

    Inverting the measurement, rather than solving the direct great-circle
    problem, keeps the traced curve on the direction field. A great-circle arc
    leaves a heading-invariant field at a rate ``(step / R)**2 tan(phi) / 2`` per
    step, which accumulates *linearly* in the step count: a due-east field at
    70 N drifts 3.4 km off its parallel over an 800 km line at a 20 km step, and
    halving the step only halves that. The step taken here has no such term. It
    follows a field aligned with a parallel or a meridian exactly, at any step
    size and any latitude; on any other heading the ``cos(phi_mid)`` factor is a
    midpoint rule, so the error is second order in the step (0.04 m at 60 N on a
    45-degree heading at a 25 km step, and 2.6 m at 100 km).

    The increment is added to the incoming longitude, so a track crossing the
    antimeridian stays on the branch its seed came in on. Latitude is not folded
    at the pole: a line stepped past 90 degrees runs off the chart rather than
    over the top, and terminates at the next tangent lookup.

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
    m_per_deg = EARTH_RADIUS_M * _DEG
    dlat = direction[:, 1] * step_m / m_per_deg
    lat_mid = lat + 0.5 * dlat
    return (
        lon + direction[:, 0] * step_m / (m_per_deg * np.cos(lat_mid * _DEG)),
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
    # Pick the initial branch by dotting xi_1 against the 45-degree direction --
    # an arbitrary tie-break, since at the seed there is no running heading yet.
    # A seed whose xi_1 lies near the anti-diagonal therefore flips branch on
    # numerical noise. The tie-break itself does not depend on `sign`, so the two
    # halves start from exactly opposite headings.
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
        # The well-definedness guard is evaluated at the RK2 midpoint too, so a
        # step whose two endpoints are both fine still terminates the line if the
        # tensor is degenerate halfway along it.
        mid_direction = _shrink_line_tangent(
            mid_lon,
            mid_lat,
            direction,
            tensor_interp=tensor_interp,
            min_anisotropy=min_anisotropy,
        )
        lon, lat = _step_lonlat_by_meters(lon, lat, mid_direction, step_m=step_m)
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
    min_anisotropy: float = 1.15,
    step_m: float = 3_000.0,
    line_length_m: float = 1_500_000.0,
) -> xr.Dataset:
    """Integrate shrink lines (``xi_1`` tensor lines) through the seed points.

    Traces the tensor-line ODE ``dr/ds = xi_1(r)`` both ways from each seed,
    where ``xi_1`` is the weak-stretch eigenvector of ``flowmap.cauchy_green()``.
    A *forward* flow map yields repelling LCS; a *backward* one yields attracting
    LCS (forward-backward duality). The integrator:

    - interpolates the tensor ``C`` (not the eigenvector) and re-diagonalises at
      each point, so it stays smooth through the near-degenerate
      ``lambda_1 ~ lambda_2`` spots where ``xi_1`` is otherwise sign-ambiguous;
    - orients each step to the running heading (an eigenvector has no intrinsic
      sign);
    - stops a line where ``xi_1`` stops being well defined -- ``min_anisotropy``,
      off the grid, or a NaN cell.

    Marches all seeds together with a midpoint (arc-length) step. ``line_length_m``
    is a *cap*: a line that terminates early is shorter, and the returned block is
    NaN-filled past termination so every row has the same length.

    Parameters
    ----------
    flowmap : FlowMap
        Advected flow map on a rectilinear grid; supplies ``cauchy_green()`` and
        the ``lon_grid``/``lat_grid`` axes.
    seed_lon, seed_lat : array_like
        Seed positions (degrees), e.g., from :func:`ftle_ridge_seeds`.
    min_anisotropy : float, optional
        Stop a line where ``lambda_2 / lambda_1`` falls below this (default
        1.15). An eigenvector's sensitivity to perturbation of ``C`` scales as
        the inverse of the *relative* gap between the eigenvalues, so this ratio,
        rather than ``lambda_2`` alone, decides whether ``xi_1`` is resolved; at
        the default a 1% error in ``C`` swings ``xi_1`` by about 2 degrees.
        Being a ratio it does not depend on ``T``, on the grid scale, or on the
        flow's own stretching rate, so the same value applies to a six-hour
        laboratory flow and to a six-month basin-scale one. It is a
        well-definedness guard; which structures count as LCS is set by
        ``quantile`` in :func:`ftle_ridge_seeds`.
    step_m : float, optional
        Arc-length step in metres (default 3000).
    line_length_m : float, optional
        Maximum length of each line in metres (default 1500 km), traced half in
        each direction from the seed; the step count per direction is
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
    # CG_grid (not "C-grid": no Arakawa staggering here) is the Cauchy-Green
    # tensor on the diagnostic grid, interpolated point-by-point during the trace.
    CG_grid = flowmap.cauchy_green().transpose("i", "j", "row", "col").values
    # The one place this package leaves the label-based xarray API: the ODE loop
    # below evaluates the tensor at millions of scattered points, which
    # ``.interp()`` cannot do without building an xarray object per step.
    tensor_interp = RegularGridInterpolator(
        (lon_axis, lat_axis), CG_grid, bounds_error=False, fill_value=np.nan
    )

    trace_kwargs = {
        "tensor_interp": tensor_interp,
        "min_anisotropy": min_anisotropy,
        "step_m": step_m,
        "n_steps": n_steps,
    }
    # Trace both ways from each seed and stitch into one curve through it: the
    # backward half reversed (so it runs into the seed), then the forward half
    # with its first point (the seed, shared) dropped. Both halves get their
    # initial heading from the same branch pick with opposite sign, so they leave
    # the seed in exactly opposite directions and the stitch is continuous to the
    # order of the scheme.
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
