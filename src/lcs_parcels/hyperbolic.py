"""Hyperbolic LCS as strain tensor lines.

Haller (2015) §5.1 / Table 1 (n=2), doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).

A repelling LCS is a *shrink line* -- a curve tangent to the weak-stretch
eigenvector ``xi_1`` of the Cauchy-Green tensor ``C``, normal to the
strong-stretch ``xi_2`` that the FTLE ridge marks.

Attracting LCS are the shrink lines of the *backward* flow (Haller & Sapsis
2011, doi:10.1063/1.3579597, https://doi.org/10.1063/1.3579597), so
:func:`shrink_lines` of a backward :class:`~lcs_parcels.FlowMap` gives them.

:func:`ftle_ridge_seeds` picks seed points, :func:`shrink_lines` integrates the
tensor lines through them, and :func:`prune_shrink_lines` drops the
near-duplicate lines that several seeds on one ridge produce.

Rectilinear grids only, with a monotonic ``lon_grid`` axis, so a domain crossing
the antimeridian is seeded on ``170, 175, 180, 185`` rather than ``170, 175,
180, -175``. A traced line may still cross the antimeridian.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels._numerics import (
    _LONLAT_ATTRS,
    EARTH_RADIUS_M,
    _eigen_2x2,
    _point_dataset,
    _rk2_step,
    _separation_m,
    _tensor_interp,
    _windowed_extrema,
)


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
    computed on this grid. It bounds *strict* maxima only, so a plateau of
    exactly equal values ties and can return adjacent seeds.

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
        :meth:`FlowMap.ftle`). Mutually exclusive with ``quantile``. Use it to
        hold several windows or regions to one common threshold.

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

    is_peak, geometry = _windowed_extrema(ftle, window_m=window_m, extremum="max")
    return _point_dataset(
        is_peak & (ftle >= threshold), dim="seed", what="FTLE ridge seed"
    ).assign_attrs(
        long_name="seed points at strong local maxima of the FTLE field",
        selector="quantile" if ftle_min is None else "ftle_min",
        ftle_threshold=threshold,
        **geometry,
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

    Returns ``NaN`` wherever the tangent is **not well defined**: off the grid,
    in a NaN cell, or where ``lambda_2 / lambda_1`` is under ``min_anisotropy``.

    Parameters
    ----------
    lon, lat : np.ndarray
        Positions (degrees), shape ``(n,)``.
    heading : np.ndarray
        Shape ``(n, 2)``; the running direction each returned vector is aligned
        with. Only its sign against ``xi_1`` matters.
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
    lambda_1, lambda_2, phi = _eigen_2x2(tensor_interp(np.column_stack([lon, lat])))
    direction = np.column_stack([np.sin(phi), -np.cos(phi)])
    # An eigenvector has no intrinsic sign, so agreeing with the running heading
    # is what keeps the marched sequence continuous instead of zig-zag.
    direction[np.sum(direction * heading, axis=1) < 0] *= -1
    direction[~(lambda_2 >= min_anisotropy * lambda_1)] = np.nan
    return direction


def _trace_half_line(
    seed_lon: np.ndarray,
    seed_lat: np.ndarray,
    sign: int,
    *,
    tensor_interp: RegularGridInterpolator,
    min_anisotropy: float,
    step_m: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """March every seed ``n_steps`` steps in one of the two ``xi_1`` directions.

    Integrates ``dr/ds = xi_1(r)`` with the midpoint (RK2) step of
    :func:`_rk2_step`.

    Parameters
    ----------
    seed_lon, seed_lat : np.ndarray
        Seed positions (degrees), shape ``(n,)``.
    sign : int
        ``+1`` or ``-1``, selecting which of the two opposite ``xi_1`` branches
        this half follows away from the seeds.
    tensor_interp, min_anisotropy, step_m
        As for :func:`_shrink_line_tangent` and :func:`_rk2_step`.
    n_steps : int
        Number of steps taken; the returned track has ``n_steps + 1`` entries,
        the first being the seeds themselves.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lon, lat)`` of shape ``(n_steps + 1, n)``, ordered along the curve
        away from the seed. Terminated lines carry ``NaN`` from the step that
        terminated them onward.
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
    track_lon = np.full((n_steps + 1, lon.size), np.nan)
    track_lat = np.full((n_steps + 1, lon.size), np.nan)
    track_lon[0], track_lat[0] = lon, lat

    def tangent(lon, lat, heading):
        return _shrink_line_tangent(
            lon,
            lat,
            heading,
            tensor_interp=tensor_interp,
            min_anisotropy=min_anisotropy,
        )

    for step in range(1, n_steps + 1):
        lon, lat, heading = _rk2_step(lon, lat, heading, tangent=tangent, step_m=step_m)
        # All seeds march in one array, so a terminated line is NaN-filled rather
        # than broken out of. That costs work on lines that died early, see #11.
        track_lon[step], track_lat[step] = lon, lat
    return track_lon, track_lat


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
    attracting LCS, by the forward-backward duality.

    A line stops where ``xi_1`` stops being well defined, through
    ``min_anisotropy``, off the grid, or in a NaN cell. ``line_length_m`` is a
    budget, so a line that stops early is shorter and NaN-padded.

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
        Length budget for each line in metres (default 1500 km), traced half in
        each direction from the seed. The step count per direction is
        ``line_length_m / (2 * step_m)`` rounded, at least 1, so the traced
        length can overshoot the budget by up to a step. Lines that terminate
        early are shorter.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on dims ``(line, point)``, one ``line`` per
        seed, ordered along the curve. Terminated points are ``NaN``.
    """
    n_steps = max(1, round(line_length_m / (2.0 * step_m)))
    trace_kwargs = {
        "tensor_interp": _tensor_interp(flowmap.cauchy_green()),
        "min_anisotropy": min_anisotropy,
        "step_m": step_m,
        "n_steps": n_steps,
    }
    # Stitch the halves into one curve, reversing the backward one and dropping
    # the forward one's shared first point. They leave in opposite directions.
    back_lon, back_lat = _trace_half_line(seed_lon, seed_lat, -1, **trace_kwargs)
    ahead_lon, ahead_lat = _trace_half_line(seed_lon, seed_lat, +1, **trace_kwargs)
    lon_lines = np.concatenate([back_lon[::-1], ahead_lon[1:]]).T  # (line, point)
    lat_lines = np.concatenate([back_lat[::-1], ahead_lat[1:]]).T
    return xr.Dataset(
        {
            "lon": xr.DataArray(
                lon_lines,
                dims=("line", "point"),
                attrs={"long_name": "longitude along the shrink line"}
                | _LONLAT_ATTRS["lon"],
            ),
            "lat": xr.DataArray(
                lat_lines,
                dims=("line", "point"),
                attrs={"long_name": "latitude along the shrink line"}
                | _LONLAT_ATTRS["lat"],
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


def _unit_sphere_xyz(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """Unit-sphere position of ``lon``/``lat`` (degrees), with an ``xyz`` dim."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    return xr.concat(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        dim="xyz",
    )


def prune_shrink_lines(
    lines: xr.Dataset,
    *,
    ftle: xr.DataArray | None = None,
    window_m: float = 30_000.0,
) -> xr.Dataset:
    """Drop shrink lines that duplicate a stronger one.

    Lines are ranked by the FTLE integrated along them (``ftle_mean`` times
    ``length_m``), or by ``length_m`` alone when no ``ftle`` is given, and walked
    from the strongest down. A line is dropped when it touches the
    ``window_m / 2`` tube of a kept line and adds under ``window_m`` outside it.

    Whole lines are kept or dropped. A line sharing nothing with a stronger line
    is always kept, whatever its length.

    Rows that are NaN at every point are dropped first and cover nothing. FTLE
    interpolated as NaN counts as zero.

    Parameters
    ----------
    lines : xr.Dataset
        A :func:`shrink_lines` dataset, ``lon``/``lat`` on ``(line, point)`` with
        NaN past termination. Other variables on ``line`` pass through.
    ftle : xr.DataArray, optional
        The field the seeds were picked from, on ``(i, j)`` with
        ``lon_grid``/``lat_grid`` coordinates, e.g., from :meth:`FlowMap.ftle`.
    window_m : float, optional
        Resolution below which two lines are one line (default 30 km); the
        ``window_m`` the seeds were picked with. The tube radius is half of it
        and the minimum new length equals it.

    Returns
    -------
    xr.Dataset
        ``lines`` restricted to the kept rows, the ``line`` coord keeping its
        original labels, plus ``length_m`` (m) and, with ``ftle``, ``ftle_mean``
        (1/s) on ``line``. Attributes record ``window_m``, ``tube_radius_m``,
        ``min_new_length_m``, ``n_lines_in`` and ``n_lines_dropped``.
    """
    tube_radius_m = window_m / 2.0
    min_new_length_m = window_m
    n_lines_in = lines.sizes["line"]

    lines = lines.isel(line=lines["lon"].notnull().any("point"))
    lon, lat = lines["lon"], lines["lat"]
    # Segment from each point to the next, NaN at the last point and past
    # termination, so `sum` counts the traced arc only.
    dx, dy = _separation_m(
        lon_a=lon, lat_a=lat, lon_b=lon.shift(point=-1), lat_b=lat.shift(point=-1)
    )
    segment_m = np.hypot(dx, dy)
    length_m = segment_m.sum("point").assign_attrs(
        long_name="arc length of the shrink line", units="m"
    )
    scores = {"length_m": length_m}
    if ftle is not None:
        ftle_axes = ftle.reset_coords(drop=True).assign_coords(
            i=ftle["lon_grid"].isel(j=0).values, j=ftle["lat_grid"].isel(i=0).values
        )
        # `interp` rejects an empty query, which every-row-NaN input produces.
        ftle_along = (
            ftle_axes.interp(i=lon, j=lat).drop_vars(["i", "j"])
            if lon.size
            else xr.zeros_like(lon)
        )
        scores["ftle_mean"] = (
            ftle_along.fillna(0.0)
            .where(lon.notnull())
            .mean("point")
            .assign_attrs(long_name="mean FTLE along the shrink line", units="1/s")
        )
    score = scores["length_m"] * scores.get("ftle_mean", 1.0)

    # `covered[line, other, point]`: the point of `line` lies within the tube of
    # some point of `other`. Compared as an angle on the unit sphere.
    xyz = _unit_sphere_xyz(lon, lat)
    cos_tube = np.cos(tube_radius_m / EARTH_RADIUS_M)
    kept: list = []
    # An input with no traced row has nothing to walk, and `concat` of nothing
    # raises.
    if lines.sizes["line"]:
        covered = xr.concat(
            [
                xr.dot(xyz, other.rename(point="other_point"), dim="xyz").max(
                    "other_point"
                )
                >= cos_tube
                for other in xyz.transpose("line", ...)
            ],
            dim=xyz["line"].rename(line="other"),
        )
    for label in score.sortby(score, ascending=False)["line"].values:
        if kept:
            hit = covered.sel(line=label, other=kept).any("other")
            new_segment = ~hit & ~hit.shift(point=-1, fill_value=True)
            new_length_m = segment_m.sel(line=label).where(new_segment).sum("point")
            if bool(hit.any()) and float(new_length_m) < min_new_length_m:
                continue
        kept.append(label)

    pruned = lines.isel(line=lines["line"].isin(kept))
    return pruned.assign(
        {name: value.sel(line=pruned["line"]) for name, value in scores.items()}
    ).assign_attrs(
        long_name="shrink lines with near-duplicates of a stronger line dropped",
        window_m=float(window_m),
        tube_radius_m=float(tube_radius_m),
        min_new_length_m=float(min_new_length_m),
        n_lines_in=int(n_lines_in),
        n_lines_dropped=int(n_lines_in - len(kept)),
    )
