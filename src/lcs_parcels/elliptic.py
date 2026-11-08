"""Elliptic LCS as closed shear lines.

Haller & Beron-Vera (2013), doi:10.1017/jfm.2013.391
(https://doi.org/10.1017/jfm.2013.391).

A coherent Lagrangian vortex boundary is a *closed shear line* -- a closed curve
tangent everywhere to one of the direction fields of the Cauchy-Green tensor
``C`` (Eq. 14)

    eta^pm_lambda = sqrt((lambda_2 - lambda^2) / (lambda_2 - lambda_1)) xi_1
                    pm sqrt((lambda^2 - lambda_1) / (lambda_2 - lambda_1)) xi_2

Every tangent element of such a curve is stretched by the same factor
``lambda`` over the window. ``eta^pm_lambda`` exists only where
``lambda_1 < lambda^2 < lambda_2``.

:func:`elliptic_centres` picks candidate centres from a field,
:func:`closed_shear_lines` searches them for closed orbits over a scan of
``lambda``, :func:`outermost_shear_lines` keeps one boundary per vortex, and
:func:`rotation_sense` reads each boundary's sense of rotation.

The stretching factor ``lambda`` is stored as ``stretch``. Rectilinear grids
only.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import xarray as xr

from lcs_parcels._numerics import (
    _LONLAT_ATTRS,
    _M_PER_DEG,
    _eigen_2x2,
    _grid_spacing_m,
    _point_dataset,
    _rk2_step,
    _separation_m,
    _tensor_interp,
    _windowed_extrema,
    _wrap_lon,
)

#: Lines traced together in one array. Bounds the memory of the search.
_LINES_PER_CHUNK = 200_000

#: Metadata of the per-orbit variables of :func:`closed_shear_lines`.
_ORBIT_ATTRS = {
    "centre_index": {
        "long_name": "index of the candidate centre searched",
        "units": "1",
    },
    "centre_lon": {"long_name": "longitude of the candidate centre searched"}
    | _LONLAT_ATTRS["lon"],
    "centre_lat": {"long_name": "latitude of the candidate centre searched"}
    | _LONLAT_ATTRS["lat"],
    "stretch": {
        "long_name": "uniform stretching factor lambda of the curve",
        "units": "1",
    },
    "branch": {"long_name": "branch of eta^pm_lambda, +1 or -1", "units": "1"},
    "area_m2": {"long_name": "area enclosed by the closed shear line", "units": "m2"},
    "radius_m": {
        "long_name": "equivalent radius sqrt(area / pi) of the closed shear line",
        "units": "m",
    },
    "centroid_lon": {"long_name": "longitude of the closed shear line centroid"}
    | _LONLAT_ATTRS["lon"],
    "centroid_lat": {"long_name": "latitude of the closed shear line centroid"}
    | _LONLAT_ATTRS["lat"],
    "residual_m": {
        "long_name": "distance from launch to return on the section",
        "units": "m",
    },
    "at_scan_edge": {
        "long_name": "closed at the first or last stretching factor of the scan",
        "units": "1",
    },
}


def _eta_tangent(
    lon: np.ndarray,
    lat: np.ndarray,
    heading: np.ndarray,
    *,
    stretch: np.ndarray,
    branch: np.ndarray,
    tensor_interp,
) -> np.ndarray:
    """Unit ``eta^branch_stretch`` at each ``(lon, lat)``, oriented to ``heading``.

    ``stretch`` and ``branch`` (``+1`` or ``-1``) are per point, shape ``(n,)``.
    Returns NaN rows off the grid, in a NaN cell, and wherever
    ``lambda_1 < stretch^2 < lambda_2`` fails.
    """
    lambda_1, lambda_2, phi = _eigen_2x2(tensor_interp(np.column_stack([lon, lat])))
    stretch_sq = stretch**2
    # xi_1 is xi_2 turned clockwise, so negating one negates the other and the
    # branches cannot swap when eta is oriented to the heading.
    xi_2 = np.column_stack([np.cos(phi), np.sin(phi)])
    xi_1 = np.column_stack([np.sin(phi), -np.cos(phi)])
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.sqrt((lambda_2 - stretch_sq) / (lambda_2 - lambda_1))
        b = np.sqrt((stretch_sq - lambda_1) / (lambda_2 - lambda_1))
    eta = a[:, None] * xi_1 + (branch * b)[:, None] * xi_2
    eta[np.sum(eta * heading, axis=1) < 0] *= -1
    eta[~((lambda_1 < stretch_sq) & (stretch_sq < lambda_2))] = np.nan
    return eta


def stretch_range(*, stretch_max: float = 1.5, step: float = 0.03) -> np.ndarray:
    """Stretching factors log-symmetric about 1, at most ``stretch_max`` and at
    least ``1 / stretch_max``.

    The values are ``exp(k * step)`` for integer ``k``, so 1 is always included
    and the spacing is ``step`` in ``ln lambda``, about ``step`` near 1.

    Raises
    ------
    ValueError
        If ``stretch_max`` is not above 1 or ``step`` is not positive.
    """
    if stretch_max <= 1.0:
        raise ValueError(f"stretch_max must exceed 1, got {stretch_max}")
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    k_max = int(np.floor(np.log(stretch_max) / step + 1e-9))
    return np.exp(step * np.arange(-k_max, k_max + 1))


def elliptic_centres(
    field: xr.DataArray,
    *,
    extremum: str,
    window_m: float = 100_000.0,
    edge_m: float | None = None,
) -> xr.Dataset:
    """Candidate vortex centres at windowed extrema of a field.

    A grid point is a centre when it is the minimum (``extremum="min"``) or
    maximum (``"max"``) of ``field`` over a square window of side ``window_m``.
    Warns when the window spans fewer than three cells.

    Points within ``edge_m`` (default ``window_m / 2``) of the domain edge or of
    a NaN cell never qualify.

    The field must carry ``lon_grid``/``lat_grid``, as every
    :class:`~lcs_parcels.FlowMap` diagnostic does.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on a ``centre`` dim. The ``attrs`` record
        ``extremum``, ``window_m``, ``edge_m`` and the cell counts they became.
    """
    is_extremum, geometry = _windowed_extrema(
        field, window_m=window_m, extremum=extremum
    )
    edge_m = 0.5 * window_m if edge_m is None else edge_m
    edge_cells_i = int(np.ceil(edge_m / geometry["grid_spacing_i_m"]))
    edge_cells_j = int(np.ceil(edge_m / geometry["grid_spacing_j_m"]))
    # An incomplete window is NaN, so the domain edge counts as a NaN cell.
    clear_of_nan = (
        field.notnull()
        .astype(float)
        .rolling(i=2 * edge_cells_i + 1, j=2 * edge_cells_j + 1, center=True)
        .min()
        .fillna(0.0)
        > 0.5
    )
    return _point_dataset(
        is_extremum & clear_of_nan, dim="centre", what="candidate vortex centre"
    ).assign_attrs(
        extremum=extremum,
        edge_m=float(edge_m),
        edge_cells_i=edge_cells_i,
        edge_cells_j=edge_cells_j,
        **{k: v for k, v in geometry.items() if k != "min_seed_separation_m"},
    )


def _trace_to_return(
    s: np.ndarray,
    family: dict,
    *,
    tensor_interp,
    section_m: float,
    step_m: float,
    n_steps: int,
    keep_track: bool = False,
):
    """Trace ``eta`` from ``s`` metres along each line's section until it returns.

    ``family`` holds per-line ``centre_lon``, ``centre_lat``, ``direction``
    (``+1`` east, ``-1`` west), ``stretch`` and ``branch``. A line *returns* at
    its first crossing of its section after more than half a turn.

    A line stops without a return where the field ends, where it crosses the
    section before half a turn, or after one and a half turns.

    Returns
    -------
    offset : np.ndarray
        Return position minus launch position ``P(s) - s`` in metres, NaN where
        the line did not return.
    track : tuple of np.ndarray or None
        With ``keep_track``, ``(lon, lat)`` of shape ``(n_steps + 2, n)``: each
        line from its launch to its return and back to its launch, NaN after.
    """
    n = s.size
    cos_centre = np.cos(np.deg2rad(family["centre_lat"]))
    lon = family["centre_lon"] + family["direction"] * s / (_M_PER_DEG * cos_centre)
    lat = family["centre_lat"].astype(float)
    dx, dy = _separation_m(
        lon_a=family["centre_lon"], lat_a=family["centre_lat"], lon_b=lon, lat_b=lat
    )
    # No running heading at launch, so the direction of travel comes from the
    # 45-degree direction. Either way round traces the same loop.
    heading = _eta_tangent(
        lon,
        lat,
        np.ones((n, 2)),
        stretch=family["stretch"],
        branch=family["branch"],
        tensor_interp=tensor_interp,
    )
    state = family | {
        "line": np.arange(n),
        "s": s,
        "cos_centre": cos_centre,
        "lon": lon,
        "lat": lat,
        "heading": heading,
        "angle": np.arctan2(dy, dx),
        "turns": np.zeros(n),
        "off_section": np.zeros(n, dtype=bool),
    }
    traceable = np.isfinite(heading).all(axis=1)
    state = {key: value[traceable] for key, value in state.items()}
    offset = np.full(n, np.nan)
    track = (np.full((n_steps + 2, n), np.nan), np.full((n_steps + 2, n), np.nan))
    track[0][0], track[1][0] = lon, lat

    step = 0
    while state["line"].size and step < n_steps:
        step += 1
        tangent = partial(
            _eta_tangent,
            stretch=state["stretch"],
            branch=state["branch"],
            tensor_interp=tensor_interp,
        )
        lon, lat, heading = _rk2_step(
            state["lon"], state["lat"], state["heading"], tangent=tangent, step_m=step_m
        )
        ended = ~np.isfinite(heading).all(axis=1)
        dx, dy = _separation_m(
            lon_a=state["centre_lon"], lat_a=state["centre_lat"], lon_b=lon, lat_b=lat
        )
        angle = np.arctan2(dy, dx)
        turns = state["turns"] + _wrap_lon(np.rad2deg(angle - state["angle"])) / 360.0

        north_old = state["lat"] - state["centre_lat"]
        north_new = lat - state["centre_lat"]
        crossed = state["off_section"] & (north_old * north_new <= 0.0) & ~ended
        fraction = np.where(crossed, north_old / (north_old - north_new + 1e-300), 0)
        lon_cross = state["lon"] + fraction * _wrap_lon(lon - state["lon"])
        s_cross = (
            state["direction"]
            * _wrap_lon(lon_cross - state["centre_lon"])
            * _M_PER_DEG
            * state["cos_centre"]
        )
        on_section = crossed & (s_cross >= 0.0) & (s_cross <= section_m)
        returned = on_section & (np.abs(turns) > 0.5)
        offset[state["line"][returned]] = (s_cross - state["s"])[returned]
        if keep_track:
            line = state["line"][~ended]
            track[0][step, line] = np.where(returned, lon_cross, lon)[~ended]
            track[1][step, line] = np.where(returned, state["centre_lat"], lat)[~ended]
            closed = state["line"][returned]
            track[0][step + 1, closed] = track[0][0, closed]
            track[1][step + 1, closed] = track[1][0, closed]

        state |= {
            "lon": lon,
            "lat": lat,
            "heading": heading,
            "angle": angle,
            "turns": turns,
            "off_section": state["off_section"] | (north_new != 0.0),
        }
        active = ~(ended | on_section | (np.abs(turns) > 1.5))
        state = {key: value[active] for key, value in state.items()}

    return offset, track if keep_track else None


def _area_and_centroid(x, y):
    """Shoelace area (m2) and centroid (m) of closed polygons along axis 0.

    Polygons in metres, NaN-padded past their closing point.
    """
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * np.nansum(cross, axis=0)
    centroid_x = np.nansum((x[:-1] + x[1:]) * cross, axis=0) / (6.0 * signed_area)
    centroid_y = np.nansum((y[:-1] + y[1:]) * cross, axis=0) / (6.0 * signed_area)
    return np.abs(signed_area), centroid_x, centroid_y


def _closing_launches(family, *, s_launch, trace, closure_tol_m):
    """Launch positions whose lines close, per family member.

    Returns the family index and the launch ``s`` of each: the launches returning
    within ``closure_tol_m``, and the bisected sign changes of ``P(s) - s``.
    """
    n_family = family["stretch"].size
    lines = {key: np.repeat(value, s_launch.size) for key, value in family.items()}
    offset = _trace_to_return(np.tile(s_launch, n_family), lines, **trace)[0]
    offset = offset.reshape(n_family, s_launch.size)

    # A launch returning within tolerance is an orbit, since on an axisymmetric
    # vortex the offset never changes sign.
    hit = np.abs(offset) < closure_tol_m
    hit_member, hit_k = np.nonzero(hit)
    sign_change = (
        (np.sign(offset[:, :-1]) * np.sign(offset[:, 1:]) < 0)
        & ~hit[:, :-1]
        & ~hit[:, 1:]
    )
    member, left = np.nonzero(sign_change)
    s_lo, s_hi, f_lo = s_launch[left], s_launch[left + 1], offset[member, left]
    while member.size and (s_hi - s_lo).max() > 0.25 * closure_tol_m:
        s_mid = 0.5 * (s_lo + s_hi)
        bracket = {key: value[member] for key, value in family.items()}
        f_mid = _trace_to_return(s_mid, bracket, **trace)[0]
        lower = np.sign(f_mid) == np.sign(f_lo)
        s_lo, f_lo = np.where(lower, s_mid, s_lo), np.where(lower, f_mid, f_lo)
        s_hi = np.where(lower, s_hi, s_mid)
        # A midpoint that does not return straddles a gap in the return map
        # rather than a zero of it.
        alive = np.isfinite(f_mid)
        member, s_lo, s_hi, f_lo = member[alive], s_lo[alive], s_hi[alive], f_lo[alive]
    return (
        np.concatenate([member, hit_member]),
        np.concatenate([0.5 * (s_lo + s_hi), s_launch[hit_k]]),
    )


def _orbit_columns(s, family, *, trace, closure_tol_m):
    """Re-trace the launches ``s`` with tracks and keep the ones that close."""
    offset, (track_lon, track_lat) = _trace_to_return(
        s, family, **trace, keep_track=True
    )
    closed = np.abs(offset) < closure_tol_m
    centre_lon, centre_lat = family["centre_lon"][closed], family["centre_lat"][closed]
    track_lon, track_lat = track_lon[:, closed], track_lat[:, closed]
    x, y = _separation_m(
        lon_a=centre_lon, lat_a=centre_lat, lon_b=track_lon, lat_b=track_lat
    )
    area, centroid_x, centroid_y = _area_and_centroid(x, y)
    return {key: value[closed] for key, value in family.items()} | {
        "lon": track_lon.T,
        "lat": track_lat.T,
        "area_m2": area,
        "radius_m": np.sqrt(area / np.pi),
        "residual_m": np.abs(offset[closed]),
        "centroid_lon": centre_lon
        + centroid_x / (_M_PER_DEG * np.cos(np.deg2rad(centre_lat))),
        "centroid_lat": centre_lat + centroid_y / _M_PER_DEG,
    }


def _orbit_dataset(columns: dict, *, attrs: dict) -> xr.Dataset:
    """The per-orbit ``columns`` as a dataset on ``(orbit, point)``, trimmed to the
    longest curve."""
    return xr.Dataset(
        {
            "lon": (
                ("orbit", "point"),
                columns["lon"],
                {"long_name": "longitude along the closed shear line"}
                | _LONLAT_ATTRS["lon"],
            ),
            "lat": (
                ("orbit", "point"),
                columns["lat"],
                {"long_name": "latitude along the closed shear line"}
                | _LONLAT_ATTRS["lat"],
            ),
        }
        | {name: ("orbit", columns[name], meta) for name, meta in _ORBIT_ATTRS.items()},
        coords={
            "orbit": (
                "orbit",
                np.arange(columns["stretch"].size),
                {"long_name": "closed shear line index"},
            ),
            "point": (
                "point",
                np.arange(columns["lon"].shape[1]),
                {"long_name": "point index along the closed shear line"},
            ),
        },
        attrs=attrs,
    ).dropna("point", how="all", subset=["lon"])


def closed_shear_lines(
    flowmap,
    *,
    centre_lon,
    centre_lat,
    stretches=None,
    max_radius_m: float = 150_000.0,
    launch_spacing_m: float | None = None,
    step_m: float | None = None,
    closure_tol_m: float | None = None,
) -> xr.Dataset:
    """Closed ``eta_lambda`` orbits about each candidate centre.

    For every centre, stretching factor and branch, ``eta^pm_lambda`` lines are
    launched every ``launch_spacing_m`` from two sections, due east and due west
    of the centre over ``max_radius_m``.

    An orbit is a launch that turns once about its centre and returns within
    ``closure_tol_m`` of itself, found directly or by bisecting a sign change of
    the return map ``P(s) - s``. Every orbit found is returned.

    Parameters
    ----------
    flowmap : FlowMap
        Flow map on a rectilinear grid.
    centre_lon, centre_lat : array_like
        Candidate centres (degrees), e.g., from :func:`elliptic_centres`.
    stretches : array_like, optional
        Stretching factors to scan, default :func:`stretch_range`.
    max_radius_m : float, optional
        Length of each section, the largest orbit radius searched (default
        150 km). Lines are traced for at most two circumferences at this radius.
    launch_spacing_m, step_m, closure_tol_m : float, optional
        Launch spacing along the sections, arc-length step, and the largest
        accepted launch-to-return distance, all in metres. They default to the
        smaller median grid spacing, half of it, and half of it.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on ``(orbit, point)``, each row a closed curve
        NaN-padded past its end. Per orbit: the ``centre_index`` searched with
        its ``centre_lon``/``centre_lat``, ``stretch``, ``branch``, ``area_m2``,
        ``radius_m`` (equivalent radius), ``centroid_lon``/``centroid_lat``,
        ``residual_m`` and ``at_scan_edge``. The ``attrs`` record the scan and
        the ``grid_spacing_m`` it derived from.

    Raises
    ------
    ValueError
        If ``centre_lon`` and ``centre_lat`` differ in length.
    """
    centre_lon = np.asarray(centre_lon, dtype=float).ravel()
    centre_lat = np.asarray(centre_lat, dtype=float).ravel()
    if centre_lon.shape != centre_lat.shape:
        raise ValueError("centre_lon and centre_lat must have the same length")
    stretches = (
        stretch_range()
        if stretches is None
        else np.asarray(stretches, dtype=float).ravel()
    )
    dx, dy = _grid_spacing_m(flowmap.lon_grid)
    spacing_m = min(float(dx.median()), float(dy.median()))
    launch_spacing_m = spacing_m if launch_spacing_m is None else launch_spacing_m
    step_m = 0.5 * spacing_m if step_m is None else step_m
    closure_tol_m = 0.5 * spacing_m if closure_tol_m is None else closure_tol_m
    s_launch = launch_spacing_m * np.arange(
        1, int(np.floor(max_radius_m / launch_spacing_m)) + 1
    )
    trace = {
        "tensor_interp": _tensor_interp(flowmap.cauchy_green()),
        "section_m": max_radius_m,
        "step_m": step_m,
        "n_steps": int(np.ceil(2.0 * 2.0 * np.pi * max_radius_m / step_m)),
    }

    # One family member per (centre, section, stretch, branch), each launched at
    # every s_launch.
    c, d, k, b = (
        index.ravel()
        for index in np.meshgrid(
            np.arange(centre_lon.size),
            [0, 1],
            np.arange(stretches.size),
            [0, 1],
            indexing="ij",
        )
    )
    family = {
        "centre_index": c,
        "centre_lon": centre_lon[c],
        "centre_lat": centre_lat[c],
        "direction": np.array([1.0, -1.0])[d],
        "stretch": stretches[k],
        "branch": np.array([1, -1])[b],
    }
    per_chunk = max(1, _LINES_PER_CHUNK // max(1, s_launch.size))
    chunks = []
    # At least one chunk, so an empty search still returns every column.
    for start in range(0, max(c.size, 1), per_chunk):
        chunk = {key: value[start : start + per_chunk] for key, value in family.items()}
        member, s = _closing_launches(
            chunk, s_launch=s_launch, trace=trace, closure_tol_m=closure_tol_m
        )
        chunks.append(
            _orbit_columns(
                s,
                {key: value[member] for key, value in chunk.items()},
                trace=trace,
                closure_tol_m=closure_tol_m,
            )
        )
    columns = {
        key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]
    }
    columns["at_scan_edge"] = np.isin(
        columns["stretch"], [stretches.min(), stretches.max()]
    )
    return _orbit_dataset(
        columns,
        attrs={
            "stretch_min": float(stretches.min()),
            "stretch_max": float(stretches.max()),
            "n_stretch": int(stretches.size),
            "max_radius_m": float(max_radius_m),
            "launch_spacing_m": float(launch_spacing_m),
            "step_m": float(step_m),
            "closure_tol_m": float(closure_tol_m),
            "grid_spacing_m": spacing_m,
            "n_centres": int(centre_lon.size),
        },
    )


def _points_inside(x_poly, y_poly, *, x, y):
    """Even-odd test of points ``(x, y)`` against a closed polygon, all in metres."""
    x1, y1 = x_poly[:-1, None], y_poly[:-1, None]
    x2, y2 = x_poly[1:, None], y_poly[1:, None]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (straddles & (x < x_cross)).sum(axis=0) % 2 == 1


def outermost_shear_lines(orbits: xr.Dataset) -> xr.Dataset:
    """The outermost closed shear line of each vortex.

    Keeps the largest-area orbit per candidate centre, then drops any whose
    centre lies inside a larger kept boundary, so several candidate centres in
    one vortex give one eddy.

    Returns
    -------
    xr.Dataset
        The kept rows of ``orbits``, keeping their ``orbit`` labels.
    """
    order = np.argsort(-orbits["area_m2"].values, kind="stable")
    _, first = np.unique(orbits["centre_index"].values[order], return_index=True)
    kept = []
    for k in order[np.sort(first)]:
        row = orbits.isel(orbit=k)
        centre_lon, centre_lat = float(row["centre_lon"]), float(row["centre_lat"])
        if not any(
            _points_inside(
                *_separation_m(
                    lon_a=centre_lon,
                    lat_a=centre_lat,
                    lon_b=other["lon"].dropna("point").values,
                    lat_b=other["lat"].dropna("point").values,
                ),
                x=np.zeros(1),
                y=np.zeros(1),
            )[0]
            for other in kept
        ):
            kept.append(row)
    return (
        orbits.sel(orbit=np.array([row["orbit"].item() for row in kept], dtype=int))
        .dropna("point", how="all", subset=["lon"])
        .assign_attrs(
            n_orbits_in=int(orbits.sizes["orbit"]),
            n_centres_with_orbits=int(first.size),
        )
    )


def rotation_sense(boundaries: xr.Dataset, *, rotation: xr.DataArray) -> xr.Dataset:
    """Mean polar rotation angle inside each closed boundary, and its sign.

    ``boundaries`` holds closed curves as ``lon``/``lat`` on a ``point`` dim, such
    as :func:`outermost_shear_lines` or an image of it. ``rotation`` is the
    :meth:`~lcs_parcels.FlowMap.polar_rotation` of a window short enough that
    each eddy turns less than half a turn.

    Returns
    -------
    xr.Dataset
        ``boundaries`` plus ``rotation_rad``, the mean of ``rotation`` over the
        grid points inside each curve, and ``rotation_sense``, its sign, ``+1``
        counter-clockwise. Both are NaN where no finite grid point lies inside.
    """
    lon_grid = rotation["lon_grid"].values.ravel()
    lat_grid = rotation["lat_grid"].values.ravel()
    theta = rotation.transpose(*rotation["lon_grid"].dims).values.ravel()
    curve_lon = boundaries["lon"].transpose(..., "point")
    curve_lat = boundaries["lat"].transpose(*curve_lon.dims)
    template = curve_lon.sum("point").reset_coords(drop=True)
    n_points = curve_lon.sizes["point"]
    mean = np.full(template.size, np.nan)
    for n, (lon, lat) in enumerate(
        zip(
            curve_lon.values.reshape(template.size, n_points),
            curve_lat.values.reshape(template.size, n_points),
            strict=True,
        )
    ):
        finite = np.isfinite(lon) & np.isfinite(lat)
        if not finite.any():
            continue
        # Closed again in case NaN took the first or last point.
        lon = np.append(lon[finite], lon[finite][0])
        lat = np.append(lat[finite], lat[finite][0])
        x, y = _separation_m(lon_a=lon[0], lat_a=lat[0], lon_b=lon, lat_b=lat)
        gx, gy = _separation_m(
            lon_a=lon[0], lat_a=lat[0], lon_b=lon_grid, lat_b=lat_grid
        )
        near = (
            np.isfinite(theta)
            & (gx >= x.min())
            & (gx <= x.max())
            & (gy >= y.min())
            & (gy <= y.max())
        )
        inside = np.flatnonzero(near)[_points_inside(x, y, x=gx[near], y=gy[near])]
        if inside.size:
            mean[n] = theta[inside].mean()
    rotation_rad = template.copy(data=mean.reshape(template.shape))
    return boundaries.assign(
        rotation_rad=rotation_rad.assign_attrs(
            long_name="mean polar rotation angle inside the boundary", units="rad"
        ),
        rotation_sense=np.sign(rotation_rad).assign_attrs(
            long_name="rotation sense, +1 counter-clockwise, -1 clockwise", units="1"
        ),
    )
