"""Elliptic LCS as closed shear lines.

Haller & Beron-Vera (2013), doi:10.1017/jfm.2013.391
(https://doi.org/10.1017/jfm.2013.391).

A coherent Lagrangian vortex boundary is a *closed shear line* -- a closed curve
tangent everywhere to one of the direction fields

    eta^pm_lambda = sqrt((lambda_2 - lambda^2) / (lambda_2 - lambda_1)) xi_1
                    pm sqrt((lambda^2 - lambda_1) / (lambda_2 - lambda_1)) xi_2

of the Cauchy-Green tensor ``C`` (Eq. 14). Every tangent element of such a
curve is stretched by the same factor ``lambda`` over the window, so the curve
returns to its initial arc length times ``lambda``. ``eta^pm_lambda`` exists
only where ``lambda_1 < lambda^2 < lambda_2``.

The workflow is :func:`elliptic_centres`, which picks candidate centres from a
field, :func:`closed_shear_lines`, which searches each centre for closed orbits
over a scan of ``lambda`` (:func:`stretch_range` by default), and
:func:`outermost_shear_lines`, which keeps the outermost orbit per vortex.
:meth:`~lcs_parcels.FlowMap.elliptic_lcs` runs them in one call.

``lambda`` here is the uniform stretching factor of the curve. It is stored as
``stretch``, since ``lambda`` is the name of the Cauchy-Green eigenvalues in
:meth:`~lcs_parcels.FlowMap.cg_eigen`.

Rectilinear grids only, as in :mod:`lcs_parcels.tensorlines`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from lcs_parcels.grids import _M_PER_DEG, _separation_m, _wrap_lon
from lcs_parcels.tensorlines import _rk2_step, _tensor_interp, _window_geometry

#: Lines traced together in one array. Bounds the memory of the search.
_LINES_PER_CHUNK = 200_000

_LONLAT_ATTRS = {
    "lon": {"units": "degrees_east"},
    "lat": {"units": "degrees_north"},
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
    C = tensor_interp(np.column_stack([lon, lat]))
    terminated = ~np.isfinite(C).all(axis=(1, 2))
    C = np.where(terminated[:, None, None], np.eye(2), C)
    c_xx, c_xy, c_yy = C[:, 0, 0], C[:, 0, 1], C[:, 1, 1]
    half_trace = 0.5 * (c_xx + c_yy)
    half_gap = np.hypot(0.5 * (c_xx - c_yy), c_xy)
    lambda_1 = np.maximum(half_trace - half_gap, 0.0)
    lambda_2 = half_trace + half_gap
    stretch_sq = np.asarray(stretch, dtype=float) ** 2
    terminated |= ~((lambda_1 < stretch_sq) & (stretch_sq < lambda_2))

    # xi_2 at angle phi, and xi_1 as xi_2 turned 90 degrees clockwise, so xi_2 is
    # always xi_1 turned counter-clockwise and the two branches cannot swap.
    phi = 0.5 * np.arctan2(2.0 * c_xy, c_xx - c_yy)
    xi_2 = np.column_stack([np.cos(phi), np.sin(phi)])
    xi_1 = np.column_stack([np.sin(phi), -np.cos(phi)])
    gap = np.maximum(lambda_2 - lambda_1, 1e-300)
    a = np.sqrt(np.clip((lambda_2 - stretch_sq) / gap, 0.0, 1.0))
    b = np.sqrt(np.clip((stretch_sq - lambda_1) / gap, 0.0, 1.0))
    eta = a[:, None] * xi_1 + (np.asarray(branch) * b)[:, None] * xi_2
    # Negating xi_1 negates xi_2 with it, so eta is fixed up to one overall sign,
    # and orienting it to the heading stays on one branch.
    eta[np.sum(eta * heading, axis=1) < 0] *= -1
    eta[terminated] = np.nan
    return eta


def stretch_range(*, stretch_max: float = 1.5, step: float = 0.03) -> np.ndarray:
    """Stretching factors log-symmetric about 1, from ``1 / stretch_max`` up.

    The values are ``exp(k * step)`` for integer ``k``, so 1 is always included
    and the spacing is ``step`` in ``ln lambda``, about ``step`` near 1. The
    largest value is at most ``stretch_max``.

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
    maximum (``"max"``) of ``field`` over a square window of side ``window_m``
    metres. Points within ``edge_m`` (default ``window_m / 2``) of the domain
    edge or of a NaN cell never qualify.

    The field must carry ``lon_grid``/``lat_grid``, as every
    :class:`~lcs_parcels.FlowMap` diagnostic does.

    Returns
    -------
    xr.Dataset
        ``lon``/``lat`` (degrees) on a ``centre`` dim. The ``attrs`` record
        ``extremum``, ``window_m``, ``edge_m`` and the cell counts they became.
    """
    if extremum not in ("min", "max"):
        raise ValueError(f'extremum must be "min" or "max", got {extremum!r}')
    edge_m = 0.5 * window_m if edge_m is None else edge_m
    geometry = _window_geometry(field, window_m)
    rolling = field.rolling(
        i=geometry["window_cells_i"],
        j=geometry["window_cells_j"],
        center=True,
        min_periods=1,
    )
    is_extremum = (
        field <= rolling.min() if extremum == "min" else field >= rolling.max()
    )

    edge_cells_i = int(np.ceil(edge_m / geometry["grid_spacing_i_m"]))
    edge_cells_j = int(np.ceil(edge_m / geometry["grid_spacing_j_m"]))
    clear_of_nan = (
        field.notnull()
        .astype(float)
        .rolling(i=2 * edge_cells_i + 1, j=2 * edge_cells_j + 1, center=True)
        .min()
        .fillna(0.0)
        > 0.5
    )
    i_index = xr.DataArray(np.arange(field.sizes["i"]), dims="i")
    j_index = xr.DataArray(np.arange(field.sizes["j"]), dims="j")
    inside = (
        (i_index >= edge_cells_i)
        & (i_index < field.sizes["i"] - edge_cells_i)
        & (j_index >= edge_cells_j)
        & (j_index < field.sizes["j"] - edge_cells_j)
    )
    keep = is_extremum & field.notnull() & clear_of_nan & inside
    picked = (
        xr.Dataset({"lon": field["lon_grid"], "lat": field["lat_grid"]})
        .reset_coords(drop=True)
        .stack(centre=("i", "j"))
        .where(keep.reset_coords(drop=True).stack(centre=("i", "j")), drop=True)
    )
    lon, lat = picked["lon"].values, picked["lat"].values
    return xr.Dataset(
        {
            "lon": (
                "centre",
                lon,
                {"long_name": "longitude of the candidate vortex centre"}
                | _LONLAT_ATTRS["lon"],
            ),
            "lat": (
                "centre",
                lat,
                {"long_name": "latitude of the candidate vortex centre"}
                | _LONLAT_ATTRS["lat"],
            ),
        },
        coords={
            "centre": (
                "centre",
                np.arange(lon.size),
                {"long_name": "candidate vortex centre index"},
            )
        },
        attrs={
            "extremum": extremum,
            "edge_m": float(edge_m),
            "edge_cells_i": edge_cells_i,
            "edge_cells_j": edge_cells_j,
        }
        | {k: v for k, v in geometry.items() if k != "min_seed_separation_m"},
    )


def _trace_to_return(
    lon_0: np.ndarray,
    lat_0: np.ndarray,
    *,
    centre_lon: np.ndarray,
    centre_lat: np.ndarray,
    direction: np.ndarray,
    stretch: np.ndarray,
    branch: np.ndarray,
    section_m: float,
    tensor_interp,
    step_m: float,
    n_steps: int,
    keep_track: bool = False,
):
    """Trace ``eta`` from launch points on a section until each line returns.

    The section runs from the centre along latitude, east (``direction=+1``) or
    west (``-1``), for ``section_m`` metres. A line *returns* at its first
    crossing of the section after one turn about its centre, and stops without
    a return where the field ends, where it crosses the section without having
    turned, or after one and a half turns.

    Returns
    -------
    returned_s : np.ndarray
        Arc length from the centre along the section of each return, in metres,
        NaN where the line did not return.
    tracks : list of (lon, lat) or None
        With ``keep_track``, each line's points up to and including its return.
    """
    n = lon_0.size
    lon, lat = lon_0.astype(float).copy(), lat_0.astype(float).copy()
    cos_centre = np.cos(np.deg2rad(centre_lat))
    returned_s = np.full(n, np.nan)

    def tangent_for(index):
        def tangent(lon, lat, heading):
            return _eta_tangent(
                lon,
                lat,
                heading,
                stretch=stretch[index],
                branch=branch[index],
                tensor_interp=tensor_interp,
            )

        return tangent

    # No running heading at launch, so the direction of travel comes from the
    # 45-degree direction. Either way round traces the same loop.
    heading = tangent_for(np.arange(n))(lon, lat, np.ones((n, 2)))
    active = np.isfinite(heading).all(axis=1)
    dx, dy = _separation_m(lon_a=centre_lon, lat_a=centre_lat, lon_b=lon, lat_b=lat)
    angle = np.arctan2(dy, dx)
    turns = np.zeros(n)
    off_section = np.zeros(n, dtype=bool)
    tracks = [[(lon[k], lat[k])] for k in range(n)] if keep_track else None

    for _ in range(n_steps):
        index = np.flatnonzero(active)
        if index.size == 0:
            break
        lon_old, lat_old = lon[index], lat[index]
        lon_new, lat_new, heading_new = _rk2_step(
            lon_old,
            lat_old,
            heading[index],
            tangent=tangent_for(index),
            step_m=step_m,
        )
        ended = ~np.isfinite(heading_new).all(axis=1)

        dx, dy = _separation_m(
            lon_a=centre_lon[index],
            lat_a=centre_lat[index],
            lon_b=lon_new,
            lat_b=lat_new,
        )
        angle_new = np.arctan2(dy, dx)
        turns[index] += _wrap_lon(np.rad2deg(angle_new - angle[index])) / 360.0
        angle[index] = angle_new

        north_old = lat_old - centre_lat[index]
        north_new = lat_new - centre_lat[index]
        was_off = off_section[index]
        off_section[index] |= north_new != 0.0
        crossed = was_off & (north_old * north_new <= 0.0) & ~ended
        fraction = np.where(crossed, north_old / (north_old - north_new + 1e-300), 0)
        lon_cross = lon_old + fraction * _wrap_lon(lon_new - lon_old)
        s_cross = (
            direction[index]
            * _wrap_lon(lon_cross - centre_lon[index])
            * _M_PER_DEG
            * cos_centre[index]
        )
        on_section = crossed & (s_cross >= 0.0) & (s_cross <= section_m)
        has_turned = np.abs(turns[index]) > 0.5
        returned = on_section & has_turned
        returned_s[index[returned]] = s_cross[returned]

        lon[index], lat[index], heading[index] = lon_new, lat_new, heading_new
        if keep_track:
            for k in np.flatnonzero(~ended):
                if returned[k]:
                    tracks[index[k]].append((lon_cross[k], centre_lat[index[k]]))
                else:
                    tracks[index[k]].append((lon_new[k], lat_new[k]))
        finished = ended | on_section | (np.abs(turns[index]) > 1.5)
        active[index[finished]] = False

    return returned_s, tracks


def _section_launch(*, centre_lon, centre_lat, direction, s):
    """Positions ``s`` metres from the centre along the section (east or west)."""
    return (
        centre_lon + direction * s / (_M_PER_DEG * np.cos(np.deg2rad(centre_lat))),
        np.asarray(centre_lat, dtype=float) * np.ones_like(s),
    )


def _polygon_metres(lon, lat, *, centre_lon, centre_lat):
    """A closed lon/lat polygon as east/north metres from a centre."""
    return _separation_m(lon_a=centre_lon, lat_a=centre_lat, lon_b=lon, lat_b=lat)


def _area_and_centroid(x, y):
    """Shoelace area (m2) and centroid (m) of a closed polygon in metres."""
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * cross.sum()
    if signed_area == 0.0:
        return 0.0, (np.nan, np.nan)
    centroid_x = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * signed_area)
    centroid_y = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * signed_area)
    return abs(signed_area), (centroid_x, centroid_y)


def _orbit_dataset(orbits: list[dict], attrs: dict) -> xr.Dataset:
    n_points = max((o["lon"].size for o in orbits), default=0)
    lon = np.full((len(orbits), n_points), np.nan)
    lat = np.full((len(orbits), n_points), np.nan)
    for k, o in enumerate(orbits):
        lon[k, : o["lon"].size] = o["lon"]
        lat[k, : o["lat"].size] = o["lat"]

    def per_orbit(key, dtype=float):
        return np.array([o[key] for o in orbits], dtype=dtype)

    return xr.Dataset(
        {
            "lon": (
                ("orbit", "point"),
                lon,
                {"long_name": "longitude along the closed shear line"}
                | _LONLAT_ATTRS["lon"],
            ),
            "lat": (
                ("orbit", "point"),
                lat,
                {"long_name": "latitude along the closed shear line"}
                | _LONLAT_ATTRS["lat"],
            ),
            "centre": (
                "orbit",
                per_orbit("centre", int),
                {"long_name": "index of the candidate centre searched", "units": "1"},
            ),
            "centre_lon": (
                "orbit",
                per_orbit("centre_lon"),
                {"long_name": "longitude of the candidate centre searched"}
                | _LONLAT_ATTRS["lon"],
            ),
            "centre_lat": (
                "orbit",
                per_orbit("centre_lat"),
                {"long_name": "latitude of the candidate centre searched"}
                | _LONLAT_ATTRS["lat"],
            ),
            "stretch": (
                "orbit",
                per_orbit("stretch"),
                {
                    "long_name": "uniform stretching factor lambda of the curve",
                    "units": "1",
                },
            ),
            "branch": (
                "orbit",
                per_orbit("branch", int),
                {"long_name": "branch of eta^pm_lambda, +1 or -1", "units": "1"},
            ),
            "area_m2": (
                "orbit",
                per_orbit("area_m2"),
                {"long_name": "area enclosed by the closed shear line", "units": "m2"},
            ),
            "radius_m": (
                "orbit",
                per_orbit("radius_m"),
                {
                    "long_name": "equivalent radius sqrt(area / pi) of the closed "
                    "shear line",
                    "units": "m",
                },
            ),
            "residual_m": (
                "orbit",
                per_orbit("residual_m"),
                {
                    "long_name": "distance from launch to return on the section",
                    "units": "m",
                },
            ),
        },
        coords={
            "orbit": (
                "orbit",
                np.arange(len(orbits)),
                {"long_name": "closed shear line index"},
            ),
            "point": (
                "point",
                np.arange(n_points),
                {"long_name": "point index along the closed shear line"},
            ),
        },
        attrs=attrs,
    )


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

    For every centre, stretching factor in ``stretches`` and branch ``pm``, the
    search launches ``eta^pm_lambda`` lines from two sections, due east and due
    west of the centre over ``max_radius_m``, every ``launch_spacing_m``. It
    reads the return map ``P(s)`` of each section and refines every sign change
    of ``P(s) - s`` by bisection. An orbit is kept when it turns once about its
    centre and returns within ``closure_tol_m`` of its launch point.

    Every orbit found is returned, so a vortex comes back as its nested family
    of loops over the scanned stretching factors.
    :func:`outermost_shear_lines` picks the boundary.

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
        NaN-padded past its end. Per orbit: the ``centre`` searched with its
        ``centre_lon``/``centre_lat``, ``stretch``, ``branch``, ``area_m2``,
        ``radius_m`` (equivalent radius) and ``residual_m``. The ``attrs``
        record the scan.
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
    geometry = _window_geometry(flowmap.lon_grid, 1.0)
    spacing_m = min(geometry["grid_spacing_i_m"], geometry["grid_spacing_j_m"])
    launch_spacing_m = spacing_m if launch_spacing_m is None else launch_spacing_m
    step_m = 0.5 * spacing_m if step_m is None else step_m
    closure_tol_m = 0.5 * spacing_m if closure_tol_m is None else closure_tol_m
    n_steps = int(np.ceil(2.0 * 2.0 * np.pi * max_radius_m / step_m))
    s_launch = launch_spacing_m * np.arange(
        1, int(np.floor(max_radius_m / launch_spacing_m)) + 1
    )
    tensor_interp = _tensor_interp(flowmap)
    trace = {
        "section_m": max_radius_m,
        "tensor_interp": tensor_interp,
        "step_m": step_m,
        "n_steps": n_steps,
    }

    # One curve family per (centre, section, stretch, branch), each launched at
    # every s_launch.
    family = np.array(
        np.meshgrid(
            np.arange(centre_lon.size),
            [1, -1],
            np.arange(stretches.size),
            [1, -1],
            indexing="ij",
        )
    ).reshape(4, -1)
    families_per_chunk = max(1, _LINES_PER_CHUNK // max(1, s_launch.size))
    orbits = []
    n_never_returned = 0
    for start in range(0, family.shape[1], families_per_chunk):
        c, d, k, b = family[:, start : start + families_per_chunk]
        props = {
            "centre_lon": centre_lon[c],
            "centre_lat": centre_lat[c],
            "direction": d.astype(float),
            "stretch": stretches[k],
            "branch": b.astype(float),
        }

        def repeat(props, count):
            return {key: np.repeat(value, count) for key, value in props.items()}

        lines = repeat(props, s_launch.size)
        s = np.tile(s_launch, c.size)
        lon_0, lat_0 = _section_launch(
            centre_lon=lines["centre_lon"],
            centre_lat=lines["centre_lat"],
            direction=lines["direction"],
            s=s,
        )
        returned_s, _ = _trace_to_return(lon_0, lat_0, **lines, **trace)
        offset = (returned_s - s).reshape(c.size, s_launch.size)
        n_never_returned += int((~np.isfinite(offset)).all(axis=1).sum())

        # A launch that already returns within tolerance is an orbit. Where every
        # launch does, as for the circles of an axisymmetric vortex, the offset
        # need never change sign.
        hit = np.isfinite(offset) & (np.abs(offset) < closure_tol_m)
        hit_fam, hit_k = np.nonzero(hit)
        # Brackets: adjacent launches, neither a hit, whose finite offsets change
        # sign.
        sign_change = (
            np.isfinite(offset[:, :-1])
            & np.isfinite(offset[:, 1:])
            & (np.sign(offset[:, :-1]) != np.sign(offset[:, 1:]))
            & ~hit[:, :-1]
            & ~hit[:, 1:]
        )
        fam, left = np.nonzero(sign_change)
        s_lo, s_hi = s_launch[left], s_launch[left + 1]
        f_lo = offset[fam, left]
        bracket = {key: value[fam] for key, value in props.items()}
        while fam.size and (s_hi - s_lo).max() > 0.25 * closure_tol_m:
            s_mid = 0.5 * (s_lo + s_hi)
            lon_m, lat_m = _section_launch(
                centre_lon=bracket["centre_lon"],
                centre_lat=bracket["centre_lat"],
                direction=bracket["direction"],
                s=s_mid,
            )
            f_mid = _trace_to_return(lon_m, lat_m, **bracket, **trace)[0] - s_mid
            # A bracket whose midpoint does not return straddles a gap in the
            # return map rather than a zero of it.
            alive = np.isfinite(f_mid)
            lower = alive & (np.sign(f_mid) == np.sign(f_lo))
            s_lo = np.where(lower, s_mid, s_lo)
            f_lo = np.where(lower, f_mid, f_lo)
            s_hi = np.where(alive & ~lower, s_mid, s_hi)
            fam, s_lo, s_hi, f_lo = fam[alive], s_lo[alive], s_hi[alive], f_lo[alive]
            bracket = {key: value[alive] for key, value in bracket.items()}
        fam = np.concatenate([fam, hit_fam])
        if fam.size == 0:
            continue
        bracket = {
            key: np.concatenate([value, props[key][hit_fam]])
            for key, value in bracket.items()
        }
        s_fix = np.concatenate([0.5 * (s_lo + s_hi), s_launch[hit_k]])
        lon_f, lat_f = _section_launch(
            centre_lon=bracket["centre_lon"],
            centre_lat=bracket["centre_lat"],
            direction=bracket["direction"],
            s=s_fix,
        )
        returned_s, tracks = _trace_to_return(
            lon_f, lat_f, **bracket, **trace, keep_track=True
        )
        residual = np.abs(returned_s - s_fix)
        for m in np.flatnonzero(residual < closure_tol_m):
            track_lon, track_lat = (np.array(v) for v in zip(*tracks[m], strict=True))
            track_lon = np.append(track_lon, lon_f[m])
            track_lat = np.append(track_lat, lat_f[m])
            x, y = _polygon_metres(
                track_lon,
                track_lat,
                centre_lon=bracket["centre_lon"][m],
                centre_lat=bracket["centre_lat"][m],
            )
            area, _ = _area_and_centroid(x, y)
            orbits.append(
                {
                    "lon": track_lon,
                    "lat": track_lat,
                    "centre": c[fam[m]],
                    "centre_lon": bracket["centre_lon"][m],
                    "centre_lat": bracket["centre_lat"][m],
                    "stretch": bracket["stretch"][m],
                    "branch": int(bracket["branch"][m]),
                    "area_m2": area,
                    "radius_m": np.sqrt(area / np.pi),
                    "residual_m": residual[m],
                }
            )

    attrs = {
        "stretch_min": float(stretches.min()) if stretches.size else np.nan,
        "stretch_max": float(stretches.max()) if stretches.size else np.nan,
        "n_stretch": int(stretches.size),
        "max_radius_m": float(max_radius_m),
        "launch_spacing_m": float(launch_spacing_m),
        "step_m": float(step_m),
        "closure_tol_m": float(closure_tol_m),
        "n_centres": int(centre_lon.size),
        "n_never_returned": n_never_returned,
    }
    return _orbit_dataset(orbits, attrs)


def _points_inside(x_poly, y_poly, x, y):
    """Even-odd test of points ``(x, y)`` against a closed polygon, all in metres."""
    x1, y1 = x_poly[:-1, None], y_poly[:-1, None]
    x2, y2 = x_poly[1:, None], y_poly[1:, None]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (straddles & (x < x_cross)).sum(axis=0) % 2 == 1


def outermost_shear_lines(orbits: xr.Dataset, *, flowmap) -> xr.Dataset:
    """The outermost closed shear line of each vortex, with its rotation sense.

    Keeps the largest-area orbit per candidate centre, then drops any whose
    centre lies inside a larger kept boundary, so several candidate centres in
    one vortex give one eddy. ``rotation_sense`` is the sign of the mean
    :meth:`~lcs_parcels.FlowMap.polar_rotation` inside the boundary against its
    domain median, ``+1`` counter-clockwise and ``-1`` clockwise.

    Returns
    -------
    xr.Dataset
        The kept rows of ``orbits`` on ``(eddy, point)``, with their original
        ``orbit`` index, plus ``centroid_lon``/``centroid_lat`` of each boundary
        and ``rotation_sense``.
    """
    theta = flowmap.polar_rotation()
    theta_lon = theta["lon_grid"].values.ravel()
    theta_lat = theta["lat_grid"].values.ravel()
    theta_anomaly = (theta - theta.median()).values.ravel()

    area = orbits["area_m2"].values
    best = {}
    for k in range(orbits.sizes["orbit"]):
        centre = int(orbits["centre"][k])
        if centre not in best or area[k] > area[best[centre]]:
            best[centre] = k

    kept, centroids, senses = [], [], []
    for k in sorted(best.values(), key=lambda k: -area[k]):
        line = orbits.isel(orbit=k)
        finite = np.isfinite(line["lon"].values)
        lon, lat = line["lon"].values[finite], line["lat"].values[finite]
        centre_lon, centre_lat = float(line["centre_lon"]), float(line["centre_lat"])
        if any(
            _points_inside(
                *_polygon_metres(
                    other["lon"],
                    other["lat"],
                    centre_lon=centre_lon,
                    centre_lat=centre_lat,
                ),
                np.array([0.0]),
                np.array([0.0]),
            )[0]
            for other in kept
        ):
            continue
        x, y = _polygon_metres(lon, lat, centre_lon=centre_lon, centre_lat=centre_lat)
        _, (centroid_x, centroid_y) = _area_and_centroid(x, y)
        gx, gy = _polygon_metres(
            theta_lon, theta_lat, centre_lon=centre_lon, centre_lat=centre_lat
        )
        near = (np.abs(gx) <= np.abs(x).max()) & (np.abs(gy) <= np.abs(y).max())
        inside = np.zeros_like(near)
        inside[near] = _points_inside(x, y, gx[near], gy[near])
        mean_anomaly = np.nanmean(theta_anomaly[inside]) if inside.any() else np.nan
        kept.append({"index": k, "lon": lon, "lat": lat})
        centroids.append(
            (
                centre_lon + centroid_x / (_M_PER_DEG * np.cos(np.deg2rad(centre_lat))),
                centre_lat + centroid_y / _M_PER_DEG,
            )
        )
        senses.append(int(np.sign(mean_anomaly)) if np.isfinite(mean_anomaly) else 0)

    index = [entry["index"] for entry in kept]
    eddies = (
        orbits.isel(orbit=index)
        .rename_dims(orbit="eddy")
        .reset_coords()
        .rename_vars(orbit="orbit_index")
    )
    eddies = eddies.assign(
        orbit_index=eddies["orbit_index"].assign_attrs(
            long_name="index of the orbit in the closed_shear_lines result",
            units="1",
        ),
        centroid_lon=(
            "eddy",
            np.array([c[0] for c in centroids], dtype=float),
            {"long_name": "longitude of the eddy boundary centroid"}
            | _LONLAT_ATTRS["lon"],
        ),
        centroid_lat=(
            "eddy",
            np.array([c[1] for c in centroids], dtype=float),
            {"long_name": "latitude of the eddy boundary centroid"}
            | _LONLAT_ATTRS["lat"],
        ),
        rotation_sense=(
            "eddy",
            np.array(senses, dtype=int),
            {
                "long_name": "rotation sense, +1 counter-clockwise, -1 clockwise",
                "units": "1",
            },
        ),
    )
    finite_rows = np.isfinite(eddies["lon"]).any("eddy")
    n_points = int(finite_rows.values.nonzero()[0].max()) + 1 if index else 0
    return (
        eddies.isel(point=slice(0, n_points))
        .assign_coords(
            eddy=("eddy", np.arange(len(index)), {"long_name": "eddy index"})
        )
        .assign_attrs(
            n_orbits_in=int(orbits.sizes["orbit"]),
            n_centres_with_orbits=len(best),
        )
    )
