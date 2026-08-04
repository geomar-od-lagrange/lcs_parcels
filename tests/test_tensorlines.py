"""Tensor-line tests: ftle_ridge_seeds and shrink_lines.

For a constant linear flow map ``F(x) = M @ x`` the Cauchy-Green tensor
``C = M^T M`` is uniform, so its eigenvectors are the same everywhere and a
shrink line (tangent to ``xi_1``) is a straight line. Picking ``M = diag(1, 3)``
makes ``C = diag(1, 9)``: ``xi_1`` is the x-axis, so the shrink line is purely
zonal (constant latitude) -- a closed-form check. ``conftest.advected_flowmap``
builds such a flow map (``AuxiliarySeed`` so ``gradF`` is defined at every grid
point, no NaN edges).
"""

import numpy as np
import pytest
import xarray as xr
from conftest import advected_flowmap, advected_flowmap_f
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels import AuxiliarySeed, ftle_ridge_seeds, shrink_lines
from lcs_parcels.grids import _lonlat_to_meters
from lcs_parcels.tensorlines import (
    _shrink_line_tangent,
    _step_lonlat_by_meters,
    _trace_half_line,
    _window_cells,
)

T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")


# --- ftle_ridge_seeds ------------------------------------------------------


def test_ftle_ridge_seeds_picks_the_peak(lon_axis, lat_axis):
    """A single smooth FTLE bump yields exactly its peak grid point as the seed."""
    lon2d, lat2d = xr.broadcast(
        xr.DataArray(lon_axis, dims="i"), xr.DataArray(lat_axis, dims="j")
    )
    ii, jj = np.meshgrid(
        np.arange(lon_axis.size), np.arange(lat_axis.size), indexing="ij"
    )
    bump = np.exp(-((ii - 2.0) ** 2 + (jj - 2.0) ** 2))
    ftle = xr.DataArray(
        bump,
        dims=("i", "j"),
        coords={
            "lon_grid": (("i", "j"), lon2d.values),
            "lat_grid": (("i", "j"), lat2d.values),
        },
    )

    lon, lat = ftle_ridge_seeds(ftle, window_m=330_000.0, quantile=0.90)

    assert lon.size == 1
    assert lon[0] == lon_axis[2]
    assert lat[0] == lat_axis[2]


def test_ftle_ridge_seeds_skips_nan(lon_axis, lat_axis):
    """NaN cells never qualify as seeds."""
    lon2d, lat2d = xr.broadcast(
        xr.DataArray(lon_axis, dims="i"), xr.DataArray(lat_axis, dims="j")
    )
    field = np.full((lon_axis.size, lat_axis.size), np.nan)
    field[1, 1] = 5.0  # a lone finite peak
    ftle = xr.DataArray(
        field,
        dims=("i", "j"),
        coords={
            "lon_grid": (("i", "j"), lon2d.values),
            "lat_grid": (("i", "j"), lat2d.values),
        },
    )

    lon, lat = ftle_ridge_seeds(ftle, window_m=330_000.0, quantile=0.5)

    assert lon.tolist() == [lon_axis[1]]
    assert lat.tolist() == [lat_axis[1]]


def _two_bump_ftle(n_lon):
    """Two unequal bumps 1 degree apart on the equator, on a grid of ``n_lon`` points.

    The bumps sit at lon = -0.5 and +0.5 (about 111 km apart) whatever the
    resolution, so the physical layout is fixed and only the cell size changes.
    """
    lon_axis = np.linspace(-3.0, 3.0, n_lon)
    lat_axis = np.linspace(-1.0, 1.0, (n_lon - 1) // 3 + 1)
    lon2d, lat2d = xr.broadcast(
        xr.DataArray(lon_axis, dims="i"), xr.DataArray(lat_axis, dims="j")
    )
    bumps = np.exp(-((lon2d + 0.5) ** 2 + lat2d**2) / 0.02) + 0.9 * np.exp(
        -((lon2d - 0.5) ** 2 + lat2d**2) / 0.02
    )
    return bumps.assign_coords(lon_grid=lon2d, lat_grid=lat2d)


def test_window_cell_count_tracks_grid_resolution():
    """The same physical window is a different number of cells on a finer grid."""
    coarse = _two_bump_ftle(61)  # 0.1 degree cells
    fine = _two_bump_ftle(121)  # 0.05 degree cells

    cells_coarse = _window_cells(coarse, 60_000.0)
    cells_fine = _window_cells(fine, 60_000.0)

    assert cells_coarse == (5, 5)
    assert cells_fine == (11, 11)


def test_window_cell_count_is_per_dimension_and_odd():
    """Each dimension gets its own count, from its own spacing in the package's
    single-reference-latitude metres frame, rounded down to an odd number.

    A mid-latitude grid of 0.1 deg by 0.05 deg cells separates the three things an
    equatorial isotropic grid hides. With ``phi_ref = 40 N`` the spacings are
    ``dx = 0.1 * 111195 * cos(40) = 8518 m`` and ``dy = 0.05 * 111195 = 5560 m``,
    so a 50 km window is ``round(5.87) = 6 -> 5`` cells along ``i`` (the
    odd-enforcement branch fires) and ``round(8.99) = 9`` along ``j``.
    """
    lon_axis = np.arange(0.0, 4.0001, 0.1)
    lat_axis = np.arange(30.0, 50.0001, 0.05)
    lon2d, lat2d = xr.broadcast(
        xr.DataArray(lon_axis, dims="i"), xr.DataArray(lat_axis, dims="j")
    )
    ftle = xr.DataArray(
        np.zeros(lon2d.shape),
        dims=("i", "j"),
        coords={"lon_grid": lon2d, "lat_grid": lat2d},
    )

    assert _window_cells(ftle, 50_000.0) == (5, 9)


@pytest.mark.parametrize("n_lon", [61, 121])
def test_ftle_ridge_seeds_selectivity_is_physical(n_lon):
    """Resolving or merging the two bumps depends on window_m, not on cell size:
    a window narrower than their 111 km separation keeps both peaks, a wider one
    keeps only the stronger -- identically on the coarse and the fine grid."""
    ftle = _two_bump_ftle(n_lon)

    lon_narrow, _ = ftle_ridge_seeds(ftle, window_m=60_000.0)
    lon_wide, _ = ftle_ridge_seeds(ftle, window_m=300_000.0)

    assert np.allclose(np.sort(lon_narrow), [-0.5, 0.5])
    assert np.allclose(lon_wide, [-0.5])  # only the stronger bump survives


# --- lifted integrator internals -------------------------------------------
#
# These take the state that used to be closed over (the interpolator, the
# anisotropy floor, the step, the reference latitude) as explicit arguments, so
# they can be driven from an analytic tensor field without building a FlowMap.

TENSOR_LON = np.linspace(-1.0, 1.0, 21)
TENSOR_LAT = np.linspace(-1.0, 1.0, 21)


def _uniform_tensor_interp(C, nan_at=None):
    """Interpolator over a constant Cauchy-Green tensor ``C`` on a small grid.

    ``nan_at``, if given, is an ``(i, j)`` index pair whose tensor is set to NaN,
    standing in for a lost/land cell.
    """
    field = np.broadcast_to(
        np.asarray(C, dtype=float), (TENSOR_LON.size, TENSOR_LAT.size, 2, 2)
    ).copy()
    if nan_at is not None:
        field[nan_at[0], nan_at[1]] = np.nan
    return RegularGridInterpolator(
        (TENSOR_LON, TENSOR_LAT), field, bounds_error=False, fill_value=np.nan
    )


def test_shrink_line_tangent_is_unit_xi1():
    """C = diag(1, 9) has xi_1 along x; the returned vector is that unit vector."""
    interp = _uniform_tensor_interp(np.diag([1.0, 9.0]))

    direction = _shrink_line_tangent(
        np.array([0.0]),
        np.array([0.0]),
        np.array([[1.0, 0.0]]),
        tensor_interp=interp,
        min_anisotropy=1.1,
    )

    assert np.allclose(direction, [[1.0, 0.0]])
    assert np.allclose(np.linalg.norm(direction, axis=1), 1.0)


def test_shrink_line_tangent_follows_the_heading():
    """An eigenvector has no intrinsic sign: the heading picks which way it points."""
    interp = _uniform_tensor_interp(np.diag([1.0, 9.0]))
    lon, lat = np.zeros(2), np.zeros(2)

    direction = _shrink_line_tangent(
        lon,
        lat,
        np.array([[1.0, 0.0], [-1.0, 0.0]]),
        tensor_interp=interp,
        min_anisotropy=1.1,
    )

    assert np.allclose(direction, [[1.0, 0.0], [-1.0, 0.0]])


def test_shrink_line_tangent_is_nan_off_grid():
    """Points outside the tensor grid terminate."""
    interp = _uniform_tensor_interp(np.diag([1.0, 9.0]))

    direction = _shrink_line_tangent(
        np.array([0.0, 5.0]),
        np.array([0.0, 0.0]),
        np.ones((2, 2)),
        tensor_interp=interp,
        min_anisotropy=1.1,
    )

    assert np.isfinite(direction[0]).all()
    assert np.isnan(direction[1]).all()


def test_shrink_line_tangent_is_nan_in_a_nan_cell():
    """A NaN tensor cell terminates a line landing on it."""
    interp = _uniform_tensor_interp(np.diag([1.0, 9.0]), nan_at=(10, 10))

    direction = _shrink_line_tangent(
        np.array([TENSOR_LON[10], TENSOR_LON[0]]),
        np.array([TENSOR_LAT[10], TENSOR_LAT[0]]),
        np.ones((2, 2)),
        tensor_interp=interp,
        min_anisotropy=1.1,
    )

    assert np.isnan(direction[0]).all()
    assert np.isfinite(direction[1]).all()


def test_shrink_line_tangent_is_nan_below_the_anisotropy_floor():
    """The guard floors ``lambda_2 / lambda_1``, so what decides is the *gap*
    between the eigenvalues and not the magnitude of ``lambda_2``.

    Both tensors here carry the same ``lambda_2 = 9``. ``diag(1, 9)`` has a ratio
    of 9 and clears a floor of 1.5; ``diag(8, 9)`` has a ratio of 1.125 and does
    not, so a guard reading ``lambda_2`` alone could not tell them apart.
    """
    args = (np.array([0.0]), np.array([0.0]), np.array([[1.0, 0.0]]))

    wide_gap = _shrink_line_tangent(
        *args,
        tensor_interp=_uniform_tensor_interp(np.diag([1.0, 9.0])),
        min_anisotropy=1.5,
    )
    narrow_gap = _shrink_line_tangent(
        *args,
        tensor_interp=_uniform_tensor_interp(np.diag([8.0, 9.0])),
        min_anisotropy=1.5,
    )

    assert np.isfinite(wide_gap).all()
    assert np.isnan(narrow_gap).all()


def test_shrink_line_tangent_passes_a_round_off_negative_lambda_1():
    """``C`` is positive semi-definite, so a ``lambda_1`` just below zero is
    round-off on an extremely anisotropic tensor, not a degeneracy: the tangent
    there is as well defined as it gets and must come back finite."""
    interp = _uniform_tensor_interp(np.diag([-1e-15, 9.0]))

    direction = _shrink_line_tangent(
        np.array([0.0]),
        np.array([0.0]),
        np.array([[1.0, 0.0]]),
        tensor_interp=interp,
        min_anisotropy=1.15,
    )

    assert np.isfinite(direction).all()
    assert np.allclose(direction, [[1.0, 0.0]])


def test_step_lonlat_moves_the_requested_arc_length():
    """A unit direction moves exactly step_m in the package's own metres frame."""
    lon0, lat0 = np.array([0.0]), np.array([20.0])
    direction = np.array([[np.cos(0.7), np.sin(0.7)]])

    lon1, lat1 = _step_lonlat_by_meters(
        lon0, lat0, direction, step_m=25_000.0, lat_ref=20.0
    )

    x0, y0 = _lonlat_to_meters(lon0, lat0, 0.0, 20.0)
    x1, y1 = _lonlat_to_meters(lon1, lat1, 0.0, 20.0)
    assert np.allclose(np.hypot(x1 - x0, y1 - y0), 25_000.0)


def test_step_lonlat_spends_more_degrees_at_higher_latitude():
    """The same eastward metres are more degrees of longitude nearer the pole."""
    lon0, lat0 = np.array([0.0]), np.array([0.0])
    east = np.array([[1.0, 0.0]])

    lon_equator, _ = _step_lonlat_by_meters(
        lon0, lat0, east, step_m=25_000.0, lat_ref=0.0
    )
    lon_polar, _ = _step_lonlat_by_meters(
        lon0, lat0, east, step_m=25_000.0, lat_ref=60.0
    )

    assert lon_polar[0] > lon_equator[0]
    assert np.allclose(lon_polar[0], lon_equator[0] / np.cos(np.deg2rad(60.0)))


def _turning_tensor(turn_per_degree, *, half_extent_deg=10.0):
    """``C`` whose ``xi_1`` direction turns with longitude, as a callable.

    ``C(lon) = R(theta) diag(1, 9) R(theta)^T`` with ``theta = turn_per_degree *
    lon``, so ``xi_1`` is ``(cos theta, sin theta)`` and a curve tracing it bends
    as it advances. Every uniform-``C`` fixture above hides the integration
    scheme: there the midpoint direction equals the direction at the current
    point, so RK2 and plain Euler produce bit-identical tracks.

    Exposes the ``RegularGridInterpolator`` call interface but evaluates exactly,
    so a measured error is the integrator's own rather than the tensor
    interpolation's. Points beyond ``half_extent_deg`` return NaN, standing in for
    leaving the grid.
    """

    def evaluate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        theta = turn_per_degree * points[:, 0]
        c, s = np.cos(theta), np.sin(theta)
        out = np.empty((points.shape[0], 2, 2))
        out[:, 0, 0] = c * c + 9.0 * s * s
        out[:, 0, 1] = out[:, 1, 0] = -8.0 * c * s
        out[:, 1, 1] = s * s + 9.0 * c * c
        out[np.max(np.abs(points), axis=1) > half_extent_deg] = np.nan
        return out

    return evaluate


def _dipping_tensor(lon_dip, *, half_width_deg):
    """``C = diag(1, lambda_2)`` with ``lambda_2`` dropping from 9 to 1 in a narrow
    longitude band, so a step can straddle a degenerate patch its endpoints miss.
    """

    def evaluate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        in_dip = np.abs(points[:, 0] - lon_dip) < half_width_deg
        out = np.zeros((points.shape[0], 2, 2))
        out[:, 0, 0] = 1.0
        out[:, 1, 1] = np.where(in_dip, 1.0, 9.0)
        return out

    return evaluate


def test_trace_half_line_is_second_order_in_the_step():
    """RK2, not Euler: halving the step cuts the endpoint change about fourfold.

    Traces the same arc length at three step sizes on a tensor field whose
    ``xi_1`` turns, and compares successive endpoints. Second order gives a ratio
    near 4; first-order Euler gives near 2.
    """
    tensor = _turning_tensor(0.5)
    arc_m = 300_000.0

    def endpoint(n_steps):
        track = _trace_half_line(
            np.array([0.0]),
            np.array([0.0]),
            +1,
            tensor_interp=tensor,
            min_anisotropy=1.1,
            step_m=arc_m / n_steps,
            lat_ref=0.0,
            n_steps=n_steps,
        )
        return np.array([track[-1][0][0], track[-1][1][0]])

    coarse, medium, fine = endpoint(10), endpoint(20), endpoint(40)
    first = np.linalg.norm(medium - coarse)
    second = np.linalg.norm(fine - medium)

    assert np.isfinite(coarse).all()  # the arc stays inside the field
    assert first > 0.0  # the step size matters at all
    assert 3.0 < first / second < 5.0


def test_trace_half_line_guard_fires_at_the_rk2_midpoint():
    """A step whose two endpoints are both fine still terminates the line if the
    tensor is degenerate halfway along it.

    One step spans a degree of longitude; the degenerate band sits at 0.5 deg, so
    the seed and the step's endpoint both see ``lambda_2 = 9`` and only the RK2
    midpoint lands below the floor.
    """
    tensor = _dipping_tensor(0.5, half_width_deg=0.1)

    track = _trace_half_line(
        np.array([0.0]),
        np.array([0.0]),
        +1,
        tensor_interp=tensor,
        min_anisotropy=1.1,
        step_m=111_195.0,  # about one degree of longitude at the equator
        lat_ref=0.0,
        n_steps=3,
    )

    lon = np.array([p[0][0] for p in track])
    assert np.isfinite(lon[0])  # the seed itself is fine
    assert np.isnan(lon[1:]).all()  # the first step dies at its midpoint


def test_trace_half_line_point_count_and_nan_padding():
    """The track is n_steps + 1 points and, once it leaves the grid, stays NaN."""
    interp = _uniform_tensor_interp(np.diag([1.0, 9.0]))

    track = _trace_half_line(
        np.array([0.0]),
        np.array([0.0]),
        +1,
        tensor_interp=interp,
        min_anisotropy=1.1,
        step_m=50_000.0,
        lat_ref=0.0,
        n_steps=10,
    )

    assert len(track) == 11
    lon = np.array([p[0][0] for p in track])
    assert np.isfinite(lon[:3]).all()  # 0.45 deg steps stay inside |lon| <= 1
    assert np.isnan(lon[-1])
    finite = np.isfinite(lon)
    assert not finite[np.argmin(finite) :].any()  # never recovers once terminated


# --- shrink_lines ----------------------------------------------------------


def _centre_seed(flowmap):
    """A single seed at the grid centre, as ``shrink_lines`` keyword arguments."""
    return {
        "seed_lon": [float(flowmap.ds["lon_grid"].mean())],
        "seed_lat": [float(flowmap.ds["lat_grid"].mean())],
    }


def test_shrink_line_is_zonal_for_diagonal_map(lon_axis, lat_axis):
    """M = diag(1, 3) => xi_1 is the x-axis => the shrink line has constant latitude."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    lines = shrink_lines(
        fm, **_centre_seed(fm), step_m=10_000.0, line_length_m=80_000.0
    )

    lon = lines["lon"].isel(line=0).values
    lat = lines["lat"].isel(line=0).values
    valid = np.isfinite(lon) & np.isfinite(lat)

    assert valid.all()  # short line stays on the grid
    assert np.ptp(lat[valid]) < 1e-9  # constant latitude (tangent to x)
    assert np.ptp(lon[valid]) > 0.1  # and spans in longitude


def test_shrink_lines_output_structure(lon_axis, lat_axis):
    """Dataset has lon/lat on (line, point); one line per seed, an odd point count."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    seed_lon = [float(fm.ds["lon_grid"].mean()), float(fm.ds["lon_grid"].mean()) + 0.1]
    seed_lat = [float(fm.ds["lat_grid"].mean()), float(fm.ds["lat_grid"].mean())]

    lines = shrink_lines(
        fm, seed_lon=seed_lon, seed_lat=seed_lat, step_m=3_000.0, line_length_m=36_000.0
    )

    assert set(lines.dims) == {"line", "point"}
    assert lines.sizes["line"] == 2
    assert lines.sizes["point"] == 2 * 6 + 1  # 36 km of line in 3 km steps
    assert {"lon", "lat"} == set(lines.data_vars)


def test_shrink_lines_stop_at_an_isotropic_tensor(lon_axis, lat_axis):
    """M = I gives C = I, an eigenvalue ratio of exactly 1: xi_1 is an arbitrary
    direction in the plane, so the line is untraceable and comes back all NaN."""
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, np.eye(2), T0, T1)
    lines = shrink_lines(
        fm, **_centre_seed(fm), min_anisotropy=1.15, line_length_m=30_000.0
    )

    assert bool(lines["lon"].isnull().all())


@pytest.mark.parametrize("sign", [+1, -1])
@pytest.mark.parametrize("t_days", [2.0, 8.0])
def test_shrink_lines_guard_is_an_eigenvalue_ratio(lon_axis, lat_axis, sign, t_days):
    """The guard floors ``lambda_2 / lambda_1``, a dimensionless number the
    window does not enter, so one floor selects the same lines whatever ``|T|``
    the flow map spans and whichever way it runs.

    ``M = diag(a, b)`` gives ``C = diag(a^2, b^2)``, so for ``a > b`` the ratio
    is exactly ``r = (a / b)^2`` at every grid point -- and, unlike the map of a
    rate-based guard, ``M`` does not have to be restated per window. A floor just
    under ``r`` must let the line through and a floor just over it must kill it,
    for both windows and both directions.
    """
    a, b = 3.0, 1.0
    r = (a / b) ** 2
    t1 = T0 + sign * np.timedelta64(int(t_days * 24), "h")
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, np.diag([a, b]), T0, t1)

    lam = fm.cg_eigen()["lambda"]
    assert np.allclose(lam.isel(eig=1) / lam.isel(eig=0), r)

    kwargs = {**_centre_seed(fm), "step_m": 3_000.0, "line_length_m": 30_000.0}
    survives = shrink_lines(fm, min_anisotropy=0.9 * r, **kwargs)
    dies = shrink_lines(fm, min_anisotropy=1.1 * r, **kwargs)

    assert bool(survives["lon"].notnull().any())
    assert bool(dies["lon"].isnull().all())


def test_shrink_lines_guard_passes_a_uniformly_compressive_map(lon_axis, lat_axis):
    """The guard is a *relative* gap, not a magnitude floor on ``lambda_2``: this
    is the case that separates the two.

    ``M = diag(0.5, 0.4)`` compresses in both directions, so ``C = diag(0.25,
    0.16)``: ``lambda_1 = 0.16``, ``lambda_2 = 0.25`` and ``det C = 0.04``. The
    ratio is 1.5625, comfortably clear of the 1.15 default, and ``xi_1`` is as
    well defined here as in any stretching flow -- so the line must survive.
    Any floor on the magnitude of ``lambda_2`` at or above 1 would kill it.

    Not a synthetic corner: ``det grad F = 0.2`` here, and in the backward
    Cabo Verde example (5-day window, measured 34 km clear of any coast) the
    0.1st percentile of ``det grad F`` is 0.196. Convergent patches this strong
    are rare but real, and they are where attracting LCS live -- which is why a
    magnitude floor terminating there is a directional bias, not just a
    conservative choice.
    """
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([0.5, 0.4]), T0, T1
    )

    lam = fm.cg_eigen()["lambda"]
    assert np.allclose(lam.isel(eig=0), 0.16)
    assert np.allclose(lam.isel(eig=1), 0.25)

    lines = shrink_lines(fm, **_centre_seed(fm), step_m=3_000.0, line_length_m=30_000.0)

    assert bool(lines["lon"].notnull().all())


def test_shrink_lines_seed_off_grid_is_nan(lon_axis, lat_axis):
    """A seed outside the grid produces an all-NaN line."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    lines = shrink_lines(
        fm,
        seed_lon=[lon_axis[0] - 50.0],
        seed_lat=[lat_axis[0] - 50.0],
        line_length_m=30_000.0,
    )

    assert bool(lines["lon"].isnull().all())
    assert bool(lines["lat"].isnull().all())


def test_shrink_line_uses_reference_latitude_metric():
    """A uniform-C shrink line is straight in the single-reference-latitude metres
    frame the tensor lives in. Stepping with a per-point cos(lat) instead bows the
    curve as it climbs in latitude, so it would not stay collinear.

    ``M = R(45) diag(1, 3) R(45)^T`` is symmetric, so ``C = M^2`` shares its
    eigenvectors and ``xi_1`` (the smaller eigenvalue) points along the 45-degree
    diagonal -- a line that spans latitude, unlike the zonal test above.
    """
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c], [c, c]])
    M = R @ np.diag([1.0, 3.0]) @ R.T
    lon_axis = np.linspace(-15.0, 15.0, 31)
    lat_axis = np.linspace(0.0, 40.0, 41)
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)

    lines = shrink_lines(
        fm, seed_lon=[0.0], seed_lat=[20.0], step_m=20_000.0, line_length_m=2_400_000.0
    )
    lon = lines["lon"].isel(line=0).values
    lat = lines["lat"].isel(line=0).values
    valid = np.isfinite(lon) & np.isfinite(lat)
    assert valid.sum() > 50  # the diagonal line stays on this wide grid

    # In the frame C lives in (one reference latitude), the line must be straight.
    lon_ref, lat_ref = float(fm.ds["lon_0"].mean()), float(fm.ds["lat_0"].mean())
    x, y = _lonlat_to_meters(lon[valid], lat[valid], lon_ref, lat_ref)
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    assert np.max(np.abs(resid)) < 1e3  # collinear to < 1 km over ~1000 km


# --- FlowMap.hyperbolic_lcs ------------------------------------------------
#
# A *uniform* linear map will not do here. It makes the FTLE constant to within
# float noise, so every point ties as a ridge point and both ridge parameters
# become inert: neither `window_m` nor `quantile` changes the seed set at all,
# and a parity test built on it cannot see `hyperbolic_lcs()` dropping either
# forward. These tests therefore run on a flow map whose stretching oscillates
# in longitude, so the FTLE has real maxima and both parameters bite (see the
# sweeps asserted in `test_hyperbolic_lcs_ridge_parameters_are_forwarded`).

LCS_LON = np.linspace(-3.0, 3.0, 25)
LCS_LAT = np.linspace(18.0, 22.0, 11)
LCS_T1 = T0 + np.timedelta64(7, "D")

LCS_KWARGS = {
    "window_m": 150_000.0,
    # Deliberately not ftle_ridge_seeds' own default of 0.90.
    "quantile": 0.5,
    "step_m": 10_000.0,
    "line_length_m": 30_000.0,
}


def _wavy_stretch_flowmap(period_m=250_000.0, base=3.0, amp=1.0):
    """A flow map whose meridional stretching oscillates with zonal position.

    ``f(dx, dy) = (dx, dy * (base + amp cos(2 pi dx / period_m)))``, so the
    Cauchy-Green tensor -- and hence the FTLE -- varies along ``i`` with several
    maxima across the domain. That makes the windowed local-maximum test and the
    quantile floor both load-bearing, unlike a constant-``C`` fixture.
    """

    def f(dx, dy):
        return dx, dy * (base + amp * np.cos(2.0 * np.pi * dx / period_m))

    return advected_flowmap_f(AuxiliarySeed, LCS_LON, LCS_LAT, f, T0, LCS_T1)


def test_hyperbolic_lcs_matches_the_manual_pipeline():
    """hyperbolic_lcs() is exactly ftle -> ftle_ridge_seeds -> shrink_lines,
    plus the FTLE field itself."""
    fm = _wavy_stretch_flowmap()

    ftle = fm.ftle()
    seed_lon, seed_lat = ftle_ridge_seeds(
        ftle, window_m=LCS_KWARGS["window_m"], quantile=LCS_KWARGS["quantile"]
    )
    manual = shrink_lines(
        fm,
        seed_lon=seed_lon,
        seed_lat=seed_lat,
        step_m=LCS_KWARGS["step_m"],
        line_length_m=LCS_KWARGS["line_length_m"],
    )

    lcs = fm.hyperbolic_lcs(**LCS_KWARGS)

    assert set(lcs.data_vars) == {"lon", "lat", "ftle"}
    assert lcs.sizes["line"] == seed_lon.size > 1  # a real, partial seed selection
    for name in ("lon", "lat"):
        np.testing.assert_array_equal(lcs[name].values, manual[name].values)
    np.testing.assert_array_equal(lcs["ftle"].values, ftle.values)


def test_hyperbolic_lcs_ridge_parameters_are_forwarded():
    """Both ridge parameters reach ``ftle_ridge_seeds``: tightening either one
    yields strictly fewer lines, so a dropped forward cannot pass unnoticed."""
    fm = _wavy_stretch_flowmap()
    trace = {
        "step_m": LCS_KWARGS["step_m"],
        "line_length_m": LCS_KWARGS["line_length_m"],
    }

    loose = fm.hyperbolic_lcs(window_m=60_000.0, quantile=0.5, **trace)
    tight_quantile = fm.hyperbolic_lcs(window_m=60_000.0, quantile=0.99, **trace)
    tight_window = fm.hyperbolic_lcs(window_m=400_000.0, quantile=0.5, **trace)

    assert tight_quantile.sizes["line"] < loose.sizes["line"]
    assert tight_window.sizes["line"] < loose.sizes["line"]


def test_hyperbolic_lcs_computes_the_ftle_once(lon_axis, lat_axis, monkeypatch):
    """The FTLE is computed a single time and handed to the ridge finder."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    calls = []
    original = type(fm).ftle

    def counting_ftle(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(type(fm), "ftle", counting_ftle)

    fm.hyperbolic_lcs(**LCS_KWARGS)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("t0", "t1", "kind"), [(T0, T1, "repelling"), (T1, T0, "attracting")]
)
def test_hyperbolic_lcs_metadata_names_the_lcs_type(lon_axis, lat_axis, t0, t1, kind):
    """A forward flow map yields repelling LCS, a backward one attracting ones,
    and the returned dataset says which without being asked."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), t0, t1
    )

    lcs = fm.hyperbolic_lcs(**LCS_KWARGS)

    assert kind in lcs.attrs["long_name"]
    assert kind in lcs["lon"].attrs["long_name"]
    assert kind in lcs["lat"].attrs["long_name"]
    assert lcs["ftle"].attrs["units"] == "1/s"


def test_shrink_lines_seed_pair_is_keyword_only(lon_axis, lat_axis):
    """Passing the seed lon/lat pair positionally raises, so a swap cannot pass
    silently -- including the ``*ftle_ridge_seeds(...)`` unpacking form."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    seed = _centre_seed(fm)
    with pytest.raises(TypeError):
        shrink_lines(fm, seed["seed_lon"], seed["seed_lat"])
