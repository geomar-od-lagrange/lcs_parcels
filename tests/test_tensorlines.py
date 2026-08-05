"""Tensor-line tests: ftle_ridge_seeds and shrink_lines.

``conftest.advected_flowmap`` advects a seed through a constant linear map ``M``
acting in the one tangent frame at the seed centroid (``AuxiliarySeedGrid``, so
``gradF`` is defined at every grid point and there are no NaN edges). The package
measures every separation in the local east/north frame of the pair it connects,
so what it recovers is ``M`` rescaled by the release and arrival cosines
(``conftest.local_frame_gradient``). For a diagonal ``M`` the rescaling is
diagonal too and touches the east component only, by ``cos(arrival) /
cos(release)``. That factor is within a percent of 1 over the reference band and
a good deal further from it at 70 N, where the same meridional map moves a point
through a much larger change in cosine, so a test that pins an absolute
eigenvalue states the factor rather than absorbing it in a tolerance.

``M = diag(1, 3)`` gives a diagonal ``C`` whose ``xi_1`` points exactly due east
whatever that factor is: the shrink line through a seed is that seed's parallel
of latitude, a closed-form check.

``_local_frame_flowmap`` builds the other closed-form case, a flow map whose
``C`` is the *same* in every local frame. Its ``xi_1`` then has one constant
compass bearing, so the shrink line is a loxodrome and can be compared against
the classical formula.
"""

import warnings

import numpy as np
import pytest
import xarray as xr
from conftest import advected_flowmap, advected_flowmap_f, seed_origin
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels import AuxiliarySeedGrid, ftle_ridge_seeds, shrink_lines
from lcs_parcels.grids import _M_PER_DEG, _separation_m
from lcs_parcels.tensorlines import (
    _shrink_line_tangent,
    _step_lonlat_by_meters,
    _trace_half_line,
    _window_geometry,
)

RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")


# --- ftle_ridge_seeds ------------------------------------------------------


def _gridded_field(values, lon_axis, lat_axis):
    """``values`` on ``(i, j)`` with the ``lon_grid``/``lat_grid`` coords
    :func:`ftle_ridge_seeds` reads."""
    lon2d, lat2d = xr.broadcast(
        xr.DataArray(lon_axis, dims="i"), xr.DataArray(lat_axis, dims="j")
    )
    return xr.DataArray(
        values,
        dims=("i", "j"),
        coords={
            "lon_grid": (("i", "j"), lon2d.values),
            "lat_grid": (("i", "j"), lat2d.values),
        },
    )


def _cells(geometry):
    """The ``(i, j)`` window cell counts of a
    :func:`~lcs_parcels.tensorlines._window_geometry` dict -- or of a seed
    dataset's attrs, which carry the same keys."""
    return geometry["window_cells_i"], geometry["window_cells_j"]


def _three_cell_window_m(field):
    """A ``window_m`` worth at least three cells in *both* dimensions of ``field``.

    ``window_m`` is one metre length against a grid whose two cell sizes need not
    match: at 72 N the fixture's 1 x 2 degree cells are 34 km by 222 km, so a
    window chosen for the zonal spacing is a single cell meridionally, and every
    grid point is then trivially a local maximum along ``j``. Three times the
    *larger* median spacing clears three cells either way. That is
    :func:`~lcs_parcels.tensorlines._window_geometry` behaving as documented --
    one median per dimension -- so the test states the grid it wants rather than
    a number that happens to suit one latitude.
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
    return 3.0 * max(float(np.abs(dx).median()), float(np.abs(dy).median()))


def test_ftle_ridge_seeds_picks_the_peak(lon_axis, lat_axis):
    """A single smooth FTLE bump yields exactly its peak grid point as the seed."""
    ii, jj = np.meshgrid(
        np.arange(lon_axis.size), np.arange(lat_axis.size), indexing="ij"
    )
    ftle = _gridded_field(
        np.exp(-((ii - 2.0) ** 2 + (jj - 2.0) ** 2)), lon_axis, lat_axis
    )

    window_m = _three_cell_window_m(ftle)
    assert min(_cells(_window_geometry(ftle, window_m))) >= 3
    seeds = ftle_ridge_seeds(ftle, window_m=window_m, quantile=0.90)

    assert seeds.sizes["seed"] == 1
    assert seeds["lon"].values[0] == lon_axis[2]
    assert seeds["lat"].values[0] == lat_axis[2]


def test_ftle_ridge_seeds_skips_nan(lon_axis, lat_axis):
    """NaN cells never qualify as seeds."""
    field = np.full((lon_axis.size, lat_axis.size), np.nan)
    field[1, 1] = 5.0  # a lone finite peak
    ftle = _gridded_field(field, lon_axis, lat_axis)

    seeds = ftle_ridge_seeds(ftle, window_m=_three_cell_window_m(ftle), quantile=0.5)

    assert seeds["lon"].values.tolist() == [lon_axis[1]]
    assert seeds["lat"].values.tolist() == [lat_axis[1]]


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

    coarse_geometry = _window_geometry(coarse, 60_000.0)
    fine_geometry = _window_geometry(fine, 60_000.0)

    assert _cells(coarse_geometry) == (5, 5)
    assert _cells(fine_geometry) == (11, 11)
    # The same physical window, so the same physical seed spacing on both grids.
    np.testing.assert_allclose(
        fine_geometry["min_seed_separation_m"],
        coarse_geometry["min_seed_separation_m"],
        rtol=0.05,
    )


def test_window_cell_count_is_per_dimension_and_odd():
    """Each dimension gets its own count, from the median local east/north
    spacing of that dimension, rounded down to an odd number.

    A mid-latitude grid of 0.1 deg by 0.05 deg cells separates the three things an
    equatorial isotropic grid hides. The zonal spacing shrinks poleward across
    this grid and its median cell sits at 40 N, so the spacings are
    ``dx = 0.1 * 111195 * cos(40) = 8518 m`` and ``dy = 0.05 * 111195 = 5560 m``:
    a 50 km window is ``round(5.87) = 6 -> 5`` cells along ``i`` (the
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

    geometry = _window_geometry(ftle, 50_000.0)

    assert _cells(geometry) == (5, 9)
    np.testing.assert_allclose(geometry["grid_spacing_i_m"], 8518.0, rtol=1e-3)
    np.testing.assert_allclose(geometry["grid_spacing_j_m"], 5560.0, rtol=1e-3)
    # (5 - 1) // 2 + 1 = 3 cells along i, (9 - 1) // 2 + 1 = 5 along j, and the
    # reported value is the smaller of the two. It is built on the SMALLEST cell,
    # not the median one: the zonal cell shrinks to 0.1 * 111195 * cos(50) =
    # 7148 m at the poleward edge, so the bound is 3 * 7148 against 5 * 5560.
    # Using the median 8518 here would claim 25554 m and be violated by the
    # seeds at 50 N.
    np.testing.assert_allclose(
        geometry["min_seed_separation_m"], 3.0 * 7148.0, rtol=1e-3
    )
    assert geometry["window_m"] == 50_000.0


@pytest.mark.parametrize("n_lon", [61, 121])
def test_ftle_ridge_seeds_selectivity_is_physical(n_lon):
    """Resolving or merging the two bumps depends on window_m, not on cell size:
    a window narrower than their 111 km separation keeps both peaks, a wider one
    keeps only the stronger -- identically on the coarse and the fine grid."""
    ftle = _two_bump_ftle(n_lon)

    narrow = ftle_ridge_seeds(ftle, window_m=60_000.0)
    wide = ftle_ridge_seeds(ftle, window_m=300_000.0)

    assert np.allclose(np.sort(narrow["lon"].values), [-0.5, 0.5])
    # only the stronger bump survives
    assert np.allclose(wide["lon"].values, [-0.5])


def test_ftle_ridge_seeds_output_structure_and_attrs():
    """The returned dataset is ``lon``/``lat`` on an indexed ``seed`` dim, each
    labelled, and its attrs report what ``window_m`` became on this grid."""
    seeds = ftle_ridge_seeds(_two_bump_ftle(61), window_m=60_000.0)

    assert set(seeds.dims) == {"seed"}
    assert set(seeds.data_vars) == {"lon", "lat"}
    assert seeds["seed"].values.tolist() == list(range(seeds.sizes["seed"]))
    assert seeds["seed"].attrs["long_name"]
    assert seeds["lon"].attrs["units"] == "degrees_east"
    assert seeds["lat"].attrs["units"] == "degrees_north"
    assert seeds["lon"].attrs["long_name"] != seeds["lat"].attrs["long_name"]
    assert set(seeds.attrs) == {
        "long_name",
        "selector",
        "ftle_threshold",
        "window_m",
        "window_cells_i",
        "window_cells_j",
        "grid_spacing_i_m",
        "grid_spacing_j_m",
        "min_seed_separation_m",
    }
    assert seeds.attrs["selector"] == "quantile"
    assert seeds.attrs["window_m"] == 60_000.0


def _lattice_ridge_ftle(window_m, lat_max=1.0):
    """A field whose maxima are packed as tightly as ``window_m`` allows.

    Crests sit on a lattice whose period, per dimension, is exactly
    ``(cells - 1) // 2 + 1`` cells -- one cell beyond the window's reach, the
    closest two windowed maxima can be. Any tighter and one crest would fall
    inside the other's window and only the larger would survive, so this is the
    extreme case ``min_seed_separation_m`` claims to bound rather than a field
    that merely happens to stay clear of it.

    ``cos`` in each index separately keeps the field separable, so the rolling
    2-D maximum is the sum of the two 1-D maxima and a point is a seed exactly
    when it is a crest in both dimensions. Returns the field and its geometry.

    ``lat_max`` sets how far the grid reaches poleward from the equator. The
    lattice is uniform in *index* space while its cells shrink in metres, so a
    wide band is what separates the median cell size from the smallest one.
    """
    lon_axis = np.arange(0.0, 6.0001, 0.1)
    lat_axis = np.arange(0.0, lat_max + 1e-9, 0.1)
    ii, jj = np.meshgrid(
        np.arange(lon_axis.size), np.arange(lat_axis.size), indexing="ij"
    )
    geometry = _window_geometry(_gridded_field(ii * 0.0, lon_axis, lat_axis), window_m)
    period_i = (geometry["window_cells_i"] - 1) // 2 + 1
    period_j = (geometry["window_cells_j"] - 1) // 2 + 1
    values = np.cos(2.0 * np.pi * ii / period_i) + np.cos(2.0 * np.pi * jj / period_j)
    return _gridded_field(values, lon_axis, lat_axis), geometry


@pytest.mark.parametrize("lat_max", [1.0, 10.0, 30.0, 60.0, 75.0])
def test_min_seed_separation_m_bounds_the_returned_seeds(lat_max):
    """No two seeds are closer than the reported ``min_seed_separation_m``, and
    they are a good deal closer than ``window_m``.

    Measured with :func:`~lcs_parcels.grids._separation_m`, the same frame the
    package works in. The field packs its crests at exactly the reported
    separation, so the bound is attained and not merely respected: the measured
    closest pair is about ``window_m / 2``, which is the whole content of #21 --
    reporting ``window_m`` here would overstate the spacing twofold and the
    second assertion would fail.

    Parametrized over the latitude band because the bound is built on the
    smallest cell rather than the median one, and only a wide band separates the
    two. A median-based bound passes at ``lat_max=1`` and is violated by 11% at
    30 degrees and by a factor of 3 at 75.
    """
    window_m = 60_000.0
    ftle, geometry = _lattice_ridge_ftle(window_m, lat_max=lat_max)
    seeds = ftle_ridge_seeds(ftle, window_m=window_m, quantile=0.0)
    assert seeds.sizes["seed"] > 20  # several ridges, not one

    a = seeds.rename(seed="seed_a")
    b = seeds.rename(seed="seed_b")
    dx, dy = _separation_m(
        lon_a=a["lon"], lat_a=a["lat"], lon_b=b["lon"], lat_b=b["lat"]
    )
    distance = np.hypot(dx, dy)
    closest = float(distance.where(distance > 0.0).min())

    # The lattice is laid out in index space and the pairs are measured in
    # metres, so the two agree only to the grid's own rounding; 1e-3 is slack
    # enough for that and nothing else.
    assert closest > geometry["min_seed_separation_m"] * (1.0 - 1e-3)
    assert closest < 0.75 * window_m  # and the bound is near window_m / 2


def test_min_seed_separation_m_bounds_strict_maxima_only():
    """A plateau of exactly equal values puts seeds closer than the bound.

    Selection is ``ftle >= rolling max``, so every cell of a flat top ties for
    the maximum and adjacent cells are all seeds. Nothing distinguishes a point
    on a flat ridge, so they are kept rather than broken arbitrarily, and the
    docstring says the bound is on *strict* maxima. Asserted here so the
    exception is a documented behaviour rather than an unnoticed one.
    """
    lon_axis = np.arange(0.0, 3.0001, 0.1)
    lat_axis = np.arange(0.0, 3.0001, 0.1)
    values = np.zeros((lon_axis.size, lat_axis.size))
    values[10:14, 10:14] = 1.0  # a 4x4 plateau, all exactly equal
    ftle = _gridded_field(values, lon_axis, lat_axis)

    seeds = ftle_ridge_seeds(ftle, window_m=60_000.0, quantile=0.99)
    geometry = _window_geometry(ftle, 60_000.0)

    # The whole plateau is selected, and its cells are adjacent.
    assert seeds.sizes["seed"] == 16
    a = seeds.rename(seed="seed_a")
    b = seeds.rename(seed="seed_b")
    dx, dy = _separation_m(
        lon_a=a["lon"], lat_a=a["lat"], lon_b=b["lon"], lat_b=b["lat"]
    )
    distance = np.hypot(dx, dy)
    closest = float(distance.where(distance > 0.0).min())
    assert closest < 0.5 * geometry["min_seed_separation_m"]


def test_ftle_ridge_seeds_warns_when_the_window_is_under_three_cells():
    """A window narrower than three cells makes every point a windowed maximum,
    so the local-maximum test stops selecting and the caller is told."""
    ftle = _two_bump_ftle(61)  # about 11 km cells

    with pytest.warns(UserWarning, match="three cells"):
        seeds = ftle_ridge_seeds(ftle, window_m=1_000.0)

    assert _cells(seeds.attrs) == (1, 1)


def test_ftle_ridge_seeds_is_silent_on_a_resolved_window():
    """A comfortably resolved window warns about nothing -- the warning marks a
    real degeneracy and does not fire on the ordinary call."""
    ftle = _two_bump_ftle(61)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        seeds = ftle_ridge_seeds(ftle, window_m=60_000.0)

    assert min(_cells(seeds.attrs)) >= 3


def test_ftle_min_reproduces_the_matching_quantile():
    """``ftle_min`` set to this field's measured 0.90 quantile selects exactly the
    seed set ``quantile=0.90`` does, and a higher floor selects strictly fewer.

    The two selectors are one floor reached two ways, so at the same floor they
    must not merely agree in count. The higher floor is 0.95, between the two
    bumps' 0.9 and 1.0, so it drops the weaker one.
    """
    ftle = _two_bump_ftle(61)
    threshold = float(ftle.quantile(0.90))

    by_quantile = ftle_ridge_seeds(ftle, window_m=60_000.0, quantile=0.90)
    by_value = ftle_ridge_seeds(ftle, window_m=60_000.0, ftle_min=threshold)
    higher = ftle_ridge_seeds(ftle, window_m=60_000.0, ftle_min=0.95)

    np.testing.assert_array_equal(by_value["lon"].values, by_quantile["lon"].values)
    np.testing.assert_array_equal(by_value["lat"].values, by_quantile["lat"].values)
    assert by_value.attrs["selector"] == "ftle_min"
    assert by_quantile.attrs["selector"] == "quantile"
    assert by_value.attrs["ftle_threshold"] == threshold
    assert 0 < higher.sizes["seed"] < by_value.sizes["seed"]


def test_ftle_min_is_the_same_floor_on_two_fields_and_a_quantile_is_not():
    """The property a quantile cannot give (#22): one ``ftle_min`` means one
    absolute threshold across fields with different FTLE distributions, while one
    ``quantile`` follows each field's own distribution.

    ``weak`` is ``strong`` scaled by a tenth, so their quantiles pick out the same
    *points* at ten times apart values -- the case where a run comparing two
    windows or two regions on one absolute criterion gets nothing from a
    quantile. A floor of 0.5 sits above every value of ``weak`` and below the
    crest of ``strong``.
    """
    strong = _two_bump_ftle(61)
    weak = 0.1 * strong

    q_strong = ftle_ridge_seeds(strong, window_m=60_000.0, quantile=0.90)
    q_weak = ftle_ridge_seeds(weak, window_m=60_000.0, quantile=0.90)
    a_strong = ftle_ridge_seeds(strong, window_m=60_000.0, ftle_min=0.5)
    a_weak = ftle_ridge_seeds(weak, window_m=60_000.0, ftle_min=0.5)

    # The quantile threshold tracks each field; the absolute one does not.
    assert q_strong.attrs["ftle_threshold"] != q_weak.attrs["ftle_threshold"]
    assert a_strong.attrs["ftle_threshold"] == a_weak.attrs["ftle_threshold"] == 0.5

    # ... so the quantile selects the same points on both, the floor does not.
    np.testing.assert_array_equal(q_strong["lon"].values, q_weak["lon"].values)
    assert q_strong.sizes["seed"] > 0
    assert a_strong.sizes["seed"] > 0
    assert a_weak.sizes["seed"] == 0


def test_ftle_ridge_seeds_rejects_both_selectors():
    """``quantile`` and ``ftle_min`` are two ways of setting one floor, so asking
    for both is an error rather than a silent precedence rule."""
    ftle = _two_bump_ftle(61)

    with pytest.raises(ValueError):
        ftle_ridge_seeds(ftle, window_m=60_000.0, quantile=0.90, ftle_min=0.5)


def test_ftle_ridge_seeds_defaults_to_the_ninetieth_percentile():
    """Giving neither selector is exactly ``quantile=0.90``, seeds and threshold
    alike -- the default the package had before ``ftle_min`` existed."""
    ftle = _two_bump_ftle(61)

    default = ftle_ridge_seeds(ftle, window_m=60_000.0)
    explicit = ftle_ridge_seeds(ftle, window_m=60_000.0, quantile=0.90)

    assert default.attrs["selector"] == "quantile"
    assert default.attrs["ftle_threshold"] == explicit.attrs["ftle_threshold"]
    np.testing.assert_array_equal(default["lon"].values, explicit["lon"].values)
    np.testing.assert_array_equal(default["lat"].values, explicit["lat"].values)


# --- lifted integrator internals -------------------------------------------
#
# These take the state that used to be closed over (the interpolator, the
# anisotropy floor, the step) as explicit arguments, so they can be driven from
# an analytic tensor field without building a FlowMap.

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
    """A unit direction moves step_m metres, measured in the local frame of the step.

    Exactly step_m, not step_m to a truncation: the step is the algebraic inverse
    of ``_separation_m``, sharing its mid-latitude cosine, so measuring the step
    with the same formula that defines it returns the length asked for to
    round-off. Measured residual 1.6e-15 relative at a 25 km step.
    """
    lon_0, lat_0 = np.array([0.0]), np.array([20.0])
    direction = np.array([[np.cos(0.7), np.sin(0.7)]])

    lon1, lat1 = _step_lonlat_by_meters(lon_0, lat_0, direction, step_m=25_000.0)

    dx, dy = _separation_m(lon_a=lon_0, lat_a=lat_0, lon_b=lon1, lat_b=lat1)
    np.testing.assert_allclose(np.hypot(dx, dy), 25_000.0, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("lat", [30.0, 60.0, 80.0])
def test_step_lonlat_spends_more_degrees_at_higher_latitude(lat):
    """A due-east step at latitude L costs exactly ``1 / cos(L)`` times the degrees
    it costs at the equator, and moves no latitude at all.

    The step inverts ``_separation_m`` at the point's *own* latitude -- there is
    no reference latitude left to pass -- and a due-east direction has a zero
    north component, so its mid-latitude is the starting latitude and the
    ``1 / cos`` is the whole of the relation, not its leading term. A
    great-circle step would leave a ``(step_m / R)^2`` correction on the ratio
    and a curvature sag on the latitude; both are zero here.
    """
    lon_0 = np.array([0.0])
    east = np.array([[1.0, 0.0]])

    lon_equator, _ = _step_lonlat_by_meters(
        lon_0, np.array([0.0]), east, step_m=25_000.0
    )
    lon_polar, lat_polar = _step_lonlat_by_meters(
        lon_0, np.array([lat]), east, step_m=25_000.0
    )

    assert lon_polar[0] > lon_equator[0]
    np.testing.assert_allclose(
        lon_polar[0] * np.cos(np.deg2rad(lat)), lon_equator[0], rtol=1e-14, atol=0.0
    )
    assert lat_polar[0] == lat


def test_step_lonlat_crosses_the_antimeridian_on_the_seed_branch():
    """A step east from 179.9 lands at 180.1, not at -179.9.

    The longitude increment is added to the incoming longitude, so a track keeps
    the branch its seed came in on instead of being folded into (-180, 180].
    """
    lon_0, lat_0 = np.array([179.9]), np.array([0.0])
    east = np.array([[1.0, 0.0]])

    lon1, lat1 = _step_lonlat_by_meters(lon_0, lat_0, east, step_m=0.2 * _M_PER_DEG)

    # Due east on the equator is the equator itself, so the arc is exact here:
    # both residuals measure 0.0, and 1e-12 degrees is 0.1 micrometre of slack.
    np.testing.assert_allclose(lon1, 180.1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(lat1, 0.0, rtol=0.0, atol=1e-12)


def _turning_tensor(turn_per_degree, *, half_extent_lon_deg=30.0):
    """``C`` whose ``xi_1`` direction turns with longitude, as a callable.

    ``C(lon) = R(theta) diag(1, 9) R(theta)^T`` with ``theta = turn_per_degree *
    lon``, so ``xi_1`` is ``(cos theta, sin theta)`` and a curve tracing it bends
    as it advances. Every uniform-``C`` fixture above hides the integration
    scheme: there the midpoint direction equals the direction at the current
    point, so RK2 and plain Euler produce bit-identical tracks.

    Exposes the ``RegularGridInterpolator`` call interface but evaluates exactly,
    so a measured error is the integrator's own rather than the tensor
    interpolation's. Longitudes beyond ``half_extent_lon_deg`` return NaN,
    standing in for leaving the grid; latitude is unbounded, so the same field
    can be traced at any latitude.
    """

    def evaluate(points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        theta = turn_per_degree * points[:, 0]
        c, s = np.cos(theta), np.sin(theta)
        out = np.empty((points.shape[0], 2, 2))
        out[:, 0, 0] = c * c + 9.0 * s * s
        out[:, 0, 1] = out[:, 1, 0] = -8.0 * c * s
        out[:, 1, 1] = s * s + 9.0 * c * c
        out[np.abs(points[:, 0]) > half_extent_lon_deg] = np.nan
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


@pytest.mark.parametrize("seed_lat", [0.0, 45.0, 70.0])
def test_trace_half_line_is_second_order_in_the_step(seed_lat):
    """RK2, not Euler: halving the step cuts the endpoint change about fourfold.

    Traces the same arc length at three step sizes on a tensor field whose
    ``xi_1`` turns, and compares successive endpoints. Second order gives a ratio
    near 4; first-order Euler gives near 2.

    Run off the equator as well as on it. At the equator the step's own
    latitude-dependent term is identically zero, so an equatorial trace alone
    cannot see a stepper that is first-order in latitude -- which is what the
    great-circle step tried before was, at ``(step / R)^2 tan(phi) / 2`` per step
    accumulating linearly in the step count. Measured ratios are 3.97 at the
    equator, 4.04 at 45 N and 4.24 at 70 N.
    """
    tensor = _turning_tensor(0.5)
    arc_m = 300_000.0

    def endpoint(n_steps):
        track = _trace_half_line(
            np.array([0.0]),
            np.array([seed_lat]),
            +1,
            tensor_interp=tensor,
            min_anisotropy=1.1,
            step_m=arc_m / n_steps,
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
    """M = diag(1, 3) => xi_1 is due east => the shrink line is a parallel of latitude.

    Exactly a parallel, not one to within a sag. ``xi_1`` here has a zero north
    component, and the step's latitude increment is that component times the step
    -- so every point of the line carries the seed's latitude bit for bit, at any
    step size and at any latitude. Measured ``ptp(lat)`` is 0.0 in all three
    regions. A great-circle step would instead have bent each arc back towards
    the equator by ``(step_m / R)^2 tan(lat) / 2``, which is what the 1e-4
    tolerance this assertion used to carry was absorbing.
    """
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([1.0, 3.0]),
        RELEASE_TIME,
        END_TIME,
    )
    lines = shrink_lines(
        fm, **_centre_seed(fm), step_m=10_000.0, line_length_m=80_000.0
    )

    lon = lines["lon"].isel(line=0).values
    lat = lines["lat"].isel(line=0).values
    valid = np.isfinite(lon) & np.isfinite(lat)

    assert valid.all()  # short line stays on the grid
    assert np.ptp(lat[valid]) < 1e-12  # a parallel, to round-off
    assert np.ptp(lon[valid]) > 0.1  # and spans in longitude


def test_shrink_lines_output_structure(lon_axis, lat_axis):
    """Dataset has lon/lat on (line, point); one line per seed, an odd point count."""
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([1.0, 3.0]),
        RELEASE_TIME,
        END_TIME,
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
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, np.eye(2), RELEASE_TIME, END_TIME
    )
    lines = shrink_lines(
        fm, **_centre_seed(fm), min_anisotropy=1.15, line_length_m=30_000.0
    )

    assert bool(lines["lon"].isnull().all())


def test_shrink_lines_default_guard_stops_a_barely_anisotropic_tensor(
    lon_axis, lat_axis
):
    """The *default* ``min_anisotropy`` is 1.15, and nothing weaker: this map's
    ratio sits between 1 and 1.15, so a default-path call must kill the line.

    ``M = diag(1.05, 1.0)`` gives ``C = diag(1.1025, 1.0)`` -- a ratio of 1.1025,
    barely anisotropic, where ``xi_1`` is a direction only to within a large
    perturbation of ``C``. Passing no ``min_anisotropy`` at all is the point:
    every other guard test states a floor explicitly, so they pin the argument
    and leave the default free to drift down to 1.0 (guard off) unnoticed.
    """
    a, b = 1.05, 1.0
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, np.diag([a, b]), RELEASE_TIME, END_TIME
    )

    lam = fm.cg_eigen()["lambda"]
    ratio = float((lam.isel(eig=1) / lam.isel(eig=0)).mean())
    assert 1.0 < ratio < 1.15  # the band that only the default can decide
    assert np.isclose(ratio, (a / b) ** 2)

    lines = shrink_lines(fm, **_centre_seed(fm), step_m=3_000.0, line_length_m=30_000.0)

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
    t1 = RELEASE_TIME + sign * np.timedelta64(int(t_days * 24), "h")
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, np.diag([a, b]), RELEASE_TIME, t1
    )

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

    ``M = diag(0.5, 0.4)`` compresses in both directions, so ``lambda_1 = 0.16``
    exactly (north is unrescaled) and ``lambda_2`` is ``0.25`` carrying the east
    component's ``cos(arrival) / cos(release)``, squared. That factor is 1 to
    within a percent over the reference band but not over the 68-76 N one, where
    ``M``'s meridional compression moves each point far enough in latitude to
    change its cosine: ``lambda_2`` runs from 0.2005 to 0.3405 there, against the
    closed form below. The ratio bottoms out at 1.25, still clear of the 1.15
    default, and ``xi_1`` is as well defined here as in any stretching flow -- so
    the line must survive. Any floor on the magnitude of ``lambda_2`` at or above
    1 would kill it.

    Not a synthetic corner: ``det grad F`` is about 0.2 here, and in the backward
    Cabo Verde example (5-day window, measured 34 km clear of any coast) the
    0.1st percentile of ``det grad F`` is 0.196. Convergent patches this strong
    are rare but real, and they are where attracting LCS live -- which is why a
    magnitude floor terminating there is a directional bias, not just a
    conservative choice.
    """
    a, b = 0.5, 0.4
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, np.diag([a, b]), RELEASE_TIME, END_TIME
    )

    # The map acts in the one tangent frame at the seed centroid, so a grid point
    # at latitude phi arrives at lat_origin + b * (phi - lat_origin).
    phi = fm.ds["lat_grid"]
    lat_origin = seed_origin(fm)[1]
    arrival = lat_origin + b * (phi - lat_origin)
    lambda_2 = (a * np.cos(np.deg2rad(arrival)) / np.cos(np.deg2rad(phi))) ** 2

    # Worst measured relative residuals across the three regions: 2.4e-12 on
    # lambda_1 and 1.2e-11 on lambda_2, both in the antimeridian/high-latitude
    # runs. 1e-9 keeps about two orders of margin over those.
    lam = fm.cg_eigen()["lambda"]
    np.testing.assert_allclose(lam.isel(eig=0), b**2, rtol=1e-9, atol=0.0)
    np.testing.assert_allclose(lam.isel(eig=1), lambda_2, rtol=1e-9, atol=0.0)
    assert bool((lam.isel(eig=1) / lam.isel(eig=0) > 1.15).all())

    lines = shrink_lines(fm, **_centre_seed(fm), step_m=3_000.0, line_length_m=30_000.0)

    assert bool(lines["lon"].notnull().all())


def test_shrink_lines_seed_off_grid_is_nan(lon_axis, lat_axis):
    """A seed outside the grid produces an all-NaN line."""
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([1.0, 3.0]),
        RELEASE_TIME,
        END_TIME,
    )
    lines = shrink_lines(
        fm,
        seed_lon=[lon_axis[0] - 50.0],
        seed_lat=[lat_axis[0] - 50.0],
        line_length_m=30_000.0,
    )

    assert bool(lines["lon"].isnull().all())
    assert bool(lines["lat"].isnull().all())


def test_shrink_line_steps_in_the_frame_the_tensor_lives_in():
    """The stepper and the tensor share one frame: every segment of a traced line
    runs along the local ``xi_1`` measured at that segment's midpoint.

    ``C`` is built from separations in the local east/north frame of each pair
    (:func:`~lcs_parcels.grids._separation_m`), so a segment must be measured the
    same way -- which is what this checks, by taking ``_separation_m`` between
    consecutive points of the line and comparing its heading with the tangent the
    integrator would read there. Measured in the same frame the tensor is built
    in, the two agree to round-off -- 2e-6 degrees of arc. Measuring the same
    segments in a single-reference-cosine frame at 20 N instead tilts them off
    ``xi_1`` by up to 1.7 degrees, a million times as much.

    ``M = R(45) diag(1, 3) R(45)^T`` puts ``xi_1`` near the 45-degree diagonal, so
    the line climbs from 12 N to 28 N and the two frames have every chance to
    disagree.
    """
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c], [c, c]])
    M = R @ np.diag([1.0, 3.0]) @ R.T
    lon_axis = np.linspace(-15.0, 15.0, 31)
    lat_axis = np.linspace(0.0, 40.0, 41)
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )

    lines = shrink_lines(
        fm, seed_lon=[0.0], seed_lat=[20.0], step_m=20_000.0, line_length_m=2_400_000.0
    )
    lon = lines["lon"].isel(line=0).values
    lat = lines["lat"].isel(line=0).values
    valid = np.isfinite(lon) & np.isfinite(lat)
    assert valid.sum() > 50  # the diagonal line stays on this wide grid
    lon, lat = lon[valid], lat[valid]

    dx, dy = _separation_m(lon_a=lon[:-1], lat_a=lat[:-1], lon_b=lon[1:], lat_b=lat[1:])
    segment = np.column_stack([dx, dy]) / np.hypot(dx, dy)[:, None]
    tangent = _shrink_line_tangent(
        0.5 * (lon[:-1] + lon[1:]),
        0.5 * (lat[:-1] + lat[1:]),
        segment,
        tensor_interp=RegularGridInterpolator(
            (fm.lon_grid.isel(j=0).values, fm.lat_grid.isel(i=0).values),
            fm.cauchy_green().transpose("i", "j", "row", "col").values,
        ),
        min_anisotropy=1.15,
    )

    # Worst measured ``1 - dot`` is 4.4e-16, two ulp; the single-cosine frame
    # above gives 4.3e-4, so 1e-12 sits far from both.
    np.testing.assert_allclose(
        np.sum(segment * tangent, axis=1), 1.0, rtol=0.0, atol=1e-12
    )


def _local_frame_flowmap(lon_axis, lat_axis, M):
    """A flow map whose Cauchy-Green tensor is ``M^T M`` in *every* local frame.

    ``conftest.advected_flowmap`` applies ``M`` in one tangent frame for the whole
    grid, so the tensor it produces varies over the grid (that is
    ``conftest.local_frame_gradient``). Here ``M`` is instead applied in each grid
    point's own east/north frame: read each arm's offset from its grid point in
    that frame, map it with ``M``, and place the advected arm back at the mapped
    offset. The arithmetic is written out rather than taken from
    ``lcs_parcels.grids``, since what the frame *is* is the thing under test.
    """
    seed = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon_c, lat_c = seed.ds["lon_grid"], seed.ds["lat_grid"]
    lon_a, lat_a = seed.ds["lon_0"], seed.ds["lat_0"]

    m_per_deg = _M_PER_DEG
    dx = m_per_deg * np.cos(np.deg2rad(0.5 * (lat_a + lat_c))) * (lon_a - lon_c)
    dy = m_per_deg * (lat_a - lat_c)
    dx_out = M[0, 0] * dx + M[0, 1] * dy
    dy_out = M[1, 0] * dx + M[1, 1] * dy

    dlat = dy_out / m_per_deg
    lat_out = lat_c + dlat
    lon_out = lon_c + dx_out / (m_per_deg * np.cos(np.deg2rad(lat_c + 0.5 * dlat)))

    dims = seed.ds["lon_0"].dims
    return seed.pset_to_flowmap(
        lon=lon_out.transpose(*dims).stack(particle=dims).values,
        lat=lat_out.transpose(*dims).stack(particle=dims).values,
        t0=RELEASE_TIME,
        t1=END_TIME,
    )


def test_shrink_line_of_a_frame_constant_tensor_is_a_loxodrome():
    """A shrink line whose ``xi_1`` holds one compass bearing is the classical
    loxodrome, and the traced line matches its closed form off the equator.

    Independent of the package throughout: the flow map is built so that ``C`` is
    ``M^T M`` in every local frame, ``M = R(45) diag(1, 3) R(45)^T`` puts ``xi_1``
    on a constant 45-degree bearing, and a curve of constant bearing satisfies
    ``d(lambda) / d(phi) = tan(bearing) / cos(phi)``, i.e.
    ``lambda = lambda_0 + tan(bearing) * (psi(phi) - psi(phi_0))`` with ``psi``
    the inverse Gudermannian ``log(tan(pi/4 + phi/2))``. Nothing in that came
    from ``lcs_parcels``.

    A 3000 km line from 40 N, spanning 30.5 N to 49.5 N, sits 8.4e-6 degrees --
    0.7 m -- off the closed form at a 20 km step, and the residual falls by four
    at each halving of the step (3.4e-5, 8.4e-6, 2.1e-6, 5.2e-7 degrees at 40,
    20, 10 and 5 km), which is the step's mid-latitude cosine truncating a rhumb
    increment and not a drift.
    The single-tangent-frame construction cannot make this check: its ``xi_1``
    turns as the line climbs, so there is no closed form to compare against.
    """
    c = np.cos(np.pi / 4)
    R = np.array([[c, -c], [c, c]])
    M = R @ np.diag([1.0, 3.0]) @ R.T
    fm = _local_frame_flowmap(
        np.linspace(-20.0, 20.0, 41), np.linspace(20.0, 60.0, 41), M
    )

    # The construction did what it claims: C is M^T M everywhere.
    uniform = xr.DataArray(
        M.T @ M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
    )
    assert float(abs(fm.cauchy_green() - uniform).max()) < 1e-6

    seed_lon, seed_lat = 0.0, 40.0
    lines = shrink_lines(
        fm,
        seed_lon=[seed_lon],
        seed_lat=[seed_lat],
        step_m=20_000.0,
        line_length_m=3_000_000.0,
    )
    lon = lines["lon"].isel(line=0).values
    lat = lines["lat"].isel(line=0).values
    valid = np.isfinite(lon) & np.isfinite(lat)
    assert valid.all()  # the whole line stays on this wide grid
    assert np.ptp(lat) > 15.0  # and climbs far enough for cos(lat) to matter

    def inverse_gudermannian(lat_deg):
        return np.log(np.tan(0.25 * np.pi + 0.5 * np.deg2rad(lat_deg)))

    # tan(45 degrees) = 1, so the bearing factor drops out.
    expected_lon = seed_lon + np.rad2deg(
        inverse_gudermannian(lat) - inverse_gudermannian(seed_lat)
    )
    # 8.4e-6 degrees measured at these parameters, second order in the step and
    # roughly linear in the line length (5.0e-6 at 2000 km, 1.3e-5 at 4000 km).
    # 1e-4 therefore survives a doubled step or a much longer line, while a
    # stepper that took the cosine at the segment's start rather than its
    # mid-latitude -- first order -- lands at 1.5e-2 degrees here, 150 times over.
    assert np.max(np.abs(lon - expected_lon)) < 1e-4


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
LCS_END_TIME = RELEASE_TIME + np.timedelta64(7, "D")

LCS_KWARGS = {
    "window_m": 150_000.0,
    # Deliberately not ftle_ridge_seeds' own default of 0.90.
    "quantile": 0.5,
    "step_m": 10_000.0,
    "line_length_m": 30_000.0,
}

#: ``window_m`` for the tests below that run on the ``lon_axis``/``lat_axis``
#: fixture grid instead of the wavy one. Its cells are 1 degree by 1-2 degrees,
#: so ``LCS_KWARGS["window_m"]`` would be a single cell there and the call would
#: (rightly) warn; 700 km clears three cells in every region.
FIXTURE_WINDOW_M = 700_000.0


def _wavy_stretch_flowmap(period_m=250_000.0, base=3.0, amp=1.0):
    """A flow map whose meridional stretching oscillates with zonal position.

    ``f(dx, dy) = (dx, dy * (base + amp cos(2 pi dx / period_m)))``, so the
    Cauchy-Green tensor -- and hence the FTLE -- varies along ``i`` with several
    maxima across the domain. That makes the windowed local-maximum test and the
    quantile floor both load-bearing, unlike a constant-``C`` fixture.
    """

    def f(dx, dy):
        return dx, dy * (base + amp * np.cos(2.0 * np.pi * dx / period_m))

    return advected_flowmap_f(
        AuxiliarySeedGrid, LCS_LON, LCS_LAT, f, RELEASE_TIME, LCS_END_TIME
    )


def test_hyperbolic_lcs_matches_the_manual_pipeline():
    """hyperbolic_lcs() is exactly ftle -> ftle_ridge_seeds -> shrink_lines,
    plus the FTLE field itself."""
    fm = _wavy_stretch_flowmap()

    ftle = fm.ftle()
    seeds = ftle_ridge_seeds(
        ftle, window_m=LCS_KWARGS["window_m"], quantile=LCS_KWARGS["quantile"]
    )
    manual = shrink_lines(
        fm,
        seed_lon=seeds["lon"],
        seed_lat=seeds["lat"],
        step_m=LCS_KWARGS["step_m"],
        line_length_m=LCS_KWARGS["line_length_m"],
    )

    lcs = fm.hyperbolic_lcs(**LCS_KWARGS)

    assert set(lcs.data_vars) == {"lon", "lat", "ftle"}
    # a real, partial seed selection
    assert lcs.sizes["line"] == seeds.sizes["seed"] > 1
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

    loose = fm.hyperbolic_lcs(window_m=150_000.0, quantile=0.5, **trace)
    tight_quantile = fm.hyperbolic_lcs(window_m=150_000.0, quantile=0.99, **trace)
    tight_window = fm.hyperbolic_lcs(window_m=400_000.0, quantile=0.5, **trace)

    assert tight_quantile.sizes["line"] < loose.sizes["line"]
    assert tight_window.sizes["line"] < loose.sizes["line"]


def test_hyperbolic_lcs_forwards_ftle_min():
    """``ftle_min`` reaches ``ftle_ridge_seeds`` too: a floor at the field's 0.99
    quantile leaves strictly fewer lines than one at its median, and asking for
    both selectors raises rather than silently dropping one."""
    fm = _wavy_stretch_flowmap()
    ftle = fm.ftle()
    trace = {
        "window_m": 150_000.0,
        "step_m": LCS_KWARGS["step_m"],
        "line_length_m": LCS_KWARGS["line_length_m"],
    }

    loose = fm.hyperbolic_lcs(ftle_min=float(ftle.quantile(0.5)), **trace)
    tight = fm.hyperbolic_lcs(ftle_min=float(ftle.quantile(0.99)), **trace)

    assert tight.sizes["line"] < loose.sizes["line"]
    assert loose.attrs["selector"] == "ftle_min"
    with pytest.raises(ValueError):
        fm.hyperbolic_lcs(quantile=0.5, ftle_min=0.0, **trace)


def test_hyperbolic_lcs_min_anisotropy_is_forwarded(lon_axis, lat_axis):
    """``min_anisotropy`` reaches ``shrink_lines``, so the guard cannot go inert
    on the convenience path while the two ridge parameters stay wired up.

    ``M = diag(3, 1)`` gives an eigenvalue ratio of 9 everywhere: a floor of 1.15
    lets the lines through, and an absurd floor kills every one of them. If the
    forward is dropped both calls fall back to the same default and trace alike.
    """
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([3.0, 1.0]),
        RELEASE_TIME,
        END_TIME,
    )
    trace = {
        "window_m": FIXTURE_WINDOW_M,
        "step_m": LCS_KWARGS["step_m"],
        "line_length_m": LCS_KWARGS["line_length_m"],
    }

    traced = fm.hyperbolic_lcs(min_anisotropy=1.15, **trace)
    guarded = fm.hyperbolic_lcs(min_anisotropy=1e9, **trace)

    assert bool(traced["lon"].notnull().any())
    assert bool(guarded["lon"].isnull().all())


def test_hyperbolic_lcs_computes_the_ftle_once(lon_axis, lat_axis, monkeypatch):
    """The FTLE is computed a single time and handed to the ridge finder."""
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([1.0, 3.0]),
        RELEASE_TIME,
        END_TIME,
    )
    calls = []
    original = type(fm).ftle

    def counting_ftle(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(type(fm), "ftle", counting_ftle)

    fm.hyperbolic_lcs(**{**LCS_KWARGS, "window_m": FIXTURE_WINDOW_M})

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("t0", "t1", "kind"),
    [(RELEASE_TIME, END_TIME, "repelling"), (END_TIME, RELEASE_TIME, "attracting")],
)
def test_hyperbolic_lcs_metadata_names_the_lcs_type(lon_axis, lat_axis, t0, t1, kind):
    """A forward flow map yields repelling LCS, a backward one attracting ones,
    and the returned dataset says which without being asked."""
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, np.diag([1.0, 3.0]), t0, t1
    )

    lcs = fm.hyperbolic_lcs(**{**LCS_KWARGS, "window_m": FIXTURE_WINDOW_M})

    assert kind in lcs.attrs["long_name"]
    assert kind in lcs["lon"].attrs["long_name"]
    assert kind in lcs["lat"].attrs["long_name"]
    assert lcs["ftle"].attrs["units"] == "1/s"


def test_shrink_lines_seed_pair_is_keyword_only(lon_axis, lat_axis):
    """Passing the seed lon/lat pair positionally raises, so a swap cannot pass
    silently."""
    fm = advected_flowmap(
        AuxiliarySeedGrid,
        lon_axis,
        lat_axis,
        np.diag([1.0, 3.0]),
        RELEASE_TIME,
        END_TIME,
    )
    seed = _centre_seed(fm)
    with pytest.raises(TypeError):
        shrink_lines(fm, seed["seed_lon"], seed["seed_lat"])
