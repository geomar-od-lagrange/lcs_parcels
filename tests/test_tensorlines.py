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
from conftest import advected_flowmap

from lcs_parcels import AuxiliarySeed, ftle_ridge_seeds, shrink_lines
from lcs_parcels.grids import _lonlat_to_meters
from lcs_parcels.tensorlines import _window_cells

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


def test_shrink_lines_stop_below_ftle_guard(lon_axis, lat_axis):
    """M = I gives zero FTLE, below the guard, so the (untraceable) line is all NaN."""
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, np.eye(2), T0, T1)
    lines = shrink_lines(
        fm, **_centre_seed(fm), ftle_min_per_day=0.005, line_length_m=30_000.0
    )

    assert bool(lines["lon"].isnull().all())


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


def test_shrink_lines_seed_pair_is_keyword_only(lon_axis, lat_axis):
    """Passing the seed lon/lat pair positionally raises, so a swap cannot pass
    silently -- including the ``*ftle_ridge_seeds(...)`` unpacking form."""
    fm = advected_flowmap(
        AuxiliarySeed, lon_axis, lat_axis, np.diag([1.0, 3.0]), T0, T1
    )
    seed = _centre_seed(fm)
    with pytest.raises(TypeError):
        shrink_lines(fm, seed["seed_lon"], seed["seed_lat"])
