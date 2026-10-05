"""Elliptic LCS: closed shear lines of the Cauchy-Green tensor.

The analytic flow maps are built in a local metres frame about the seed
centroid (:func:`conftest.advected_flowmap_f`), with no Parcels.

An axisymmetric vortex turns each circle rigidly, so every circle is a closed
``eta_lambda`` orbit at ``lambda = 1``.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import EARTH_RADIUS_M, advected_flowmap, advected_flowmap_f
from scipy.interpolate import RegularGridInterpolator

from lcs_parcels import (
    NeighborSeedGrid,
    closed_shear_lines,
    elliptic_centres,
    outermost_shear_lines,
    stretch_range,
)
from lcs_parcels.elliptic import _eta_tangent

T0 = np.datetime64("2020-01-01")
WINDOW = np.timedelta64(10, "D")
WINDOW_S = 10 * 86_400.0

# A Gaussian vortex turning its core by about 2 rad, resolved to about 0.1 rad
# per cell so the central-difference grad F stays accurate.
OMEGA_0 = 2.3e-6
VORTEX_R_M = 40_000.0
VORTEX_AXIS = np.arange(-1.2, 1.2 + 1e-9, 0.02)
MAX_RADIUS_M = 60_000.0
M_PER_DEG = EARTH_RADIUS_M * np.pi / 180.0


def _angular_velocity(r2, omega_0):
    return omega_0 * np.exp(-r2 / VORTEX_R_M**2)


def _vortex_map(omega_0):
    """Exact flow map of the steady vortex: rotate each circle by Omega(r) T."""

    def f(dx, dy):
        angle = _angular_velocity(dx**2 + dy**2, omega_0) * WINDOW_S
        c, s = np.cos(angle), np.sin(angle)
        return c * dx - s * dy, s * dx + c * dy

    return f


def _vortex_in_strain_map(omega_0, strain):
    """RK4 flow map of the vortex in the uniform strain u = (s x, -s y)."""

    def velocity(x, y):
        omega = _angular_velocity(x**2 + y**2, omega_0)
        return strain * x - omega * y, -strain * y + omega * x

    def f(dx, dy):
        x, y = np.asarray(dx, dtype=float), np.asarray(dy, dtype=float)
        dt = 3_600.0
        for _ in range(round(WINDOW_S / dt)):
            k1 = velocity(x, y)
            k2 = velocity(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1])
            k3 = velocity(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1])
            k4 = velocity(x + dt * k3[0], y + dt * k3[1])
            x = x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            y = y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return x, y

    return f


def _vortex_flowmap(omega_0=OMEGA_0, strain=0.0):
    f = (
        _vortex_map(omega_0)
        if strain == 0.0
        else _vortex_in_strain_map(omega_0, strain)
    )
    return advected_flowmap_f(
        NeighborSeedGrid, VORTEX_AXIS, VORTEX_AXIS, f, T0, T0 + WINDOW
    )


@pytest.fixture(scope="module")
def vortex():
    return _vortex_flowmap()


@pytest.fixture(scope="module")
def vortex_orbits(vortex):
    return closed_shear_lines(
        vortex, centre_lon=[0.0], centre_lat=[0.0], max_radius_m=MAX_RADIUS_M
    )


def _distance_from_origin_m(lon, lat):
    return np.hypot(lon * M_PER_DEG, lat * M_PER_DEG)


# --- the eta_lambda tangent --------------------------------------------------

TENSOR_AXIS = np.linspace(-1.0, 1.0, 21)
# lambda_1 = 1/4 along east, lambda_2 = 4 along north
C_DIAG = np.diag([0.25, 4.0])


def _uniform_tensor_interp(C, nan_at=None):
    field = np.broadcast_to(
        np.asarray(C, dtype=float), (TENSOR_AXIS.size, TENSOR_AXIS.size, 2, 2)
    ).copy()
    if nan_at is not None:
        field[nan_at[0], nan_at[1]] = np.nan
    return RegularGridInterpolator(
        (TENSOR_AXIS, TENSOR_AXIS), field, bounds_error=False, fill_value=np.nan
    )


def _eta(stretch, branch, *, heading=(1.0, 1.0), C=C_DIAG, nan_at=None, at=0.0):
    return _eta_tangent(
        np.array([at]),
        np.array([0.0]),
        np.array([heading]),
        stretch=np.array([stretch]),
        branch=np.array([branch]),
        tensor_interp=_uniform_tensor_interp(C, nan_at=nan_at),
    )[0]


@pytest.mark.parametrize("branch", [+1, -1])
@pytest.mark.parametrize("stretch", [0.6, 1.0, 1.9])
def test_eta_tangent_is_a_unit_vector(stretch, branch):
    assert np.linalg.norm(_eta(stretch, branch)) == pytest.approx(1.0)


def test_eta_tangent_matches_haller_beron_vera_eq_14():
    """eta^pm = sqrt((l2 - lam^2) / (l2 - l1)) xi_1 pm sqrt((lam^2 - l1) / (l2 -
    l1)) xi_2, with xi_1 east and xi_2 north for C = diag(1/4, 4)."""
    a, b = np.sqrt((4.0 - 1.0) / 3.75), np.sqrt((1.0 - 0.25) / 3.75)
    np.testing.assert_allclose(_eta(1.0, +1), [a, b])
    np.testing.assert_allclose(_eta(1.0, -1), [a, -b])


def test_eta_branches_are_mirror_images_about_xi_1():
    """The two branches share the xi_1 component and oppose in the xi_2 one,
    whatever sign the eigensolver gives xi_1."""
    rotated = np.array([[0.0, -1.0], [1.0, 0.0]])
    C = rotated @ C_DIAG @ rotated.T  # xi_1 north, xi_2 west
    plus = _eta(1.0, +1, C=C, heading=(-1.0, 1.0))
    minus = _eta(1.0, -1, C=C, heading=(1.0, 1.0))
    assert plus[1] == pytest.approx(minus[1])
    assert plus[0] == pytest.approx(-minus[0])


def test_eta_tangent_follows_the_heading():
    assert _eta(1.0, +1, heading=(-1.0, -1.0)) @ np.array([-1.0, -1.0]) > 0


@pytest.mark.parametrize("stretch", [0.4, 0.5, 2.0, 2.1])
def test_eta_tangent_is_nan_unless_lambda_squared_is_strictly_inside(stretch):
    """eta_lambda exists only where lambda_1 < lambda^2 < lambda_2."""
    assert np.isnan(_eta(stretch, +1)).all()


def test_eta_tangent_reduces_to_xi_1_as_lambda_squared_meets_lambda_1():
    np.testing.assert_allclose(_eta(0.5 + 1e-6, +1), [1.0, 0.0], atol=1e-2)


def test_eta_tangent_is_nan_off_grid_and_in_a_nan_cell():
    assert np.isnan(_eta(1.0, +1, at=5.0)).all()
    assert np.isnan(_eta(1.0, +1, nan_at=(10, 10))).all()


# --- stretch_range ------------------------------------------------------------


def test_stretch_range_is_log_symmetric_about_one_and_includes_it():
    stretches = stretch_range(stretch_max=1.5, step=0.03)
    assert 1.0 in stretches
    np.testing.assert_allclose(np.log(stretches), -np.log(stretches[::-1]))
    np.testing.assert_allclose(np.diff(np.log(stretches)), 0.03)
    assert stretches.max() <= 1.5
    assert stretches.max() * np.exp(0.03) > 1.5


@pytest.mark.parametrize(
    "kwargs", [{"stretch_max": 1.0}, {"stretch_max": 0.9}, {"step": 0.0}]
)
def test_stretch_range_rejects_an_empty_or_degenerate_range(kwargs):
    with pytest.raises(ValueError):
        stretch_range(**kwargs)


# --- polar rotation -------------------------------------------------------------

EQUATOR_AXIS = np.linspace(-1.0, 1.0, 9)


def _rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


@pytest.mark.parametrize("angle", [0.7, -0.7])
def test_polar_rotation_of_a_rigid_rotation_is_its_angle(angle):
    fm = advected_flowmap(
        NeighborSeedGrid, EQUATOR_AXIS, EQUATOR_AXIS, _rotation(angle), T0, T0 + WINDOW
    )
    theta = fm.polar_rotation().isel(i=slice(1, -1), j=slice(1, -1))
    np.testing.assert_allclose(theta.values, angle, atol=1e-3)


def test_polar_rotation_wraps_past_half_a_turn():
    """theta is the angle of R, defined modulo 2 pi, so a rotation by 4 rad
    reads as 4 - 2 pi: clockwise."""
    fm = advected_flowmap(
        NeighborSeedGrid, EQUATOR_AXIS, EQUATOR_AXIS, _rotation(4.0), T0, T0 + WINDOW
    )
    theta = fm.polar_rotation().isel(i=slice(1, -1), j=slice(1, -1))
    np.testing.assert_allclose(theta.values, 4.0 - 2 * np.pi, atol=1e-3)


def test_polar_rotation_of_pure_strain_is_zero():
    fm = advected_flowmap(
        NeighborSeedGrid,
        EQUATOR_AXIS,
        EQUATOR_AXIS,
        np.diag([2.0, 0.5]),
        T0,
        T0 + WINDOW,
    )
    theta = fm.polar_rotation().isel(i=slice(1, -1), j=slice(1, -1))
    np.testing.assert_allclose(theta.values, 0.0, atol=1e-3)


def test_polar_rotation_is_labelled_and_on_the_grid():
    fm = advected_flowmap(
        NeighborSeedGrid, EQUATOR_AXIS, EQUATOR_AXIS, _rotation(0.3), T0, T0 + WINDOW
    )
    theta = fm.polar_rotation()
    assert theta.name == "polar_rotation"
    assert theta.attrs["units"] == "rad"
    assert theta.attrs["long_name"]
    assert set(theta.dims) == {"i", "j"}
    assert {"lon_grid", "lat_grid"} <= set(theta.coords)


# --- elliptic_centres -----------------------------------------------------------

CENTRE_AXIS = np.arange(0.0, 4.0 + 1e-9, 0.05)


def _field(values):
    """``values`` on the centre grid, tilted by 1e-3 per degree so no stretch of
    it is a plateau, every cell of which would tie as a windowed extremum."""
    lon_grid, lat_grid = np.meshgrid(CENTRE_AXIS, CENTRE_AXIS, indexing="ij")
    return xr.DataArray(
        values + 1e-3 * (lon_grid + lat_grid),
        dims=("i", "j"),
        coords={
            "lon_grid": (("i", "j"), lon_grid),
            "lat_grid": (("i", "j"), lat_grid),
        },
        name="indicator",
        attrs={"long_name": "test indicator", "units": "1"},
    )


def _two_dips():
    lon_grid, lat_grid = np.meshgrid(CENTRE_AXIS, CENTRE_AXIS, indexing="ij")
    dip = 0.0
    for lon_c, lat_c in [(1.0, 1.0), (3.0, 2.5)]:
        dip = dip + np.exp(-((lon_grid - lon_c) ** 2 + (lat_grid - lat_c) ** 2) / 0.1)
    return _field(2.0 - dip)


def test_elliptic_centres_finds_windowed_minima():
    centres = elliptic_centres(_two_dips(), extremum="min", window_m=100_000.0)
    found = sorted(zip(centres["lon"].values, centres["lat"].values, strict=True))
    np.testing.assert_allclose(found, [(1.0, 1.0), (3.0, 2.5)], atol=0.051)


def test_elliptic_centres_finds_windowed_maxima_of_the_negated_field():
    minima = elliptic_centres(_two_dips(), extremum="min", window_m=100_000.0)
    maxima = elliptic_centres(-_two_dips(), extremum="max", window_m=100_000.0)
    np.testing.assert_array_equal(minima["lon"].values, maxima["lon"].values)
    np.testing.assert_array_equal(minima["lat"].values, maxima["lat"].values)


def test_elliptic_centres_needs_the_extremum_stated():
    with pytest.raises(TypeError):
        elliptic_centres(_two_dips())
    with pytest.raises(ValueError):
        elliptic_centres(_two_dips(), extremum="lowest")


def test_elliptic_centres_keeps_clear_of_the_edge():
    """A dip 0.1 degree (11 km) from the edge sits inside the 50 km margin."""
    lon_grid, lat_grid = np.meshgrid(CENTRE_AXIS, CENTRE_AXIS, indexing="ij")
    field = _field(2.0 - np.exp(-((lon_grid - 0.1) ** 2 + (lat_grid - 2.0) ** 2) / 0.1))
    centres = elliptic_centres(field, extremum="min", window_m=100_000.0)
    assert centres.sizes["centre"] == 0


def test_elliptic_centres_keeps_clear_of_nan_cells():
    values = _two_dips().values.copy()
    values[20, 22] = np.nan  # 0.1 degree from the dip at (1, 1)
    centres = elliptic_centres(_field(values), extremum="min", window_m=100_000.0)
    found = sorted(zip(centres["lon"].values, centres["lat"].values, strict=True))
    np.testing.assert_allclose(found, [(3.0, 2.5)], atol=0.051)


def test_elliptic_centres_output_structure():
    centres = elliptic_centres(_two_dips(), extremum="min", window_m=100_000.0)
    assert set(centres.data_vars) >= {"lon", "lat"}
    assert centres["lon"].dims == ("centre",)
    assert centres["lon"].attrs["units"] == "degrees_east"
    assert centres["lat"].attrs["units"] == "degrees_north"
    assert centres["centre"].attrs["long_name"]
    for key in ("extremum", "window_m", "edge_m", "window_cells_i", "window_cells_j"):
        assert key in centres.attrs


# --- closed_shear_lines on the vortex ---------------------------------------------


def test_closed_shear_lines_finds_orbits_in_the_vortex(vortex_orbits):
    assert vortex_orbits.sizes["orbit"] > 0


def test_every_vortex_orbit_is_a_circle_about_the_centre(vortex_orbits):
    """Orbits wider than the two-cell span of the central-difference stencil are
    circles. A loop one cell across is below the resolution of grad F."""
    stencil_span_m = 2 * float(vortex_orbits.attrs["launch_spacing_m"])
    resolved = vortex_orbits["radius_m"] > stencil_span_m
    for orbit in np.flatnonzero(resolved.values):
        line = vortex_orbits.isel(orbit=orbit)
        r = _distance_from_origin_m(line["lon"].values, line["lat"].values)
        r = r[np.isfinite(r)]
        assert r.std() / r.mean() < 0.02
        assert float(line["radius_m"]) == pytest.approx(r.mean(), rel=0.02)


def test_every_vortex_orbit_closes_at_lambda_one(vortex_orbits):
    """The circles do not stretch their tangent elements, so they close at
    lambda = 1, within one step of the default scan."""
    np.testing.assert_array_less(np.abs(np.log(vortex_orbits["stretch"])), 0.03 + 1e-9)


def test_vortex_orbits_span_the_sections(vortex_orbits):
    """Circles are orbits at every radius the sections reach, out to where the
    search stops."""
    radii = vortex_orbits["radius_m"].values
    assert radii.min() < 0.3 * MAX_RADIUS_M
    assert radii.max() > 0.8 * MAX_RADIUS_M


def test_vortex_orbits_are_closed_curves(vortex_orbits):
    for orbit in range(vortex_orbits.sizes["orbit"]):
        line = vortex_orbits.isel(orbit=orbit)
        lon = line["lon"].values[np.isfinite(line["lon"].values)]
        lat = line["lat"].values[np.isfinite(line["lat"].values)]
        assert (lon[0], lat[0]) == (lon[-1], lat[-1])
        assert float(line["residual_m"]) < float(vortex_orbits.attrs["closure_tol_m"])


def test_closed_shear_lines_output_structure(vortex_orbits):
    for name in ("lon", "lat"):
        assert vortex_orbits[name].dims == ("orbit", "point")
    for name in (
        "centre",
        "centre_lon",
        "centre_lat",
        "stretch",
        "branch",
        "area_m2",
        "radius_m",
        "residual_m",
    ):
        assert vortex_orbits[name].dims == ("orbit",)
    assert set(np.unique(vortex_orbits["branch"])) <= {-1, 1}
    for key in (
        "stretch_min",
        "stretch_max",
        "n_stretch",
        "max_radius_m",
        "launch_spacing_m",
        "step_m",
        "closure_tol_m",
        "n_never_returned",
    ):
        assert key in vortex_orbits.attrs


def test_closed_shear_lines_takes_the_centre_pair_by_keyword(vortex):
    with pytest.raises(TypeError):
        closed_shear_lines(vortex, [0.0], [0.0])


def test_closed_shear_lines_with_no_centres_returns_no_orbits(vortex):
    orbits = closed_shear_lines(vortex, centre_lon=[], centre_lat=[])
    assert orbits.sizes["orbit"] == 0
    assert "radius_m" in orbits


def test_closed_shear_lines_scans_the_given_stretches_only(vortex):
    orbits = closed_shear_lines(
        vortex,
        centre_lon=[0.0],
        centre_lat=[0.0],
        stretches=np.array([1.0]),
        max_radius_m=MAX_RADIUS_M,
    )
    assert orbits.sizes["orbit"] > 0
    np.testing.assert_array_equal(orbits["stretch"].values, 1.0)
    assert orbits.attrs["n_stretch"] == 1


def test_closed_shear_lines_bisects_a_sign_change_of_the_return_map():
    """In the strained vortex at a 5 m tolerance, the orbits come from bisecting
    sign changes of P(s) - s, and they still close within 5 m."""
    fm = _vortex_flowmap(strain=STRAINS[0])
    orbits = closed_shear_lines(
        fm,
        centre_lon=[0.0],
        centre_lat=[0.0],
        stretches=np.array([1.0]),
        max_radius_m=MAX_RADIUS_M,
        closure_tol_m=5.0,
    )
    launch_spacing_m = orbits.attrs["launch_spacing_m"]
    centre_distance_m = np.abs(orbits["lon"].isel(point=0)) * M_PER_DEG
    off_launch = np.abs(
        centre_distance_m / launch_spacing_m
        - np.round(centre_distance_m / launch_spacing_m)
    )
    assert (off_launch > 1e-3).any()
    np.testing.assert_array_less(orbits["residual_m"], 5.0)


def test_closed_shear_lines_off_the_grid_finds_nothing(vortex):
    orbits = closed_shear_lines(vortex, centre_lon=[5.0], centre_lat=[0.0])
    assert orbits.sizes["orbit"] == 0
    assert orbits.attrs["n_never_returned"] == 2 * 2 * orbits.attrs["n_stretch"]


def test_closed_shear_lines_needs_matching_centre_arrays(vortex):
    with pytest.raises(ValueError):
        closed_shear_lines(vortex, centre_lon=[0.0, 0.1], centre_lat=[0.0])


def test_a_line_out_of_steps_does_not_return(vortex):
    """Two steps cannot take a line round its centre."""
    from lcs_parcels.elliptic import _section_launch, _trace_to_return
    from lcs_parcels.tensorlines import _tensor_interp

    one = np.ones(1)
    lon_0, lat_0 = _section_launch(
        centre_lon=0 * one, centre_lat=0 * one, direction=one, s=20_000.0 * one
    )
    returned_s, _ = _trace_to_return(
        lon_0,
        lat_0,
        centre_lon=0 * one,
        centre_lat=0 * one,
        direction=one,
        stretch=one,
        branch=one,
        section_m=MAX_RADIUS_M,
        tensor_interp=_tensor_interp(vortex),
        step_m=1_000.0,
        n_steps=2,
    )
    assert np.isnan(returned_s).all()


# --- selection ------------------------------------------------------------------


def test_outermost_shear_lines_keeps_the_largest_orbit(vortex, vortex_orbits):
    eddies = outermost_shear_lines(vortex_orbits)
    assert eddies.sizes["eddy"] == 1
    assert float(eddies["area_m2"].max()) == float(vortex_orbits["area_m2"].max())
    assert eddies["lon"].dims == ("eddy", "point")


@pytest.mark.parametrize(("omega_0", "sense"), [(OMEGA_0, 1), (-OMEGA_0, -1)])
def test_rotation_sense_follows_the_vortex(omega_0, sense):
    fm = _vortex_flowmap(omega_0=omega_0)
    orbits = closed_shear_lines(
        fm, centre_lon=[0.0], centre_lat=[0.0], max_radius_m=MAX_RADIUS_M
    )
    eddies = outermost_shear_lines(orbits, rotation=fm.polar_rotation())
    assert eddies["rotation_sense"].values.tolist() == [sense]


def test_rotation_sense_is_zero_where_the_field_misses_the_boundary(
    vortex, vortex_orbits
):
    """A rotation field with no point inside a boundary gives it no sense."""
    corner = vortex.polar_rotation().isel(i=slice(0, 10), j=slice(0, 10))
    eddies = outermost_shear_lines(vortex_orbits, rotation=corner)
    assert eddies["rotation_sense"].values.tolist() == [0]


def test_rotation_sense_needs_a_rotation_field(vortex_orbits):
    """Without a rotation field the boundaries carry no rotation sense."""
    assert "rotation_sense" not in outermost_shear_lines(vortex_orbits)


def test_outermost_shear_lines_merges_centres_inside_one_boundary(vortex):
    """Two candidate centres 5.5 km apart in one vortex give one eddy."""
    orbits = closed_shear_lines(
        vortex,
        centre_lon=[0.0, 0.05],
        centre_lat=[0.0, 0.0],
        max_radius_m=MAX_RADIUS_M,
    )
    eddies = outermost_shear_lines(orbits)
    assert eddies.sizes["eddy"] == 1


def test_eddy_centroid_is_the_vortex_centre(vortex, vortex_orbits):
    eddies = outermost_shear_lines(vortex_orbits)
    distance = _distance_from_origin_m(
        float(eddies["centroid_lon"][0]), float(eddies["centroid_lat"][0])
    )
    assert distance < 2_000.0


def test_outermost_shear_lines_of_no_orbits_is_empty(vortex):
    orbits = closed_shear_lines(vortex, centre_lon=[], centre_lat=[])
    eddies = outermost_shear_lines(orbits, rotation=vortex.polar_rotation())
    assert eddies.sizes["eddy"] == 0


# --- the vortex in strain --------------------------------------------------------


STRAINS = (2.3e-7, 4.6e-7)


def _saddle_radius_m(strain):
    """Radius of the saddles of the steady flow, where Omega(r) equals the strain.

    The closed streamlines lie inside the separatrix through them, within this
    radius of the centre."""
    return VORTEX_R_M * np.sqrt(np.log(OMEGA_0 / strain))


@pytest.fixture(scope="module")
def strained_boundaries():
    radii = {}
    for strain in STRAINS:
        fm = _vortex_flowmap(strain=strain)
        orbits = closed_shear_lines(
            fm, centre_lon=[0.0], centre_lat=[0.0], max_radius_m=MAX_RADIUS_M
        )
        eddies = outermost_shear_lines(orbits)
        radii[strain] = float(eddies["radius_m"].max())
    return radii


@pytest.mark.parametrize("strain", STRAINS)
def test_the_boundary_lies_inside_the_separatrix(strained_boundaries, strain):
    assert 0.0 < strained_boundaries[strain] < _saddle_radius_m(strain)


def test_stronger_strain_moves_the_boundary_inward(strained_boundaries):
    assert strained_boundaries[STRAINS[1]] < strained_boundaries[STRAINS[0]]


# --- dilation removed -------------------------------------------------------
#
# A conformal map w = z + eps z^2 has grad W = s R, an isotropic dilation s that
# varies over the domain. After the vortex it multiplies C by s^2, which
# C / |det grad F| removes again.

CONFORMAL_EPS = 2.5e-6  # 1/m, so s changes by about 20 percent across 40 km


def _dilated_vortex_map(dx, dy):
    x, y = _vortex_map(OMEGA_0)(dx, dy)
    w = (x + 1j * y) + CONFORMAL_EPS * (x + 1j * y) ** 2
    return w.real, w.imag


@pytest.fixture(scope="module")
def dilated_vortex():
    return advected_flowmap_f(
        NeighborSeedGrid, VORTEX_AXIS, VORTEX_AXIS, _dilated_vortex_map, T0, T0 + WINDOW
    )


@pytest.fixture(scope="module")
def dilated_orbits(dilated_vortex):
    return {
        remove: closed_shear_lines(
            dilated_vortex,
            centre_lon=[0.0],
            centre_lat=[0.0],
            max_radius_m=MAX_RADIUS_M,
            remove_dilation=remove,
        )
        for remove in (False, True)
    }


def test_removing_the_dilation_recovers_the_circles(dilated_orbits):
    """With the dilation divided out, the vortex circles close again, at
    lambda = 1 within one step, with a radius scatter under 2 percent."""
    orbits = dilated_orbits[True]
    stencil_span_m = 2 * float(orbits.attrs["launch_spacing_m"])
    resolved = np.flatnonzero((orbits["radius_m"] > stencil_span_m).values)
    assert resolved.size > 0
    for orbit in resolved:
        line = orbits.isel(orbit=orbit)
        r = _distance_from_origin_m(line["lon"].values, line["lat"].values)
        r = r[np.isfinite(r)]
        assert r.std() / r.mean() < 0.02
    np.testing.assert_array_less(np.abs(np.log(orbits["stretch"])), 0.03 + 1e-9)


def test_a_varying_dilation_breaks_the_plain_search(dilated_orbits):
    assert dilated_orbits[False].sizes["orbit"] < dilated_orbits[True].sizes["orbit"]


def test_the_search_records_whether_it_removed_the_dilation(dilated_orbits):
    assert dilated_orbits[False].attrs["remove_dilation"] == 0
    assert dilated_orbits[True].attrs["remove_dilation"] == 1


def test_removing_the_dilation_leaves_an_area_preserving_map_alone(vortex):
    """The pure vortex has det grad F = 1, so both searches agree."""
    plain = closed_shear_lines(
        vortex, centre_lon=[0.0], centre_lat=[0.0], max_radius_m=MAX_RADIUS_M
    )
    removed = closed_shear_lines(
        vortex,
        centre_lon=[0.0],
        centre_lat=[0.0],
        max_radius_m=MAX_RADIUS_M,
        remove_dilation=True,
    )
    assert removed.sizes["orbit"] == plain.sizes["orbit"]
    np.testing.assert_allclose(removed["radius_m"], plain["radius_m"], rtol=1e-3)


# --- one call -------------------------------------------------------------------


def test_elliptic_lcs_passes_remove_dilation_on(dilated_vortex):
    eddies = dilated_vortex.elliptic_lcs(
        window_m=60_000.0, max_radius_m=MAX_RADIUS_M, remove_dilation=True
    )
    assert eddies.attrs["remove_dilation"] == 1


def test_elliptic_lcs_finds_the_vortex(vortex):
    eddies = vortex.elliptic_lcs(window_m=60_000.0, max_radius_m=MAX_RADIUS_M)
    assert eddies.sizes["eddy"] == 1
    distance = _distance_from_origin_m(
        float(eddies["centroid_lon"][0]), float(eddies["centroid_lat"][0])
    )
    assert distance < 2_000.0
    assert "rotation_sense" not in eddies


# --- metadata -------------------------------------------------------------------

INDEX_COORDS = {"orbit", "point", "eddy", "centre"}


def _assert_labelled(ds):
    for name, array in ds.data_vars.items():
        assert array.attrs.get("long_name"), f"{name} has no long_name"
        assert array.attrs.get("units"), f"{name} has no units"
    for name, coord in ds.coords.items():
        assert coord.attrs.get("long_name"), f"{name} has no long_name"
        if name not in INDEX_COORDS:
            assert coord.attrs.get("units"), f"{name} has no units"


def test_elliptic_outputs_are_labelled(vortex, vortex_orbits):
    _assert_labelled(vortex_orbits)
    eddies = outermost_shear_lines(vortex_orbits, rotation=vortex.polar_rotation())
    _assert_labelled(eddies)
    assert eddies["rotation_sense"].attrs["units"] == "1"
    assert eddies["area_m2"].attrs["units"] == "m2"
    assert eddies["radius_m"].attrs["units"] == "m"
    assert eddies["stretch"].attrs["units"] == "1"
    # the stretching factor is not the Cauchy-Green eigenvalue
    assert "lambda" not in eddies
