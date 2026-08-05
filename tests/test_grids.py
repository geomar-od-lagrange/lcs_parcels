"""Shape / dim / coord contracts for the SeedGrid and FlowMap families.

A ``SeedGrid`` carries no time: ``from_axes`` takes no ``t0``, and the dataset is
all-coordinates (no ``t0``/``T``, no advected ``lon``/``lat``). Time and the
advected positions enter at ingest, ``seed.pset_to_flowmap(lon=..., lat=...,
t0=..., t1=...)``, which returns a ``FlowMap`` carrying scalar ``t0``/``T``
coords and ``lon``/``lat`` data vars.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import advected_flowmap

from lcs_parcels import AuxiliarySeedGrid, NeighborFlowMap, NeighborSeedGrid
from lcs_parcels.grids import _separation_m

# Release/end times supplied only at ingest (the seed grid carries no time).
RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")

# Generic (sheared) linear flow map, for the diagnostics driven below.
M = np.array([[2.0, 0.5], [0.0, 3.0]])


# --- seed shape ------------------------------------------------------------


def test_neighbor_seed_from_axes_dims(lon_axis, lat_axis):
    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    ds = seed.ds

    # curvilinear reference lon_0/lat_0 and diagnostic grid lon_grid/lat_grid on
    # logical dims (i, j)
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert set(ds["lon_grid"].dims) == {"i", "j"}
    assert set(ds["lat_grid"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # A seed grid carries no time and no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0

    # NeighborSeedGrid carries no displacement stencil dim
    assert "displacement" not in ds.dims


def test_neighbor_seed_lon_lat_values(lon_axis, lat_axis):
    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    ds = seed.ds

    # 2D reference lon_0/lat_0 broadcast from the 1D axes (label-based access).
    lon_iv = lon_axis[2]
    lat_jv = lat_axis[3]
    assert float(ds["lon_0"].isel(i=2, j=3)) == lon_iv
    assert float(ds["lat_0"].isel(i=2, j=3)) == lat_jv
    # lon_0 varies along i and is constant along j; lat_0 the reverse.
    assert float(ds["lon_0"].isel(i=2, j=0)) == float(ds["lon_0"].isel(i=2, j=4))
    assert float(ds["lat_0"].isel(i=0, j=3)) == float(ds["lat_0"].isel(i=3, j=3))


def test_auxiliary_seed_from_axes_dims(lon_axis, lat_axis):
    seed = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    ds = seed.ds

    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # AuxiliarySeedGrid adds a four-arm stencil on the `displacement` dim (no centre,
    # no diagonals).
    assert ds.sizes["displacement"] == 4
    assert list(ds["displacement"].values) == ["east", "north", "west", "south"]

    # The reference positions x_0 are the explicit per-arm positions: lon_0/lat_0
    # carry the displacement dim.
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat_0"].dims) == {"i", "j", "displacement"}

    # The diagnostic grid points are kept on (i, j), without the stencil dim.
    assert set(ds["lon_grid"].dims) == {"i", "j"}
    assert set(ds["lat_grid"].dims) == {"i", "j"}

    # A seed grid carries no time and no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference + grid coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ns = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    aus = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis)

    # NeighborSeedGrid has no auxiliary stencil: no displacement dim, and its
    # reference positions are the grid points themselves on (i, j).
    assert "displacement" not in ns.ds.dims
    assert set(ns.ds["lon_0"].dims) == {"i", "j"}

    # AuxiliarySeedGrid carries the four-arm displacement stencil, so its reference
    # positions are per-arm and distinct from the grid points.
    assert aus.ds.sizes["displacement"] == 4
    assert set(aus.ds["lon_0"].dims) == {"i", "j", "displacement"}


def test_grid_coords_are_canonical_across_stencils(lon_axis, lat_axis):
    """Both stencils carry the diagnostic grid under one name, on (i, j), and both
    expose it as .lon_grid/.lat_grid. For the neighbour stencil it equals the
    reference release positions, stored rather than reconstructed."""
    ns = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    aus = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis)

    for seed in (ns, aus):
        assert set(seed.lon_grid.dims) == {"i", "j"}
        assert set(seed.lat_grid.dims) == {"i", "j"}
        assert seed.lon_grid.identical(seed.ds["lon_grid"])
        assert seed.lat_grid.identical(seed.ds["lat_grid"])

    # Neighbour: the grid point IS the release point, and both are stored.
    assert np.array_equal(ns.lon_grid.values, ns.ds["lon_0"].values)
    assert np.array_equal(ns.lat_grid.values, ns.ds["lat_0"].values)

    # Auxiliary: the arms straddle the grid point, which is their mean.
    assert np.allclose(aus.lon_grid, aus.ds["lon_0"].mean("displacement"))
    assert np.allclose(aus.lat_grid, aus.ds["lat_0"].mean("displacement"))


def test_auxiliary_arm_geometry_and_separation(lon_axis, lat_axis):
    """AuxiliarySeedGrid places its four arms at exactly +/- ``aux_separation_m``.

    Every arm is laid down in the local east/north frame of its own grid point,
    so each arm sits ``s`` meters from that point in its labelled direction and
    zero meters off it in the perpendicular one, at *every* grid point, not on
    average over the grid. Separations are read back with :func:`_separation_m`,
    the same per-pair local frame the diagnostics measure in.
    """
    s = 2_500.0  # non-default aux_separation_m
    seed = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis, aux_separation_m=s)
    lon_0, lat_0 = seed.ds["lon_0"], seed.ds["lat_0"]
    lon_grid, lat_grid = seed.ds["lon_grid"], seed.ds["lat_grid"]

    # Each arm relative to its own grid point: s along the labelled direction, 0
    # across it. Dims (i, j) after selecting one arm.
    expected = {
        "east": (+s, 0.0),
        "west": (-s, 0.0),
        "north": (0.0, +s),
        "south": (0.0, -s),
    }
    for arm, (dx_expected, dy_expected) in expected.items():
        dx, dy = _separation_m(
            lon_a=lon_grid,
            lat_a=lat_grid,
            lon_b=lon_0.sel(displacement=arm, drop=True),
            lat_b=lat_0.sel(displacement=arm, drop=True),
        )
        assert float(abs(dx - dx_expected).max()) < 1e-6
        assert float(abs(dy - dy_expected).max()) < 1e-6

    # Opposing arms span exactly 2s along their axis and share the centre on the
    # other one, again at every grid point.
    ew_dx, ew_dy = _separation_m(
        lon_a=lon_0.sel(displacement="west", drop=True),
        lat_a=lat_0.sel(displacement="west", drop=True),
        lon_b=lon_0.sel(displacement="east", drop=True),
        lat_b=lat_0.sel(displacement="east", drop=True),
    )
    ns_dx, ns_dy = _separation_m(
        lon_a=lon_0.sel(displacement="south", drop=True),
        lat_a=lat_0.sel(displacement="south", drop=True),
        lon_b=lon_0.sel(displacement="north", drop=True),
        lat_b=lat_0.sel(displacement="north", drop=True),
    )
    assert float(abs(ew_dx - 2 * s).max()) < 1e-6
    assert float(abs(ew_dy).max()) < 1e-6
    assert float(abs(ns_dy - 2 * s).max()) < 1e-6
    assert float(abs(ns_dx).max()) < 1e-6


# --- flow-map shape --------------------------------------------------------


def test_neighbor_flowmap_shape(lon_axis, lat_axis):
    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)
    ds = fm.ds

    # The flow map adds the advected positions as its only data vars, on (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j"}
    assert set(ds["lat"].dims) == {"i", "j"}

    # Reference positions are carried through unchanged on (i, j).
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}

    # Scalar t0 (release time) and signed window T = t1 - t0 land as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == RELEASE_TIME
    assert ds["T"] == (END_TIME - RELEASE_TIME)


def test_auxiliary_flowmap_shape(lon_axis, lat_axis):
    seed = AuxiliarySeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)
    ds = fm.ds

    # Advected arm positions carry the displacement dim on top of (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat"].dims) == {"i", "j", "displacement"}

    # Reference arms + diagnostic grid points carried through unchanged.
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lon_grid"].dims) == {"i", "j"}
    assert set(ds["lat_grid"].dims) == {"i", "j"}

    # Scalar t0 / signed T as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == RELEASE_TIME
    assert ds["T"] == (END_TIME - RELEASE_TIME)


def test_grid_image_is_on_the_diagnostic_grid(lon_axis, lat_axis):
    """Both stencils reduce their advected positions onto (i, j); the auxiliary
    one as the centroid of its four arms."""
    for seed_cls in (NeighborSeedGrid, AuxiliarySeedGrid):
        seed = seed_cls.from_axes(lon=lon_axis, lat=lat_axis)
        lon, lat = seed.to_parcels_pset()
        fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)

        image = fm.grid_image
        assert set(image.data_vars) == {"lon", "lat"}
        assert set(image["lon"].dims) == {"i", "j"}
        assert set(image["lat"].dims) == {"i", "j"}
        # Undisplaced particles: the image of a grid point is the grid point.
        assert np.allclose(image["lon"], fm.lon_grid)
        assert np.allclose(image["lat"], fm.lat_grid)


# --- the wrapped dataset is the caller's -----------------------------------


@pytest.mark.parametrize("seed_cls", (NeighborSeedGrid, AuxiliarySeedGrid))
def test_diagnostics_leave_the_flowmap_dataset_untouched(seed_cls, lon_axis, lat_axis):
    """Running the diagnostics writes nothing back into ``.ds``.

    Every operator builds a new object and stamps its metadata onto that, so the
    dataset the caller handed in comes out of the whole chain identical --
    values, names, coords and attrs alike. A field that attached its attributes
    by updating an ``attrs`` dict in place would fail this the moment the object
    it updated turned out to be a view of the input rather than a fresh array.
    """
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)
    before = fm.ds.copy(deep=True)

    fm.deformation_gradient()
    fm.cauchy_green()
    fm.cg_eigen()
    fm.ftle()
    _ = fm.grid_image
    fm.image(lon_0=fm.ds["lon_grid"], lat_0=fm.ds["lat_grid"])
    # window_m is sized to the fixture grid, whose cells run 30-110 km: the
    # default 30 km would be a single cell and the call would rightly warn.
    fm.hyperbolic_lcs(window_m=700_000.0, step_m=1_000.0, line_length_m=6_000.0)
    fm.to_seed()

    xr.testing.assert_identical(fm.ds, before)


# --- reprs -----------------------------------------------------------------


def test_seed_repr_is_a_one_line_summary(lon_axis, lat_axis):
    """Class, grid shape and lon/lat extent on one line, not the dataset."""
    for seed_cls in (NeighborSeedGrid, AuxiliarySeedGrid):
        text = repr(seed_cls.from_axes(lon=lon_axis, lat=lat_axis))

        assert "\n" not in text
        assert seed_cls.__name__ in text
        assert "4x5 grid" in text
        assert f"lon {lon_axis.min():.2f}..{lon_axis.max():.2f}" in text
        assert f"lat {lat_axis.min():.2f}..{lat_axis.max():.2f}" in text


def test_flowmap_repr_adds_the_release_time_and_signed_window(lon_axis, lat_axis):
    """The flow-map repr is the seed summary plus t0 and the signed window.

    The window carries its sign and nothing else: repelling versus attracting is
    a property of the diagnostic, not of the flow map, so the repr names no
    direction, since the sign of T already says which way the map runs.
    """
    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()

    forward = repr(seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME))
    backward = repr(
        seed.pset_to_flowmap(lon=lon, lat=lat, t0=END_TIME, t1=RELEASE_TIME)
    )

    assert "\n" not in forward
    assert "NeighborFlowMap" in forward
    assert "4x5 grid" in forward
    assert "t0 2020-01-01T00:00:00" in forward
    assert forward.endswith("T +1.0 days>")
    assert backward.endswith("T -1.0 days>")
    for text in (forward, backward):
        assert "repelling" not in text
        assert "attracting" not in text


def test_repr_survives_an_all_nan_grid(lon_axis, lat_axis):
    """An all-NaN dataset still reprs: every particle lost, and even the grid
    coords NaN, prints rather than raising or warning."""
    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon, _ = seed.to_parcels_pset()
    lost = [np.nan] * len(lon)
    fm = seed.pset_to_flowmap(lon=lost, lat=lost, t0=RELEASE_TIME, t1=END_TIME)

    assert "NeighborFlowMap" in repr(fm)
    # the grid itself is intact
    assert f"lon {lon_axis.min():.2f}..{lon_axis.max():.2f}" in repr(fm)

    nan_grid = NeighborFlowMap(
        fm.ds.assign_coords(
            lon_grid=fm.lon_grid * np.nan, lat_grid=fm.lat_grid * np.nan
        )
    )
    assert "lon nan..nan" in repr(nan_grid)


# --- keyword-only lon/lat pairs --------------------------------------------


def test_lonlat_pairs_are_keyword_only(lon_axis, lat_axis):
    """Passing a lon/lat pair positionally raises, so a swap cannot pass silently."""
    with pytest.raises(TypeError):
        NeighborSeedGrid.from_axes(lon_axis, lat_axis)
    with pytest.raises(TypeError):
        AuxiliarySeedGrid.from_axes(lon_axis, lat_axis)

    seed = NeighborSeedGrid.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    with pytest.raises(TypeError):
        seed.pset_to_flowmap(lon, lat, t0=RELEASE_TIME, t1=END_TIME)

    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)
    with pytest.raises(TypeError):
        fm.image(fm.ds["lon_0"], fm.ds["lat_0"])
