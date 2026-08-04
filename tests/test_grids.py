"""Shape / dim / coord contracts for the Seed and FlowMap families.

A ``Seed`` is time-free: ``from_axes`` takes no ``t0``, and the dataset is
all-coordinates (no ``t0``/``T``, no advected ``lon``/``lat``). Time and the
advected positions enter at ingest, ``seed.pset_to_flowmap(lon=..., lat=...,
t0=..., t1=...)``, which returns a ``FlowMap`` carrying scalar ``t0``/``T``
coords and ``lon``/``lat`` data vars.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import advected_flowmap

from lcs_parcels import AuxiliarySeed, NeighborFlowMap, NeighborSeed
from lcs_parcels.grids import _lonlat_to_meters

# Release/end times supplied only at ingest (the seed itself is time-free).
RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")

# Generic (sheared) linear flow map, for the diagnostics driven below.
M = np.array([[2.0, 0.5], [0.0, 3.0]])


# --- seed shape ------------------------------------------------------------


def test_neighbor_seed_from_axes_dims(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    ds = seed.ds

    # curvilinear reference lon_0/lat_0 and diagnostic grid lon_grid/lat_grid on
    # logical dims (i, j)
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert set(ds["lon_grid"].dims) == {"i", "j"}
    assert set(ds["lat_grid"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0

    # NeighborSeed carries no displacement stencil dim
    assert "displacement" not in ds.dims


def test_neighbor_seed_lon_lat_values(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
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
    seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)
    ds = seed.ds

    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # AuxiliarySeed adds a four-arm stencil on the `displacement` dim (no centre,
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

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference + grid coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ns = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    aus = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)

    # NeighborSeed has no auxiliary stencil: no displacement dim, and its
    # reference positions are the grid points themselves on (i, j).
    assert "displacement" not in ns.ds.dims
    assert set(ns.ds["lon_0"].dims) == {"i", "j"}

    # AuxiliarySeed carries the four-arm displacement stencil, so its reference
    # positions are per-arm and distinct from the grid points.
    assert aus.ds.sizes["displacement"] == 4
    assert set(aus.ds["lon_0"].dims) == {"i", "j", "displacement"}


def test_grid_coords_are_canonical_across_stencils(lon_axis, lat_axis):
    """Both stencils carry the diagnostic grid under one name, on (i, j), and both
    expose it as .lon_grid/.lat_grid. For the neighbour stencil it equals the
    reference release positions -- stored, not reconstructed."""
    ns = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    aus = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)

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
    """AuxiliarySeed places its four arms at exactly +/- ``aux_separation_m``.

    Read the arm positions back into the meters frame
    (:func:`_lonlat_to_meters`) with a non-default separation and assert: the
    east-west and north-south spans are each exactly ``2s``, opposing arms share
    the centre on the other axis (no cross-offset), and the geographic direction
    of each labelled arm is correct.
    """
    s = 2_500.0  # non-default aux_separation_m
    seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis, aux_separation_m=s)
    lon0, lat0 = seed.ds["lon_0"], seed.ds["lat_0"]
    lon_grid, lat_grid = seed.ds["lon_grid"], seed.ds["lat_grid"]
    lon_ref, lat_ref = float(lon_grid.mean()), float(lat_grid.mean())
    x, y = _lonlat_to_meters(lon0, lat0, lon_ref, lat_ref)  # dims (i, j, displacement)

    ew = x.sel(displacement="east") - x.sel(displacement="west")
    ns = y.sel(displacement="north") - y.sel(displacement="south")
    assert float(abs(ew - 2 * s).max()) < 1e-6  # east-west span == 2s
    assert float(abs(ns - 2 * s).max()) < 1e-6  # north-south span == 2s

    # opposing arms share the centre on the OTHER axis (no cross-offset)
    assert (
        float(abs(y.sel(displacement="east") - y.sel(displacement="west")).max()) < 1e-6
    )
    assert (
        float(abs(x.sel(displacement="north") - x.sel(displacement="south")).max())
        < 1e-6
    )

    # geographic direction is correct
    assert (
        float((lon0.sel(displacement="east") - lon0.sel(displacement="west")).min()) > 0
    )
    assert (
        float((lat0.sel(displacement="north") - lat0.sel(displacement="south")).min())
        > 0
    )


# --- flow-map shape --------------------------------------------------------


def test_neighbor_flowmap_shape(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
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
    seed = AuxiliarySeed.from_axes(lon=lon_axis, lat=lat_axis)
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
    for seed_cls in (NeighborSeed, AuxiliarySeed):
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


@pytest.mark.parametrize("seed_cls", (NeighborSeed, AuxiliarySeed))
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
    fm.image(lon0=fm.ds["lon_grid"], lat0=fm.ds["lat_grid"])
    fm.hyperbolic_lcs(step_m=1_000.0, line_length_m=6_000.0)
    fm.to_seed()

    xr.testing.assert_identical(fm.ds, before)


# --- reprs -----------------------------------------------------------------


def test_seed_repr_is_a_one_line_summary(lon_axis, lat_axis):
    """Class, grid shape and lon/lat extent on one line -- not the dataset."""
    for seed_cls in (NeighborSeed, AuxiliarySeed):
        text = repr(seed_cls.from_axes(lon=lon_axis, lat=lat_axis))

        assert "\n" not in text
        assert seed_cls.__name__ in text
        assert "4x5 grid" in text
        assert "lon -2.00..1.00" in text
        assert "lat 10.00..14.00" in text


def test_flowmap_repr_adds_the_release_time_and_signed_window(lon_axis, lat_axis):
    """The flow-map repr is the seed summary plus t0 and the signed window.

    The window carries its sign and nothing else: repelling versus attracting is
    a property of the diagnostic, not of the flow map, so the repr names no
    direction -- the sign of T already says which way the map runs.
    """
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
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
    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, _ = seed.to_parcels_pset()
    lost = [np.nan] * len(lon)
    fm = seed.pset_to_flowmap(lon=lost, lat=lost, t0=RELEASE_TIME, t1=END_TIME)

    assert "NeighborFlowMap" in repr(fm)
    assert "lon -2.00..1.00" in repr(fm)  # the grid itself is intact

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
        NeighborSeed.from_axes(lon_axis, lat_axis)
    with pytest.raises(TypeError):
        AuxiliarySeed.from_axes(lon_axis, lat_axis)

    seed = NeighborSeed.from_axes(lon=lon_axis, lat=lat_axis)
    lon, lat = seed.to_parcels_pset()
    with pytest.raises(TypeError):
        seed.pset_to_flowmap(lon, lat, t0=RELEASE_TIME, t1=END_TIME)

    fm = seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME)
    with pytest.raises(TypeError):
        fm.image(fm.ds["lon_0"], fm.ds["lat_0"])
