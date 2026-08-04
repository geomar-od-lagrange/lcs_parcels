"""FlowMap.image: interpolating the flow map at arbitrary reference points.

``image`` maps reference positions ``x_0`` to their advected positions
``F_{t0}^{t1}(x_0)`` -- the primitive that evolves an extracted material curve.
For a constant linear flow map ``F(x) = M @ x`` the advected-position field is
linear in ``x_0``, so linear interpolation is *exact*: at a grid node it returns
the node's stored advected position, and at a midpoint it returns the average of
the two nodes. ``conftest.advected_flowmap`` builds such a map on a
``NeighborSeed`` (its advected positions live on the ``(i, j)`` grid, exactly the
rectilinear field ``image`` reads).
"""

import numpy as np
import xarray as xr
from conftest import advected_flowmap, apply_linear_map_to_pset

from lcs_parcels import AuxiliarySeed, NeighborSeed

RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")
M = np.array([[2.0, 0.5], [0.0, 3.0]])  # generic (sheared) linear map


def _flowmap(lon_axis, lat_axis):
    return advected_flowmap(NeighborSeed, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)


def test_image_at_grid_node_returns_stored_position(lon_axis, lat_axis):
    """At a reference grid node, image returns that node's stored advected position."""
    fm = _flowmap(lon_axis, lat_axis)
    node = {"i": 1, "j": 2}
    out = fm.image(lon0=fm.ds["lon_0"].isel(**node), lat0=fm.ds["lat_0"].isel(**node))

    assert np.isclose(out["lon"], fm.ds["lon"].isel(**node))
    assert np.isclose(out["lat"], fm.ds["lat"].isel(**node))


def test_image_off_node_is_exact_for_linear_map(lon_axis, lat_axis):
    """A linear flow map interpolates exactly: a midpoint maps to the node average."""
    fm = _flowmap(lon_axis, lat_axis)
    a, b = {"i": 1, "j": 2}, {"i": 2, "j": 2}  # neighbours along i (same latitude row)
    lon0_mid = 0.5 * (fm.ds["lon_0"].isel(**a) + fm.ds["lon_0"].isel(**b))
    lat0_mid = fm.ds["lat_0"].isel(**a)
    out = fm.image(lon0=lon0_mid, lat0=lat0_mid)

    assert np.isclose(
        out["lon"], 0.5 * (fm.ds["lon"].isel(**a) + fm.ds["lon"].isel(**b))
    )
    assert np.isclose(
        out["lat"], 0.5 * (fm.ds["lat"].isel(**a) + fm.ds["lat"].isel(**b))
    )


def test_image_preserves_indexer_dims(lon_axis, lat_axis):
    """The output carries whatever dims the reference points do (param, or line/point)."""
    fm = _flowmap(lon_axis, lat_axis)

    curve = fm.ds[["lon_0", "lat_0"]].isel(j=2).rename(i="param")
    along = fm.image(lon0=curve["lon_0"], lat0=curve["lat_0"])
    assert along["lon"].dims == ("param",)

    grid = fm.ds[["lon_0", "lat_0"]].rename(i="line", j="point")
    lattice = fm.image(lon0=grid["lon_0"], lat0=grid["lat_0"])
    assert set(lattice["lon"].dims) == {"line", "point"}


def test_image_returns_the_reference_points_as_x0(lon_axis, lat_axis):
    """The requested reference positions come back as lon_0/lat_0 -- they are x_0,
    not diagnostic grid points, so they must not reuse the lon_grid/lat_grid name."""
    fm = _flowmap(lon_axis, lat_axis)
    curve = fm.ds[["lon_0", "lat_0"]].isel(j=2).rename(i="param")
    out = fm.image(lon0=curve["lon_0"], lat0=curve["lat_0"])

    assert "lon_grid" not in out.coords
    assert "lat_grid" not in out.coords
    assert np.allclose(out["lon_0"], curve["lon_0"])
    assert np.allclose(out["lat_0"], curve["lat_0"])
    assert "reference" in out["lon_0"].attrs["long_name"]


def test_image_off_grid_and_nan_inputs_are_nan(lon_axis, lat_axis):
    """Off-grid points and NaN inputs map to NaN; valid points stay finite."""
    fm = _flowmap(lon_axis, lat_axis)
    centre_lon = float(fm.ds["lon_0"].mean())
    centre_lat = float(fm.ds["lat_0"].mean())

    lon0 = xr.DataArray([centre_lon, lon_axis[0] - 50.0, np.nan], dims="param")
    lat0 = xr.DataArray([centre_lat, lat_axis[0] - 50.0, centre_lat], dims="param")
    out = fm.image(lon0=lon0, lat0=lat0)

    assert np.isfinite(out["lon"].isel(param=0))  # interior point: finite
    assert bool(out["lon"].isel(param=1).isnull())  # off-grid: NaN
    assert bool(out["lon"].isel(param=2).isnull())  # NaN input: NaN


def test_image_accepts_cf_named_indexer_dims(lon_axis, lat_axis):
    """Reference points whose *dims* are called ``lon``/``lat`` are accepted.

    That is what ``grid["lon"]``, ``grid["lat"]`` hand you off an ordinary
    rectilinear CF dataset -- the most natural way to call ``image`` -- and it
    collides with the names of the returned data variables. Building the result
    by merging (``assign``/``assign_coords``) cannot tell an incoming ``lon``
    data variable from a dimension of the same name and raises ``MergeError``
    here, so this covers both the outer-product case (two dims, ``lon`` and
    ``lat``) and a single shared dim named ``lon``.
    """
    fm = _flowmap(lon_axis, lat_axis)
    centre_lon = float(fm.ds["lon_0"].mean())
    centre_lat = float(fm.ds["lat_0"].mean())
    grid = xr.Dataset(
        coords={
            "lon": ("lon", [centre_lon, centre_lon + 0.1]),
            "lat": ("lat", [centre_lat, centre_lat + 0.1, centre_lat + 0.2]),
        }
    )

    outer = fm.image(lon0=grid["lon"], lat0=grid["lat"])
    assert set(outer["lon"].dims) == {"lon", "lat"}
    assert outer["lon"].notnull().all()

    shared = fm.image(
        lon0=xr.DataArray([centre_lon, centre_lon + 0.1], dims="lon"),
        lat0=xr.DataArray([centre_lat, centre_lat + 0.1], dims="lon"),
    )
    assert shared["lon"].dims == ("lon",)
    assert shared["lon"].notnull().all()

    for out in (outer, shared):
        assert out["lon"].attrs["units"] == "degrees_east"
        assert out["lat"].attrs["units"] == "degrees_north"
        assert "advected" in out["lon"].attrs["long_name"]
        assert "advected" in out["lat"].attrs["long_name"]
        assert "reference" in out["lon_0"].attrs["long_name"]
        assert "reference" in out["lat_0"].attrs["long_name"]
        assert out["lon_0"].attrs["units"] == "degrees_east"
        assert out["lat_0"].attrs["units"] == "degrees_north"


def test_image_on_auxiliary_flowmap(lon_axis, lat_axis):
    """image works on the auxiliary (i, j, displacement) layout that shrink_lines
    supports: it maps the grid point through the flow map (arm centroid)."""
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)
    node = {"i": 1, "j": 2}
    lon_grid = fm.ds["lon_grid"].isel(**node)
    lat_grid = fm.ds["lat_grid"].isel(**node)
    out = fm.image(lon0=lon_grid, lat0=lat_grid)

    # For a linear map the arm centroid is exactly the grid point's advected position.
    origin = (float(fm.ds["lon_0"].mean()), float(fm.ds["lat_0"].mean()))
    exp_lon, exp_lat = apply_linear_map_to_pset(
        [float(lon_grid)], [float(lat_grid)], M, origin
    )
    assert np.isclose(out["lon"], exp_lon[0])
    assert np.isclose(out["lat"], exp_lat[0])


def test_image_auxiliary_lost_arm_maps_to_nan(lon_axis, lat_axis):
    """A grid point with a lost (NaN) arm images to NaN, not the centroid of the
    surviving arms -- matching the deformation-gradient path."""
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)
    bad = (fm.ds["i"] == 1) & (fm.ds["j"] == 2) & (fm.ds["displacement"] == "west")
    fm.ds["lon"] = fm.ds["lon"].where(~bad)

    out = fm.image(lon0=fm.ds["lon_grid"], lat0=fm.ds["lat_grid"])
    assert bool(out["lon"].isel(i=1, j=2).isnull())  # the crippled grid point
    assert bool(out["lon"].isel(i=2, j=2).notnull())  # an intact neighbour
