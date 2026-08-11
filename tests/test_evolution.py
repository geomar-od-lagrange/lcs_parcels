"""FlowMap.image: interpolating the flow map at arbitrary reference points.

``image`` maps reference positions ``x_0`` to their advected positions
``F_{t0}^{t1}(x_0)``, the primitive that evolves an extracted material curve.
For a constant linear flow map ``F(x) = M @ x`` the advected-position field is
linear in ``x_0``, so linear interpolation is *exact*. At a grid node it
returns the node's stored advected position, and at a midpoint it returns the
average of the two nodes. ``conftest.advected_flowmap`` builds such a map on a
``NeighborSeedGrid`` (its advected positions live on the ``(i, j)`` grid, exactly the
rectilinear field ``image`` reads).
"""

import numpy as np
import xarray as xr
from conftest import (
    advected_flowmap,
    advected_scattered_flowmap,
    apply_linear_map_to_pset,
)

from lcs_parcels import (
    AuxiliarySeedGrid,
    NeighborSeedGrid,
    UnstructuredAuxiliarySeedGrid,
)
from lcs_parcels.grids import _wrap_lon

RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")
M = np.array([[2.0, 0.5], [0.0, 3.0]])  # generic (sheared) linear map


def _flowmap(lon_axis, lat_axis):
    return advected_flowmap(
        NeighborSeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )


def test_image_at_grid_node_returns_stored_position(lon_axis, lat_axis):
    """At a reference grid node, image returns that node's stored advected position."""
    fm = _flowmap(lon_axis, lat_axis)
    node = {"i": 1, "j": 2}
    out = fm.image(lon_0=fm.ds["lon_0"].isel(**node), lat_0=fm.ds["lat_0"].isel(**node))

    assert np.isclose(out["lon"], fm.ds["lon"].isel(**node))
    assert np.isclose(out["lat"], fm.ds["lat"].isel(**node))


def test_image_off_node_is_exact_for_linear_map(lon_axis, lat_axis):
    """A linear flow map interpolates exactly, so a midpoint maps to the node
    average."""
    fm = _flowmap(lon_axis, lat_axis)
    a, b = {"i": 1, "j": 2}, {"i": 2, "j": 2}  # neighbours along i (same latitude row)
    lon0_mid = 0.5 * (fm.ds["lon_0"].isel(**a) + fm.ds["lon_0"].isel(**b))
    lat0_mid = fm.ds["lat_0"].isel(**a)
    out = fm.image(lon_0=lon0_mid, lat_0=lat0_mid)

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
    along = fm.image(lon_0=curve["lon_0"], lat_0=curve["lat_0"])
    assert along["lon"].dims == ("param",)

    grid = fm.ds[["lon_0", "lat_0"]].rename(i="line", j="point")
    lattice = fm.image(lon_0=grid["lon_0"], lat_0=grid["lat_0"])
    assert set(lattice["lon"].dims) == {"line", "point"}


def test_image_returns_the_reference_points_as_x0(lon_axis, lat_axis):
    """The requested reference positions come back as lon_0/lat_0, since they are x_0,
    not diagnostic grid points, so they must not reuse the lon_grid/lat_grid name."""
    fm = _flowmap(lon_axis, lat_axis)
    curve = fm.ds[["lon_0", "lat_0"]].isel(j=2).rename(i="param")
    out = fm.image(lon_0=curve["lon_0"], lat_0=curve["lat_0"])

    assert "lon_grid" not in out.coords
    assert "lat_grid" not in out.coords
    assert np.allclose(out["lon_0"], curve["lon_0"])
    assert np.allclose(out["lat_0"], curve["lat_0"])
    assert "reference" in out["lon_0"].attrs["long_name"]


def test_image_off_grid_and_nan_inputs_are_nan(lon_axis, lat_axis):
    """Off-grid points and NaN inputs map to NaN. Valid points stay finite."""
    fm = _flowmap(lon_axis, lat_axis)
    centre_lon = float(fm.ds["lon_0"].mean())
    centre_lat = float(fm.ds["lat_0"].mean())

    lon_0 = xr.DataArray([centre_lon, lon_axis[0] - 50.0, np.nan], dims="param")
    lat_0 = xr.DataArray([centre_lat, lat_axis[0] - 50.0, centre_lat], dims="param")
    out = fm.image(lon_0=lon_0, lat_0=lat_0)

    assert np.isfinite(out["lon"].isel(param=0))  # interior point: finite
    assert bool(out["lon"].isel(param=1).isnull())  # off-grid: NaN
    assert bool(out["lon"].isel(param=2).isnull())  # NaN input: NaN


def test_image_accepts_cf_named_indexer_dims(lon_axis, lat_axis):
    """Reference points whose *dims* are called ``lon``/``lat`` are accepted.

    This pattern is what ``grid["lon"]``, ``grid["lat"]`` hand you off an
    ordinary rectilinear CF dataset, the most natural way to call ``image``. It
    collides with the names of the returned data variables. Building the result
    by merging (``assign``/``assign_coords``) cannot tell an incoming ``lon``
    data variable from a dimension of the same name, and raises ``MergeError``
    here. This test therefore covers both the outer-product case (two dims,
    ``lon`` and ``lat``) and a single shared dim named ``lon``.
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

    outer = fm.image(lon_0=grid["lon"], lat_0=grid["lat"])
    assert set(outer["lon"].dims) == {"lon", "lat"}
    assert outer["lon"].notnull().all()

    shared = fm.image(
        lon_0=xr.DataArray([centre_lon, centre_lon + 0.1], dims="lon"),
        lat_0=xr.DataArray([centre_lat, centre_lat + 0.1], dims="lon"),
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
    supports. It maps the grid point through the flow map (arm centroid)."""
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    node = {"i": 1, "j": 2}
    lon_grid = fm.ds["lon_grid"].isel(**node)
    lat_grid = fm.ds["lat_grid"].isel(**node)
    out = fm.image(lon_0=lon_grid, lat_0=lat_grid)

    # For a linear map the arm centroid is exactly the grid point's advected position.
    origin = (float(fm.ds["lon_0"].mean()), float(fm.ds["lat_0"].mean()))
    exp_lon, exp_lat = apply_linear_map_to_pset(
        [float(lon_grid)], [float(lat_grid)], M, origin
    )
    assert np.isclose(out["lon"], exp_lon[0])
    assert np.isclose(out["lat"], exp_lat[0])


def test_image_auxiliary_lost_arm_maps_to_nan(lon_axis, lat_axis):
    """A grid point with a lost (NaN) arm images to NaN, not the centroid of the
    surviving arms, matching the deformation-gradient path."""
    fm = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    bad = (fm.ds["i"] == 1) & (fm.ds["j"] == 2) & (fm.ds["displacement"] == "west")
    fm.ds["lon"] = fm.ds["lon"].where(~bad)

    out = fm.image(lon_0=fm.ds["lon_grid"], lat_0=fm.ds["lat_grid"])
    assert bool(out["lon"].isel(i=1, j=2).isnull())  # the crippled grid point
    assert bool(out["lon"].isel(i=2, j=2).notnull())  # an intact neighbour


def test_image_on_an_unstructured_flowmap(lon_axis, lat_axis):
    """``image`` reads the flow map between grid points that carry no axes.

    The map is linear, so a linear interpolant is exact whether it reads the
    field along two axes or over a triangulation. The answer at a grid point is
    that point's own stored image, and at the midpoint of two, the average.
    """
    fm = advected_scattered_flowmap(lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)
    image = fm.grid_image
    a, b = 0, 1 + lat_axis.size  # two points on neither the same row nor column

    at_nodes = fm.image(lon_0=fm.lon_grid, lat_0=fm.lat_grid)
    # Compared as a difference, because the image comes back re-anchored on each
    # grid point's branch and the advected positions arrive wrapped.
    assert np.allclose(_wrap_lon(at_nodes["lon"] - image["lon"]), 0.0)
    assert np.allclose(at_nodes["lat"], image["lat"])

    midpoint = fm.image(
        lon_0=fm.lon_grid.isel(grid_point=[a, b]).mean().expand_dims("param"),
        lat_0=fm.lat_grid.isel(grid_point=[a, b]).mean().expand_dims("param"),
    )
    assert np.allclose(
        midpoint["lat"], float(image["lat"].isel(grid_point=[a, b]).mean()), atol=1e-6
    )


def test_image_outside_the_convex_hull_is_nan(lon_axis, lat_axis):
    """A reference point the grid points do not enclose maps to NaN, the
    scattered counterpart of falling off a rectilinear grid."""
    fm = advected_scattered_flowmap(lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)

    out = fm.image(
        lon_0=xr.DataArray([float(lon_axis[0]) - 50.0], dims="param"),
        lat_0=xr.DataArray([float(lat_axis[0])], dims="param"),
    )

    assert bool(out["lon"].isnull().all())


def test_the_two_interpolators_agree_on_the_same_region(lon_axis, lat_axis):
    """One region read along two axes and off a triangulation gives one answer.

    ``UnstructuredAuxiliarySeedGrid`` inherits ``from_axes``, so the same grid
    points can be put through either interpolator. The flow map is linear, so
    both are exact and the two must agree. Nothing else in the suite would catch
    the scattered path quietly falling back to the rectilinear one, or the seam
    being handed the wrong array.
    """
    structured = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    scattered = advected_flowmap(
        UnstructuredAuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    assert type(structured._interpolator(structured.ftle())) is not type(
        scattered._interpolator(scattered.ftle())
    )

    # Midway between four grid points, where the two schemes could differ.
    lon_0 = 0.5 * (structured.lon_grid.isel(i=1) + structured.lon_grid.isel(i=2))
    lat_0 = 0.5 * (structured.lat_grid.isel(j=1) + structured.lat_grid.isel(j=2))
    lon_0, lat_0 = xr.broadcast(lon_0.isel(j=1, drop=True), lat_0.isel(i=1, drop=True))

    a = structured.image(lon_0=lon_0, lat_0=lat_0)
    b = scattered.image(lon_0=lon_0, lat_0=lat_0)

    assert np.allclose(_wrap_lon(a["lon"] - b["lon"]), 0.0, atol=1e-6)
    assert np.allclose(a["lat"], b["lat"], atol=1e-6)


def test_image_does_not_relabel_its_output_from_the_indexers(lon_axis, lat_axis):
    """Attributes the reference points carried stay off the advected ones.

    Reference points read off a CF dataset bring a ``standard_name`` and an
    ``axis``, and a returned latitude that inherits the longitude's would tell
    every CF-aware consumer the wrong thing.
    """
    fm = _flowmap(lon_axis, lat_axis)
    cf = {"standard_name": "longitude", "axis": "X"}

    out = fm.image(
        lon_0=xr.DataArray([float(lon_axis[1])], dims="param", attrs=cf),
        lat_0=xr.DataArray([float(lat_axis[1])], dims="param"),
    )

    for name in ("lon", "lat"):
        assert "standard_name" not in out[name].attrs
        assert "axis" not in out[name].attrs
    assert out["lat"].attrs["units"] == "degrees_north"
    assert "latitude" in out["lat"].attrs["long_name"]
