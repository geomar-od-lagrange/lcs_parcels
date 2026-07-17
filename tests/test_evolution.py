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

from conftest import advected_flowmap
from lcs_parcels import NeighborSeed

T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")
M = np.array([[2.0, 0.5], [0.0, 3.0]])  # generic (sheared) linear map


def _flowmap(lon_axis, lat_axis):
    return advected_flowmap(NeighborSeed, lon_axis, lat_axis, M, T0, T1)


def test_image_at_grid_node_returns_stored_position(lon_axis, lat_axis):
    """At a reference grid node, image returns that node's stored advected position."""
    fm = _flowmap(lon_axis, lat_axis)
    node = dict(i=1, j=2)
    out = fm.image(fm.ds["lon_0"].isel(**node), fm.ds["lat_0"].isel(**node))

    assert np.isclose(out["lon"], fm.ds["lon"].isel(**node))
    assert np.isclose(out["lat"], fm.ds["lat"].isel(**node))


def test_image_off_node_is_exact_for_linear_map(lon_axis, lat_axis):
    """A linear flow map interpolates exactly: a midpoint maps to the node average."""
    fm = _flowmap(lon_axis, lat_axis)
    a, b = dict(i=1, j=2), dict(i=2, j=2)  # neighbours along i (same latitude row)
    lon0_mid = 0.5 * (fm.ds["lon_0"].isel(**a) + fm.ds["lon_0"].isel(**b))
    lat0_mid = fm.ds["lat_0"].isel(**a)
    out = fm.image(lon0_mid, lat0_mid)

    assert np.isclose(out["lon"], 0.5 * (fm.ds["lon"].isel(**a) + fm.ds["lon"].isel(**b)))
    assert np.isclose(out["lat"], 0.5 * (fm.ds["lat"].isel(**a) + fm.ds["lat"].isel(**b)))


def test_image_preserves_indexer_dims(lon_axis, lat_axis):
    """The output carries whatever dims the reference points do (param, or line/point)."""
    fm = _flowmap(lon_axis, lat_axis)

    curve = fm.ds[["lon_0", "lat_0"]].isel(j=2).rename(i="param")
    along = fm.image(curve["lon_0"], curve["lat_0"])
    assert along["lon"].dims == ("param",)

    grid = fm.ds[["lon_0", "lat_0"]].rename(i="line", j="point")
    lattice = fm.image(grid["lon_0"], grid["lat_0"])
    assert set(lattice["lon"].dims) == {"line", "point"}


def test_image_off_grid_and_nan_inputs_are_nan(lon_axis, lat_axis):
    """Off-grid points and NaN inputs map to NaN; valid points stay finite."""
    fm = _flowmap(lon_axis, lat_axis)
    centre_lon = float(fm.ds["lon_0"].mean())
    centre_lat = float(fm.ds["lat_0"].mean())

    lon0 = xr.DataArray([centre_lon, lon_axis[0] - 50.0, np.nan], dims="param")
    lat0 = xr.DataArray([centre_lat, lat_axis[0] - 50.0, centre_lat], dims="param")
    out = fm.image(lon0, lat0)

    assert np.isfinite(out["lon"].isel(param=0))  # interior point: finite
    assert bool(out["lon"].isel(param=1).isnull())  # off-grid: NaN
    assert bool(out["lon"].isel(param=2).isnull())  # NaN input: NaN
