"""Metric tests: separations are taken in the local frame of each pair of points.

The package measures every separation in the local east/north frame of the two
points it connects -- the longitude difference wrapped and scaled by the cosine
of the pair's mid-latitude, the latitude difference by the Earth radius alone.
There is no shared projection and no standard parallel, which is what these
tests pin down:

* a rigid meridional translation is a genuine zonal deformation on a sphere
  (issue #18), because a parallel is shorter at higher latitude;
* a seed straddling the antimeridian behaves exactly like the same seed placed
  away from it, and longitudes above 180 survive ingest unchanged (issue #13);
* advected positions handed back wrapped to ``[-180, 180)`` -- what Parcels and
  CMEMS produce -- give the same gradF, FTLE and ``image()`` as the same
  positions on the seed's own branch;
* the auxiliary arms span ``2s`` at every latitude, measured by a geodesic
  written out here rather than by the package's own separation.

The flows here are prescribed directly on the emitted particle set rather than
through ``conftest``'s map helpers, which act in one flat tangent frame and so
would carry the very metric assumption under test.
"""

import numpy as np
import pytest
import xarray as xr

from lcs_parcels import AuxiliarySeedGrid, NeighborSeedGrid
from lcs_parcels.grids import (
    _circular_mean_lon,
    _wrap_lon,
)

RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")
T_SEC = abs((END_TIME - RELEASE_TIME) / np.timedelta64(1, "s"))

#: Auxiliary arm separation s (meters); the reference arm span is 2s.
AUX_S = 1_000.0

#: Mean Earth radius (meters) for the geodesic below. Written out rather than
#: imported from the package, so that geodesic is an independent measurement of
#: where the arms ended up and not a restatement of the formula that placed them.
R_M = 6_371_000.0


def _haversine_m(*, lon_a, lat_a, lon_b, lat_b):
    """Great-circle distance (meters) between two points on a sphere of radius
    ``R_M``, by the haversine formula."""
    phi_a, phi_b = np.deg2rad(lat_a), np.deg2rad(lat_b)
    half_dphi = 0.5 * (phi_b - phi_a)
    half_dlam = 0.5 * np.deg2rad(lon_b - lon_a)
    h = np.sin(half_dphi) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(half_dlam) ** 2
    return 2.0 * R_M * np.arcsin(np.sqrt(h))


def _translated_flowmap(*, lon, lat, dlon, dlat, aux_separation_m=AUX_S):
    """Seed from 1-D axes and rigidly translate every particle by ``(dlon, dlat)``.

    The translation is prescribed in degrees on the emitted particle set, so it
    is a rigid motion on the sphere's coordinates and carries no frame of its
    own. Returns the ingested ``AuxiliaryFlowMap``.
    """
    seed = AuxiliarySeedGrid.from_axes(
        lon=lon, lat=lat, aux_separation_m=aux_separation_m
    )
    lon_0, lat_0 = seed.to_parcels_pset()
    return seed.pset_to_flowmap(
        lon=np.asarray(lon_0) + dlon,
        lat=np.asarray(lat_0) + dlat,
        t0=RELEASE_TIME,
        t1=END_TIME,
    )


# --- rigid meridional translation (issue #18) ------------------------------


@pytest.mark.parametrize(
    ("lat_start", "dlat"),
    [(10.0, -8.0), (45.0, -15.0), (70.0, -15.0)],
)
def test_rigid_meridional_translation_stretches_zonally(lon_axis, lat_start, dlat):
    """A rigid shift in latitude deforms: gradF == diag(cos(phi+d)/cos(phi), 1).

    Moving every particle the same number of degrees north or south leaves each
    zonal separation spanning the same longitude difference, but a parallel is
    shorter at higher latitude, so that separation is stretched by
    ``cos(phi + d) / cos(phi)`` -- exactly what the local frames measure. The
    meridional separation is untouched, so the tensor is diagonal.

    Each case is equatorward (``d < 0``), so the ratio exceeds 1 and the FTLE is
    positive and equal to ``(1 / |T|) * log`` of that ratio, the larger singular
    value. ``AuxiliarySeedGrid`` keeps the field free of NaN edges.

    The single-standard-parallel frame this replaced returned ``gradF == I`` and
    a zero FTLE here; the last two assertions pin how large the missing signal
    is.
    """
    lat_axis = np.linspace(lat_start, lat_start + 4.0, 5)
    g = _translated_flowmap(lon=lon_axis, lat=lat_axis, dlon=0.0, dlat=dlat)
    gradF = g.deformation_gradient()

    phi = g.ds["lat_grid"]
    ratio = np.cos(np.deg2rad(phi + dlat)) / np.cos(np.deg2rad(phi))

    assert bool(gradF.notnull().all())
    assert float(abs(gradF.sel(row="x", col="x") - ratio).max()) < 1e-6
    assert float(abs(gradF.sel(row="y", col="y") - 1.0).max()) < 1e-6
    assert float(abs(gradF.sel(row="x", col="y")).max()) < 1e-6
    assert float(abs(gradF.sel(row="y", col="x")).max()) < 1e-6

    # The singular values are (ratio, 1) with ratio > 1, so the FTLE is
    # (1 / |T|) log(ratio).
    ftle = g.ftle()
    expected = np.log(ratio) / T_SEC
    assert float(abs(ftle - expected).max()) < 1e-12
    assert float(ftle.min()) > 0.0

    # Regression pin: the old frame measured no deformation at all here. The
    # zonal stretch is at least a percent, and |T| * FTLE is the log of it.
    assert float((ratio - 1.0).min()) > 0.01
    assert float((ftle * T_SEC).min()) > 0.01


# --- antimeridian (issue #13) ----------------------------------------------


def test_antimeridian_seed_matches_shifted_seed(lat_axis):
    """A seed crossing 180 gives the same diagnostics as one that does not.

    The axis runs 175..185 monotonically -- a wrapped axis (175, 180, -175) is
    not supported, since the interpolation axis would be non-monotonic. Both
    seeds are pushed through the same rigid translation, whose deformation
    depends on latitude alone, so the two fields must agree value for value:
    crossing the antimeridian costs nothing.
    """
    crossing = np.linspace(175.0, 185.0, 5)
    away = crossing - 40.0

    g_crossing = _translated_flowmap(lon=crossing, lat=lat_axis, dlon=0.7, dlat=-5.0)
    g_away = _translated_flowmap(lon=away, lat=lat_axis, dlon=0.7, dlat=-5.0)

    for a, b in [
        (g_crossing.deformation_gradient(), g_away.deformation_gradient()),
        (g_crossing.ftle(), g_away.ftle()),
    ]:
        assert bool(np.isfinite(a).all())
        xr.testing.assert_allclose(a.drop_vars("lon_grid"), b.drop_vars("lon_grid"))


def test_longitudes_above_180_are_stored_unchanged(lat_axis):
    """Ingest normalises nothing: what the user hands in comes back out.

    Longitudes are kept in whatever convention they arrive in -- only
    differences and means are wrapped -- so a seed and an advected position set
    beyond 180 stay beyond 180 in ``flowmap.ds``.
    """
    seed = AuxiliarySeedGrid.from_axes(lon=np.linspace(175.0, 185.0, 5), lat=lat_axis)
    lon_0, lat_0 = seed.to_parcels_pset()
    lon_advected = np.asarray(lon_0) + 0.5

    g = seed.pset_to_flowmap(lon=lon_advected, lat=lat_0, t0=RELEASE_TIME, t1=END_TIME)

    assert float(g.ds["lon_grid"].max()) == 185.0
    assert float(g.ds["lon_0"].max()) > 185.0
    stored = g.ds["lon"].stack(particle=g.ds["lon_0"].dims).values
    np.testing.assert_array_equal(stored, lon_advected)


def test_grid_image_and_circular_mean_straddle_the_antimeridian():
    """Arms reported on either branch of 180 still average to their centre.

    ``_circular_mean_lon`` anchors on the first member and averages wrapped
    offsets from it, so a stencil whose arms come back as 179.99 and -179.99
    averages to 180 rather than to the naive arithmetic mean of 0. The flow map
    is the identity with its advected longitudes reported in ``[-180, 180)``:
    the grid image must return to the release position and gradF to the
    identity.
    """
    lon = xr.DataArray(
        [179.99, -179.99, 179.99, -179.99],
        dims="displacement",
        coords={"displacement": ["east", "north", "west", "south"]},
    )
    assert abs(_wrap_lon(float(_circular_mean_lon(lon, "displacement")) - 180.0)) < 1e-9
    # What a plain mean would have said, i.e. the failure being guarded against.
    assert abs(float(lon.mean("displacement"))) < 1e-9

    seed = AuxiliarySeedGrid.from_axes(lon=np.array([180.0]), lat=np.array([10.0]))
    lon_0, lat_0 = seed.to_parcels_pset()
    # Identity flow, with the advected arms reported on the [-180, 180) branch.
    wrapped = ((np.asarray(lon_0) + 180.0) % 360.0) - 180.0
    assert wrapped.min() < 0.0 < wrapped.max()  # the arms really do straddle
    g = seed.pset_to_flowmap(lon=wrapped, lat=lat_0, t0=RELEASE_TIME, t1=END_TIME)

    image = g.grid_image.isel(i=0, j=0)
    assert abs(_wrap_lon(float(image["lon"]) - 180.0)) < 1e-9
    assert abs(float(image["lat"]) - 10.0) < 1e-9

    identity = xr.DataArray(
        np.eye(2), dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
    )
    assert float(abs(g.deformation_gradient() - identity).max()) < 1e-9


#: A monotonic seed axis around the antimeridian. 179.5 is on it deliberately:
#: the advection below carries that grid point's image onto 180 exactly, so its
#: four auxiliary arms -- a kilometre apart -- land on opposite branches once the
#: positions are folded. Without such a point the auxiliary stencil never
#: straddles, because a stencil only reaches 0.01 degrees.
WRAP_LON_AXIS = np.array([175.0, 177.0, 179.5, 182.0, 185.0])
WRAP_LAT_AXIS = np.linspace(10.0, 14.0, 5)


def _wrapped_and_plain_flowmaps(seed_cls):
    """One flow map from advected positions on the seed's branch, one from the
    same positions folded into ``[-180, 180)``.

    The advection is a translation plus a shear prescribed in degrees on the
    particle set, so both gradF columns are non-trivial and the off-diagonals are
    non-zero: a wrongly wrapped longitude difference cannot cancel out of the
    tensor.
    """
    seed = seed_cls.from_axes(lon=WRAP_LON_AXIS, lat=WRAP_LAT_AXIS)
    lon_0, lat_0 = (
        np.asarray(seed.to_parcels_pset()[0]),
        np.asarray(seed.to_parcels_pset()[1]),
    )
    lon = lon_0 + 0.5 + 0.02 * (lat_0 - 12.0)
    lat = lat_0 - 3.0 + 0.01 * (lon_0 - 180.0)
    folded = ((lon + 180.0) % 360.0) - 180.0
    assert folded.min() < 0.0 < folded.max()  # the images really do straddle
    return (
        seed.pset_to_flowmap(lon=lon, lat=lat, t0=RELEASE_TIME, t1=END_TIME),
        seed.pset_to_flowmap(lon=folded, lat=lat, t0=RELEASE_TIME, t1=END_TIME),
    )


@pytest.mark.parametrize("seed_cls", [AuxiliarySeedGrid, NeighborSeedGrid])
def test_wrapped_advected_positions_match_unwrapped_ones(seed_cls):
    """Advected longitudes returned on the ``[-180, 180)`` branch give the same
    gradF and FTLE as the same positions on the seed's own branch.

    This is the case the antimeridian actually arrives in (issue #13): the seed
    axis is monotonic 175..185, but the advection hands the positions back folded
    into ``[-180, 180)`` -- what Parcels and CMEMS do -- so the pair of points a
    stencil differences lands on opposite branches and their raw difference is
    off by 360. Both stencils meet it: the neighbour stencil straddles wherever
    the cut falls between two grid points, the auxiliary one only where a grid
    point's image lands within metres of 180, which ``WRAP_LON_AXIS`` arranges.

    What is left over is the fold, not the wrap. Wrapping a difference costs
    nothing -- it subtracts the nearest multiple of 360 and is bit-for-bit the
    identity on a difference already in range -- but folding a position near 180
    onto the far branch is a subtraction that keeps 15 digits of a 3-digit
    number, so the folded input is itself a slightly different position: 3e-14
    degrees, which is 3 nanometres. gradF inherits exactly that and no more.
    """
    plain, folded = _wrapped_and_plain_flowmaps(seed_cls)

    for a, b in [
        (plain.deformation_gradient(), folded.deformation_gradient()),
        (plain.ftle(), folded.ftle()),
    ]:
        # The neighbour stencil has no edge cells to difference, so it is NaN
        # there; the interior is where the answer lives, and it is finite.
        assert bool(np.isfinite(a.isel(i=slice(1, -1), j=slice(1, -1))).all())
        np.testing.assert_allclose(a.values, b.values, rtol=1e-11, atol=0.0)


@pytest.mark.parametrize("seed_cls", [AuxiliarySeedGrid, NeighborSeedGrid])
def test_image_reads_wrapped_advected_positions(seed_cls):
    """``image()`` interpolates the advected field across the branch cut.

    The advected longitudes are arithmetic once they reach the interpolant, so a
    field folded into ``[-180, 180)`` tears from 179.9 to -179.9 between two
    adjacent grid points and a linear interpolant reads the tear as a 40 000 km
    jump -- the whole grid image, not just the two cells at the cut, comes back
    wrong. Re-anchoring each image on the branch of its own grid point removes
    the tear, and the folded field then maps the same reference points to the
    same places the unfolded one does, to round-off in a longitude of order 180.
    """
    plain, folded = _wrapped_and_plain_flowmaps(seed_cls)

    # The folded grid image really is torn: adjacent grid points 2.5 degrees
    # apart in longitude come back over 300 degrees apart.
    torn = folded.grid_image["lon"].isel(j=0).values
    assert np.abs(np.diff(torn)).max() > 300.0

    query_lon = xr.DataArray([176.3, 179.4, 182.7], dims="p")
    query_lat = xr.DataArray([11.1, 12.5, 13.2], dims="p")
    image_plain = plain.image(lon_0=query_lon, lat_0=query_lat)
    image_folded = folded.image(lon_0=query_lon, lat_0=query_lat)

    # Read across the cut and on the seed's branch, not folded back into it.
    assert float(image_folded["lon"].max()) > 180.0
    assert float(abs(image_folded["lon"] - image_plain["lon"]).max()) < 1e-12
    np.testing.assert_array_equal(image_folded["lat"].values, image_plain["lat"].values)


# --- arm geometry ----------------------------------------------------------


def test_arm_spans_are_2s_by_an_independent_geodesic(lon_axis, lat_axis):
    """The auxiliary arms span ``2s``, measured by a geodesic of our own.

    ``AuxiliarySeedGrid.from_axes`` places the arms by inverting exactly the relation
    :func:`~lcs_parcels.grids._arm_separation_m` reads them back with, so
    checking one against the other cancels both the Earth radius and the cosine
    convention out of the answer. :func:`_haversine_m` is written out in this
    file against its own radius, so it does not: an arm placed with a different
    radius, or with a shared standard parallel, comes back the wrong length.

    The tolerance is what the closed form earns and no more. The east and west
    arms sit on a common parallel, and the geodesic between them cuts inside that
    parallel, so it is *shorter* than ``2s`` by about ``(s / R)^2 sin^2(phi)``
    relative: 5e-7 m at 12 N and 1.3e-4 m at 76 N for the 1 km arms here. The
    north and south arms lie on a meridian, which is itself a geodesic, so their
    span is exact to round-off.
    """
    seed = AuxiliarySeedGrid.from_axes(
        lon=lon_axis, lat=lat_axis, aux_separation_m=AUX_S
    )
    lon_0 = seed.ds["lon_0"].reset_coords(drop=True)
    lat_0 = seed.ds["lat_0"].reset_coords(drop=True)

    def span(positive, negative):
        return _haversine_m(
            lon_a=lon_0.sel(displacement=negative, drop=True),
            lat_a=lat_0.sel(displacement=negative, drop=True),
            lon_b=lon_0.sel(displacement=positive, drop=True),
            lat_b=lat_0.sel(displacement=positive, drop=True),
        )

    span_x = span("east", "west")
    span_y = span("north", "south")
    assert float(abs(span_x - 2 * AUX_S).max()) < 5e-4
    assert float(abs(span_y - 2 * AUX_S).max()) < 1e-8
    # A shared standard parallel would stretch the span by cos(parallel) /
    # cos(phi), which is 1096 m across the 68-76 N band; this is flat to the
    # geodesic's own sag.
    assert float(span_x.max() - span_x.min()) < 5e-4


def test_arms_spanning_90_degrees_of_longitude_are_rejected():
    """Near a pole ``s`` is more than 90 degrees of longitude, and an arm placed
    there aliases through the wrap onto the far side of the pole -- a wrong
    gradient rather than a NaN. That is refused, not approximated.

    The guard is on the *placement*, not on the latitude: it trips wherever
    ``R cos(lat)`` falls below ``2 s / pi``, so the latitude it fires at moves
    with ``s``. Asserted here by driving the same latitude to both outcomes with
    two separations three orders of magnitude apart.
    """
    # 2 s / pi is 637 m from the pole at s = 1 km, so 89.999 N (111 m) is inside
    # the guard and 89.9 N (11 km) is outside it.
    with pytest.raises(ValueError, match="90 degrees"):
        AuxiliarySeedGrid.from_axes(
            lon=np.array([0.0]), lat=np.array([89.999]), aux_separation_m=AUX_S
        )
    AuxiliarySeedGrid.from_axes(
        lon=np.array([0.0]), lat=np.array([89.9]), aux_separation_m=AUX_S
    )
    # Same latitudes, s scaled by 1e-3 and 1e3: the threshold moves with s, so
    # 89.999 N now passes and 89.9 N now raises.
    AuxiliarySeedGrid.from_axes(
        lon=np.array([0.0]), lat=np.array([89.999]), aux_separation_m=AUX_S / 1e3
    )
    with pytest.raises(ValueError, match="90 degrees"):
        AuxiliarySeedGrid.from_axes(
            lon=np.array([0.0]), lat=np.array([89.9]), aux_separation_m=AUX_S * 1e3
        )
