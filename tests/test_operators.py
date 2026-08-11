"""Operator tests: gradF -> C -> eigen -> FTLE (diagnostics on a FlowMap).

The ``conftest.advected_flowmap`` helper seeds a grid, emits its particle set,
and advects through ``M`` about the seed centroid. It then ingests via
``seed.pset_to_flowmap``, and the signed window ``T = t1 - t0`` lands on the
``FlowMap``.

That advection acts in the one tangent frame at the centroid, while the package
measures every separation in the local east/north frame of the pair it connects.
So a constant ``M`` does not give a constant gradF. The expectation is ``M``
rescaled per grid point by ``conftest.analytic_gradient``, and the whole chain
downstream of it varies over the grid too.

A non-symmetric ``M = [[2.0, 0.5], [0.0, 3.0]]`` is used for the general tests so
that ``C`` is a non-trivial check.
"""

import numpy as np
import pytest
import xarray as xr
from conftest import (
    DEG,
    EARTH_RADIUS_M,
    advected_flowmap,
    advected_flowmap_f,
    advected_scattered_flowmap,
    analytic_gradient,
    apply_map_to_lonlat,
    local_frame_gradient,
    seed_origin,
)

from lcs_parcels import AuxiliarySeedGrid, NeighborSeedGrid
from lcs_parcels.grids import _arm_separation_m, _central_separation_m, _separation_m

# Release time and integration end time; the signed window T = END_TIME - RELEASE_TIME spans one
# day (|T| = 86400 s).
RELEASE_TIME = np.datetime64("2020-01-01")
END_TIME = np.datetime64("2020-01-02")

# Non-symmetric linear flow map, applied in the seed-centroid tangent frame.
M = np.array([[2.0, 0.5], [0.0, 3.0]])

# The same map as a (row, col) tensor. gradF is *not* equal to this. It is this
# rescaled into the local frames, so it serves as the contrast case.
M_TENSOR = xr.DataArray(
    M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
)

# Integration window in seconds (one day) used by the analytic FTLE.
T_SEC = abs((END_TIME - RELEASE_TIME) / np.timedelta64(1, "s"))


def cauchy_green_of(gradF):
    """``(grad F)^T grad F`` for a ``(row, col)`` tensor, contracted over ``row``."""
    left = gradF.rename({"col": "row_out"})
    right = gradF.rename({"col": "col_out"})
    product = xr.dot(left, right, dim="row")
    return product.rename({"row_out": "row", "col_out": "col"})


def analytic_eigenvalues(gradF):
    """Ascending eigenvalues of ``gradF^T gradF``, on ``(i, j, eig)``."""
    C = cauchy_green_of(gradF).transpose("i", "j", "row", "col")
    values = np.linalg.eigvalsh(C.values)
    return xr.DataArray(
        values, dims=("i", "j", "eig"), coords={"eig": [0, 1]}
    ).assign_coords(i=C["i"], j=C["j"])


# --- stencil separations ----------------------------------------------------


def test_central_separation_spans_two_cells_and_nans_the_edges():
    """``_central_separation_m`` is ``(index + 1) - (index - 1)``, NaN at both ends."""
    lon = xr.DataArray(np.arange(5.0)[:, None] * np.ones(3), dims=("i", "j"))
    lat = xr.zeros_like(lon)

    dx, dy = _central_separation_m(lon, lat, "i")

    assert set(dx.dims) == {"i", "j"}
    assert np.isnan(dx.isel(i=0)).all()
    assert np.isnan(dx.isel(i=-1)).all()
    # Two one-degree cells along i, on the equator, so the span is 2 degrees of
    # longitude in meters everywhere inside; nothing moves north.
    assert np.allclose(dx.isel(i=slice(1, -1)), 2.0 * EARTH_RADIUS_M * DEG)
    assert np.allclose(dy.isel(i=slice(1, -1)), 0.0)

    # Constant along j, so the j separation vanishes where it is defined.
    dx_j, dy_j = _central_separation_m(lon, lat, "j")
    assert np.allclose(dx_j.isel(j=1), 0.0)
    assert np.allclose(dy_j.isel(j=1), 0.0)


def test_central_separation_wraps_lon_and_uses_the_mid_latitude():
    """A wrapped longitude difference, scaled by the cosine of the pair's mid-latitude."""
    lon = xr.DataArray(np.array([[179.0], [180.0], [-179.0]]), dims=("i", "j"))
    lat = xr.DataArray(np.array([[0.0], [5.0], [40.0]]), dims=("i", "j"))

    dx, dy = _central_separation_m(lon, lat, "i")
    dx_mid = float(dx.isel(i=1, j=0))

    # The pair straddles the antimeridian by 2 degrees, not 358.
    assert dx_mid == pytest.approx(
        2.0 * EARTH_RADIUS_M * np.cos(20.0 * DEG) * DEG, rel=1e-12
    )
    assert abs(dx_mid) < 0.1 * abs(358.0 * EARTH_RADIUS_M * DEG)

    # 20 degrees is the mid-latitude of the differenced pair (0 and 40), not the
    # latitude of the centre point (5).
    assert abs(dx_mid - 2.0 * EARTH_RADIUS_M * np.cos(5.0 * DEG) * DEG) > 1.0e4

    assert float(dy.isel(i=1, j=0)) == pytest.approx(
        40.0 * EARTH_RADIUS_M * DEG, rel=1e-12
    )


def test_arm_separation_subtracts_opposing_arms_onto_the_grid():
    """``_arm_separation_m`` differences two ``displacement`` labels back onto ``(i, j)``."""
    coords = {"displacement": ["east", "north", "west", "south"]}
    lon = xr.DataArray(
        np.array([[[-179.0, 179.5, 179.0, 179.5]]]),
        dims=("i", "j", "displacement"),
        coords=coords,
    )
    lat = xr.DataArray(
        np.array([[[0.0, 1.0, 0.0, -1.0]]]),
        dims=("i", "j", "displacement"),
        coords=coords,
    )

    dx_ew, dy_ew = _arm_separation_m(lon, lat, "east", "west")
    dx_ns, dy_ns = _arm_separation_m(lon, lat, "north", "south")

    assert set(dx_ew.dims) == {"i", "j"}
    assert "displacement" not in dx_ew.coords

    # The east/west arms straddle the antimeridian by 2 degrees on the
    # equator, not 358.
    assert np.allclose(dx_ew, 2.0 * EARTH_RADIUS_M * DEG)
    assert np.allclose(dy_ew, 0.0)
    assert float(abs(dx_ew).max()) < 0.1 * abs(358.0 * EARTH_RADIUS_M * DEG)

    # The north/south arms share a longitude and span 2 degrees of latitude.
    assert np.allclose(dx_ns, 0.0)
    assert np.allclose(dy_ns, 2.0 * EARTH_RADIUS_M * DEG)

    # Antisymmetric in its two arms.
    dx_we, dy_we = _arm_separation_m(lon, lat, "west", "east")
    assert np.allclose(dx_we, -dx_ew)
    assert np.allclose(dy_we, -dy_ew)


def test_arm_separation_east_uses_the_arm_pair_mid_latitude():
    """The east component scales with the cosine of the two arms' mid-latitude."""
    coords = {"displacement": ["east", "west"]}
    lon = xr.DataArray(
        np.array([[[1.0, -1.0]]]), dims=("i", "j", "displacement"), coords=coords
    )
    lat = xr.DataArray(
        np.array([[[40.0, 0.0]]]), dims=("i", "j", "displacement"), coords=coords
    )

    dx, dy = _arm_separation_m(lon, lat, "east", "west")

    assert float(dx.isel(i=0, j=0)) == pytest.approx(
        2.0 * EARTH_RADIUS_M * np.cos(20.0 * DEG) * DEG, rel=1e-12
    )
    assert float(dy.isel(i=0, j=0)) == pytest.approx(
        40.0 * EARTH_RADIUS_M * DEG, rel=1e-12
    )


# --- deformation gradient --------------------------------------------------


def test_deformation_gradient_dims_and_coords(lon_axis, lat_axis):
    """gradF has dims ``(i, j, row, col)`` with ``row``/``col`` valued ['x', 'y'].

    ``row`` and ``col`` are dimension coordinates of size 2 valued
    ``['x', 'y']``; the tensor carries no ``comp`` coord (``comp`` is the
    eigenvector component dim).
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    gradF = g.deformation_gradient()

    assert set(gradF.dims) == {"i", "j", "row", "col"}
    assert gradF.sizes["row"] == 2
    assert gradF.sizes["col"] == 2
    assert list(gradF["row"].values) == ["x", "y"]
    assert list(gradF["col"].values) == ["x", "y"]
    assert "comp" not in gradF.coords

    # The diagnostic is reported at the grid point, so it carries lon_grid/
    # lat_grid, not the per-arm release positions it was differenced from.
    assert set(gradF["lon_grid"].dims) == {"i", "j"}
    assert set(gradF["lat_grid"].dims) == {"i", "j"}
    assert "lon_0" not in gradF.coords
    assert "lat_0" not in gradF.coords


def test_deformation_gradient_matches_local_frame_neighbor(lon_axis, lat_axis):
    """NeighborSeedGrid: gradF matches the analytic gradient at every *interior* point.

    Neighbour differencing has no stencil at the domain edge, so boundary cells
    are NaN. Check the interior to ~1e-6; assert the edges are NaN where their
    stencil step is missing.
    """
    g = advected_flowmap(
        NeighborSeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    gradF = g.deformation_gradient()
    expected = analytic_gradient(M, flowmap=g, origin=seed_origin(g))

    interior = gradF.isel(i=slice(1, -1), j=slice(1, -1))
    assert bool(interior.notnull().all())
    expected_interior = expected.isel(i=slice(1, -1), j=slice(1, -1))
    assert float(abs(interior - expected_interior).max()) < 1e-6

    # The expectation is not degenerate. It varies over the grid, and it is far
    # from the constant M the advection was built from.
    spread = expected.max(("i", "j")) - expected.min(("i", "j"))
    assert float(spread.max()) > 0.01
    assert float(abs(expected - M_TENSOR).max()) > 0.01

    # The i-derivative (col='x') is undefined on the i edges; likewise the
    # j-derivative (col='y') on the j edges.
    assert bool(gradF.isel(i=0).sel(col="x").isnull().all())
    assert bool(gradF.isel(i=-1).sel(col="x").isnull().all())
    assert bool(gradF.isel(j=0).sel(col="y").isnull().all())
    assert bool(gradF.isel(j=-1).sel(col="y").isnull().all())


def test_deformation_gradient_matches_local_frame_auxiliary(lon_axis, lat_axis):
    """AuxiliarySeedGrid: gradF matches the analytic gradient everywhere, edges included.

    The per-point auxiliary stencil makes the gradient well-defined at every grid
    point, leaving no NaN edges to exclude.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    gradF = g.deformation_gradient()
    expected = analytic_gradient(M, flowmap=g, origin=seed_origin(g))

    assert bool(gradF.notnull().all())
    assert float(abs(gradF - expected).max()) < 1e-6

    # The expectation is not degenerate. It varies over the grid and differs
    # from the constant M by far more than the tolerance above.
    spread = expected.max(("i", "j")) - expected.min(("i", "j"))
    assert float(spread.max()) > 0.01
    assert float(abs(expected - M_TENSOR).max()) > 0.01


def test_deformation_gradient_varying_jacobian_auxiliary(lon_axis, lat_axis):
    """AuxiliarySeedGrid: gradF equals a spatially varying analytic Jacobian.

    The map is quadratic in the centroid tangent frame,
    ``f(dx, dy) = (dx + a*dx**2, dy + b*dy**2)``. Its Jacobian there is
    ``diag(1 + 2*a*X, 1 + 2*b*Y)``, with ``(X, Y)`` each grid point's meters
    position from the centroid. Central differencing is exact for a quadratic, so
    gradF must match that Jacobian, rescaled into the local frames, to ~1e-6.
    The comparison exercises per-point differencing, not the constant-``M`` case.
    """
    a, b = 1.0e-6, -0.8e-6

    def f(dx, dy):
        return dx + a * dx**2, dy + b * dy**2

    g = advected_flowmap_f(
        AuxiliarySeedGrid, lon_axis, lat_axis, f, RELEASE_TIME, END_TIME
    )
    gradF = g.deformation_gradient()

    release = ["lon_0", "lat_0"]
    lon_grid = g.ds["lon_grid"].drop_vars(release, errors="ignore")
    lat_grid = g.ds["lat_grid"].drop_vars(release, errors="ignore")
    lon_0, lat_0 = seed_origin(g)

    # Grid-point positions in the centroid tangent frame, dims (i, j). That frame
    # uses the origin's cosine throughout, so each call here holds one coordinate
    # fixed. A pair sharing lat_0 has mid-latitude lat_0, and a pair sharing
    # lon_0 has no east component to scale.
    X, _ = _separation_m(lon_a=lon_0, lat_a=lat_0, lon_b=lon_grid, lat_b=lat_0)
    _, Y = _separation_m(lon_a=lon_0, lat_a=lat_0, lon_b=lon_0, lat_b=lat_grid)

    fxx = 1 + 2 * a * X
    fyy = 1 + 2 * b * Y
    zero = xr.zeros_like(X)
    row_x = xr.concat([fxx, zero], dim="col")
    row_y = xr.concat([zero, fyy], dim="col")
    flat_jacobian = xr.concat([row_x, row_y], dim="row")

    _, lat_advected = apply_map_to_lonlat(
        lon_grid.values, lat_grid.values, f, (lon_0, lat_0)
    )
    expected = local_frame_gradient(
        flat_jacobian,
        lat_grid=lat_grid,
        lat_advected=lat_grid.copy(data=lat_advected),
        lat_origin=lat_0,
    )
    assert float(abs(gradF - expected).max()) < 1e-6

    # Sanity: the Jacobian varies across the grid (not the constant-M
    # case), so this test exercises per-point differencing.
    assert float(fxx.max() - fxx.min()) > 0.1


# --- Cauchy-Green ----------------------------------------------------------


def test_cauchy_green_symmetry(lon_axis, lat_axis):
    """C is symmetric: ``C == C`` transposed over ``(row, col)``.

    Compare ``C`` with its ``(row, col)`` transpose via
    ``xr.testing.assert_allclose``. True for any gradF.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    C = g.cauchy_green()

    C_transposed = C.rename({"row": "col", "col": "row"}).transpose(*C.dims)
    xr.testing.assert_allclose(C, C_transposed)


def test_cauchy_green_equals_gradF_T_gradF(lon_axis, lat_axis):
    """C equals ``gradF^T gradF`` built from the analytic gradient.

    Use AuxiliarySeedGrid to avoid NaN edges; check to ~1e-6. The constant ``M.T @ M``
    is not the answer, so assert it misses by far more than that tolerance.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    C = g.cauchy_green()

    expected = cauchy_green_of(analytic_gradient(M, flowmap=g, origin=seed_origin(g)))
    assert float(abs(C - expected).max()) < 1e-6

    constant = xr.DataArray(
        M.T @ M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
    )
    assert float(abs(expected - constant).max()) > 0.01


# --- eigen-analysis --------------------------------------------------------


def test_cg_eigen_shapes_and_order(lon_axis, lat_axis):
    """cg_eigen returns ``lambda`` (i,j,eig) and ``xi`` (i,j,comp,eig), ascending.

    Assert dims/sizes (``eig`` and ``comp`` size 2) and that eigenvalues are
    ascending along ``eig`` (``lambda.isel(eig=1) >= lambda.isel(eig=0)``).
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    eigen = g.cg_eigen()
    lam = eigen["lambda"]
    xi = eigen["xi"]

    assert set(lam.dims) == {"i", "j", "eig"}
    assert set(xi.dims) == {"i", "j", "comp", "eig"}
    assert lam.sizes["eig"] == 2
    assert xi.sizes["eig"] == 2
    assert xi.sizes["comp"] == 2
    assert bool((lam.isel(eig=1) >= lam.isel(eig=0)).all())


def test_cg_eigen_relation(lon_axis, lat_axis):
    """The eigenpairs satisfy ``C @ xi == lambda * xi`` with orthonormal ``xi``.

    Contract ``C``'s ``col`` against ``xi``'s ``comp`` with ``xr.dot``, relabel
    the surviving axis back to ``comp``, and compare to ``lambda * xi``. The
    eigen-relation is scale-invariant, so pin normalization separately. The Gram
    matrix ``xi^T xi`` must be the identity over ``eig`` (unit-norm, mutually
    orthogonal).
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    C = g.cauchy_green()
    eigen = g.cg_eigen()
    lam = eigen["lambda"]
    xi = eigen["xi"]

    # Sum C_{row,col} xi_{comp=col} over the shared component dim; the output
    # component is the surviving 'row' axis, relabelled back to 'comp'.
    Cxi = xr.dot(C.rename({"col": "comp"}), xi, dim="comp").rename({"row": "comp"})
    rhs = lam * xi
    order = ("i", "j", "comp", "eig")
    xr.testing.assert_allclose(Cxi.transpose(*order), rhs.transpose(*order))

    # Orthonormality: xi_{comp,a} xi_{comp,b} summed over comp == delta_{a,b}.
    gram = xr.dot(xi, xi.rename(eig="eig_b"), dim="comp")
    identity = xr.DataArray(
        np.eye(2), dims=("eig", "eig_b"), coords={"eig": [0, 1], "eig_b": [0, 1]}
    )
    assert float(abs(gram - identity).max()) < 1e-6


def test_cg_eigen_values_match_analytic(lon_axis, lat_axis):
    """Eigenvalues equal those of the analytic ``gradF^T gradF``, per grid point.

    Compare ``lambda`` (ascending) against ``numpy.linalg.eigvalsh`` of the
    analytic Cauchy-Green tensor to ~1e-6.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    lam = g.cg_eigen()["lambda"]

    expected = analytic_eigenvalues(
        analytic_gradient(M, flowmap=g, origin=seed_origin(g))
    )
    assert float(abs(lam - expected).max()) < 1e-6

    # The eigenvalues vary over the grid, so this is not the constant-M answer.
    constant = np.linalg.eigvalsh(M.T @ M)
    assert float(abs(expected - xr.DataArray(constant, dims="eig")).max()) > 0.01


# --- FTLE ------------------------------------------------------------------


def test_ftle_pure_stretch_follows_the_local_frame_gradient(lon_axis, lat_axis):
    """Pure stretch ``M = diag(a, b)`` gives an FTLE that varies over the grid.

    The local east/north rescaling makes the x-stretch latitude-dependent, so the
    expectation is ``(1 / |T|) * log(sqrt(lambda_max))`` of the *analytic*
    gradient's own Cauchy-Green tensor, not ``log(max(a, b)) / |T|``. ``a > b``
    puts the rescaled x-stretch in charge of ``lambda_max``, so the variation
    reaches the FTLE. Use AuxiliarySeedGrid so the field is NaN-free.
    """
    a, b = 3.0, 2.0
    M_stretch = np.array([[a, 0.0], [0.0, b]])
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M_stretch, RELEASE_TIME, END_TIME
    )
    ftle = g.ftle()

    lam_max = analytic_eigenvalues(
        analytic_gradient(M_stretch, flowmap=g, origin=seed_origin(g))
    ).isel(eig=1, drop=True)
    expected = (1.0 / T_SEC) * np.log(np.sqrt(lam_max))

    assert set(ftle.dims) == {"i", "j"}
    assert "eig" not in ftle.coords  # the eigenvalue pick leaves no scalar coord
    # The FTLE is O(1e-5) here, so 1e-6 would pass on a constant field; hold the
    # match to round-off instead.
    assert float(abs(ftle - expected).max()) < 1e-12

    # The field is not degenerate. It varies by far more than the tolerance
    # above, so this is not the old constant log(max(a, b)) / |T| answer.
    assert float(expected.max() - expected.min()) > 1e-9


def test_ftle_matches_eigen(lon_axis, lat_axis):
    """ftle is consistent with cg_eigen's largest eigenvalue.

    For a general ``M``, ``ftle == (1 / |T|) * log(sqrt(lambda.isel(eig=1)))``.
    Use AuxiliarySeedGrid so both fields are NaN-free.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    ftle = g.ftle()

    # ftle() reports on (i, j) with no leftover `eig` coord (drop=True on the
    # eigenvalue pick), so drop it here too.
    lam_max = g.cg_eigen()["lambda"].isel(eig=1, drop=True)
    expected = (1.0 / T_SEC) * np.log(np.sqrt(lam_max))
    xr.testing.assert_allclose(ftle, expected)


def test_ftle_backward_equals_forward(lon_axis, lat_axis):
    """Backward integration (``t1 < t0``, negative ``T``) gives the same FTLE.

    The FTLE divides by ``|T|``, so the sign of ``T`` (attracting vs. repelling)
    must not change its value. Ingesting the same advected positions with the
    bounds swapped (``t0=END_TIME``, ``t1=RELEASE_TIME`` -> ``T < 0``) must reproduce the forward
    field exactly.
    """
    g_fwd = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    g_bwd = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, END_TIME, RELEASE_TIME
    )

    # Sanity: the stored windows are equal and opposite.
    assert g_fwd.ds["T"] == -g_bwd.ds["T"]
    assert g_bwd.ds["T"] < np.timedelta64(0, "s")
    # The FTLE fields must match value-for-value. (Subtraction drops the
    # conflicting scalar t0/T coords, leaving a difference indexed by (i, j) alone.)
    assert float(abs(g_fwd.ftle() - g_bwd.ftle()).max()) < 1e-12


# --- NaN propagation -------------------------------------------------------


def test_nan_propagates_through_chain(lon_axis, lat_axis):
    """A lost particle (NaN) flows ``gradF -> C -> eigen -> ftle`` and isolates.

    Knock out a single arm of one ``AuxiliarySeedGrid`` cell so its stencil is
    incomplete. That one cell's FTLE must be NaN while every other cell stays
    finite, so the NaN propagates the whole chain and does not leak to neighbours.
    Also guards that ``np.linalg.eigh`` returns NaN rather than raising
    ``LinAlgError`` on a NaN sub-matrix.
    """
    g = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    # Drop the 'south' arm of cell (i=0, j=1) -> NaN advected position there.
    bad = (g.ds["i"] == 0) & (g.ds["j"] == 1) & (g.ds["displacement"] == "south")
    g.ds["lon"] = g.ds["lon"].where(~bad)

    ftle = g.ftle()
    # Exactly one cell is NaN, and it is the corrupted one. The NaN propagated
    # the whole chain and did not leak to any neighbour.
    assert bool(ftle.sel(i=0, j=1).isnull())
    assert int(ftle.isnull().sum()) == 1


# --- the same stencil on an unstructured layout ----------------------------


def test_auxiliary_diagnostics_do_not_read_the_layout(lon_axis, lat_axis):
    """The same points give the same gradient laid out as axes or as a set.

    The four-arm stencil is differenced against a grid point's own arms and never
    against another grid point, so nothing between ``deformation_gradient`` and
    ``ftle`` has a layout to read. The match is measured rather than asserted.
    The flat results are the structured ones ravelled, exactly.
    """
    structured = advected_flowmap(
        AuxiliarySeedGrid, lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )
    scattered = advected_scattered_flowmap(
        lon_axis, lat_axis, M, RELEASE_TIME, END_TIME
    )

    for name, flat in (
        ("deformation_gradient", scattered.deformation_gradient()),
        ("cauchy_green", scattered.cauchy_green()),
    ):
        grid = getattr(structured, name)().transpose("i", "j", "row", "col")
        np.testing.assert_array_equal(
            grid.values.reshape(-1, 2, 2), flat.transpose("grid_point", "row", "col")
        )
    np.testing.assert_array_equal(
        structured.ftle().transpose("i", "j").values.ravel(), scattered.ftle()
    )


def test_unstructured_gradient_matches_the_local_frame(lon_axis, lat_axis):
    """grad F on a point set is the analytic local-frame answer, not just the
    structured one repeated. Both paths could share a mistake."""
    fm = advected_scattered_flowmap(lon_axis, lat_axis, M, RELEASE_TIME, END_TIME)

    expected = analytic_gradient(M, flowmap=fm, origin=seed_origin(fm))
    gradF = fm.deformation_gradient().transpose(*expected.dims)

    assert float(np.abs(gradF - expected).max()) < 1e-9
