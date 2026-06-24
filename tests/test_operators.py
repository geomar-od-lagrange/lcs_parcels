"""Operators on a known linear flow map: gradF -> C -> eigen -> FTLE.

For a constant linear flow map ``F(x) = M @ x`` (in the local meters tangent
frame), the deformation gradient ``gradF`` equals ``M`` at every grid point.
We seed a grid, emit its particle set, advect it through such an ``M`` with the
conftest helper, ingest the result, and assert the analytic answers.
"""

import numpy as np
import pytest
import xarray as xr

from lcs_parcels import AuxiliaryGrid, NeighborGrid

from conftest import apply_linear_map_to_pset


# A non-symmetric constant map so that C = M^T M is a non-trivial check.
M = np.array([[2.0, 0.5], [0.0, 3.0]])
T = 86400.0  # 1 day, seconds


def _advected_grid(cls, lon_axis, lat_axis, M):
    seed = cls.from_axes(lon_axis, lat_axis)
    lon, lat = seed.to_parcels_pset()
    origin = (float(lon_axis.mean()), float(lat_axis.mean()))
    lon_out, lat_out = apply_linear_map_to_pset(lon, lat, M, origin)
    return cls.from_parcels_pset_lon_lat(seed, lon_out, lat_out, T=T)


# --- deformation gradient --------------------------------------------------


def test_deformation_gradient_dims_and_coords(lon_axis, lat_axis):
    g = _advected_grid(NeighborGrid, lon_axis, lat_axis, M)
    gradF = g.deformation_gradient()

    assert set(gradF.dims) == {"i", "j", "row", "col"}
    assert gradF.sizes["row"] == 2
    assert gradF.sizes["col"] == 2
    # component label coord on both row and col (or shared comp coord)
    assert list(gradF["comp"].values) == ["x", "y"]


def test_deformation_gradient_equals_M(lon_axis, lat_axis):
    g = _advected_grid(NeighborGrid, lon_axis, lat_axis, M)
    gradF = g.deformation_gradient()

    # gradF == M at every interior grid point (boundary may be NaN for
    # neighbor differencing; check the interior with label-based selection).
    interior = gradF.isel(
        i=slice(1, -1), j=slice(1, -1)
    )
    for r in range(2):
        for c in range(2):
            vals = interior.isel(row=r, col=c).values
            np.testing.assert_allclose(vals, M[r, c], atol=1e-6)


def test_auxiliary_deformation_gradient_equals_M(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    gradF = g.deformation_gradient()

    # AuxiliaryGrid differences within each per-point stencil, so the gradient
    # is well-defined at every (i, j) including the boundary.
    for r in range(2):
        for c in range(2):
            vals = gradF.isel(row=r, col=c).values
            np.testing.assert_allclose(vals, M[r, c], atol=1e-6)


# --- Cauchy-Green ----------------------------------------------------------


def test_cauchy_green_symmetry(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    C = g.cauchy_green()

    assert set(C.dims) == {"i", "j", "row", "col"}
    # C is symmetric: C == C transposed over (row, col).
    C_swapped = C.rename({"row": "col", "col": "row"})
    xr.testing.assert_allclose(C, C_swapped)


def test_cauchy_green_equals_MT_M(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    C = g.cauchy_green()

    expected = M.T @ M
    for r in range(2):
        for c in range(2):
            vals = C.isel(row=r, col=c).values
            np.testing.assert_allclose(vals, expected[r, c], atol=1e-6)


# --- eigen-analysis --------------------------------------------------------


def test_cg_eigen_shapes_and_order(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    eig = g.cg_eigen()

    lam = eig["lambda"]
    xi = eig["xi"]
    assert set(lam.dims) == {"i", "j", "eig"}
    assert set(xi.dims) == {"i", "j", "comp", "eig"}
    assert lam.sizes["eig"] == 2
    assert xi.sizes["comp"] == 2

    # eigenvalues sorted ascending along eig
    lo = lam.isel(eig=0)
    hi = lam.isel(eig=1)
    assert bool((hi >= lo).all())


def test_cg_eigen_relation(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    C = g.cauchy_green()
    eig = g.cg_eigen()
    lam = eig["lambda"]
    xi = eig["xi"]

    # C @ xi == lambda * xi for each eigenpair. Use label-based matmul over the
    # component dims: contract C's col with xi's comp.
    Cxi = xr.dot(
        C.rename({"col": "comp"}),
        xi,
        dims=["comp"],
    ).rename({"row": "comp"})
    expected = lam * xi
    xr.testing.assert_allclose(Cxi, expected, atol=1e-6)


def test_cg_eigen_values_match_analytic(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    eig = g.cg_eigen()
    lam = eig["lambda"]

    analytic = np.sort(np.linalg.eigvalsh(M.T @ M))
    np.testing.assert_allclose(lam.isel(eig=0).values, analytic[0], atol=1e-6)
    np.testing.assert_allclose(lam.isel(eig=1).values, analytic[1], atol=1e-6)


# --- FTLE ------------------------------------------------------------------


def test_ftle_pure_stretch(lon_axis, lat_axis):
    a, b = 4.0, 2.0
    M_stretch = np.diag([a, b])
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M_stretch)

    ftle = g.ftle()
    assert set(ftle.dims) == {"i", "j"}

    # lambda_max = max(a, b)**2 -> ftle = (1/|T|) log sqrt(lambda_max)
    lambda_max = max(a, b) ** 2
    expected = (1.0 / abs(T)) * np.log(np.sqrt(lambda_max))
    np.testing.assert_allclose(ftle.values, expected, atol=1e-9)


def test_ftle_matches_eigen(lon_axis, lat_axis):
    g = _advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M)
    ftle = g.ftle()
    lam = g.cg_eigen()["lambda"]

    lambda_max = lam.isel(eig=1)
    expected = (1.0 / abs(T)) * np.log(np.sqrt(lambda_max))
    xr.testing.assert_allclose(ftle, expected, atol=1e-9)
