"""Operator tests: gradF -> C -> eigen -> FTLE (diagnostics on a FlowMap).

For a constant linear flow map ``F(x) = M @ x`` in the local meters frame,
``gradF`` equals ``M`` at every grid point, so the whole chain has closed-form
answers. The ``conftest.advected_flowmap`` helper seeds a time-free grid, emits
its particle set, advects through ``M`` about the seed centroid, and ingests via
``seed.pset_to_flowmap(..., t0, t1)`` (the signed window ``T = t1 - t0`` lands on
the ``FlowMap``).

A non-symmetric ``M = [[2.0, 0.5], [0.0, 3.0]]`` is used for the general tests so
that ``C = M^T M`` is a non-trivial check.
"""

import numpy as np
import xarray as xr

from conftest import advected_flowmap, advected_flowmap_f
from lcs_parcels import AuxiliarySeed, NeighborSeed
from lcs_parcels.grids import _lonlat_to_meters

# Release time and integration end time; the signed window T = T1 - T0 spans one
# day (|T| = 86400 s).
T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")

# Non-symmetric linear flow map; C = M^T M is then a non-trivial check.
M = np.array([[2.0, 0.5], [0.0, 3.0]])

# The same map as a (row, col) tensor, for label-based broadcasting against
# gradF (gradF.sel(row=a, col=b) = dF_a/dx0_b = M[a, b]).
M_TENSOR = xr.DataArray(
    M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
)

# Integration window in seconds (one day) used by the analytic FTLE.
T_SEC = abs((T1 - T0) / np.timedelta64(1, "s"))


# --- deformation gradient --------------------------------------------------


def test_deformation_gradient_dims_and_coords(lon_axis, lat_axis):
    """gradF has dims ``(i, j, row, col)`` with ``row``/``col`` valued ['x', 'y'].

    ``row`` and ``col`` are dimension coordinates of size 2 valued
    ``['x', 'y']``; the tensor carries no ``comp`` coord (``comp`` is the
    eigenvector component dim).
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    gradF = g.deformation_gradient()

    assert set(gradF.dims) == {"i", "j", "row", "col"}
    assert gradF.sizes["row"] == 2
    assert gradF.sizes["col"] == 2
    assert list(gradF["row"].values) == ["x", "y"]
    assert list(gradF["col"].values) == ["x", "y"]
    assert "comp" not in gradF.coords


def test_deformation_gradient_equals_M_neighbor(lon_axis, lat_axis):
    """NeighborSeed: gradF == M at every *interior* grid point.

    Neighbour differencing has no stencil at the domain edge, so boundary cells
    are NaN. Check the interior against ``M`` to ~1e-6; assert the edges are NaN
    where their stencil step is missing.
    """
    g = advected_flowmap(NeighborSeed, lon_axis, lat_axis, M, T0, T1)
    gradF = g.deformation_gradient()

    interior = gradF.isel(i=slice(1, -1), j=slice(1, -1))
    assert bool(interior.notnull().all())
    # gradF.sel(row=a, col=b) == M[a, b], broadcast over (i, j) by label.
    assert float(abs(interior - M_TENSOR).max()) < 1e-6

    # The i-derivative (col='x') is undefined on the i edges; likewise the
    # j-derivative (col='y') on the j edges.
    assert bool(gradF.isel(i=0).sel(col="x").isnull().all())
    assert bool(gradF.isel(i=-1).sel(col="x").isnull().all())
    assert bool(gradF.isel(j=0).sel(col="y").isnull().all())
    assert bool(gradF.isel(j=-1).sel(col="y").isnull().all())


def test_deformation_gradient_equals_M_auxiliary(lon_axis, lat_axis):
    """AuxiliarySeed: gradF == M at *every* grid point, including the boundary.

    The per-point auxiliary stencil makes the gradient well-defined everywhere,
    so there are no NaN edges to exclude. Check each component against ``M``.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    gradF = g.deformation_gradient()

    assert bool(gradF.notnull().all())
    assert float(abs(gradF - M_TENSOR).max()) < 1e-6


def test_deformation_gradient_varying_jacobian_auxiliary(lon_axis, lat_axis):
    """AuxiliarySeed: gradF equals a spatially-VARYING analytic Jacobian.

    The map is quadratic in the meters frame,
    ``f(dx, dy) = (dx + a*dx**2, dy + b*dy**2)``, whose exact Jacobian is
    ``diag(1 + 2*a*X, 1 + 2*b*Y)`` with ``(X, Y)`` each grid centre's meters
    position from the centroid. Central differencing is exact for a quadratic, so
    gradF must match the analytic per-point Jacobian to ~1e-6 -- exercising
    per-point differencing, not the constant-``M`` case.
    """
    a, b = 1.0e-6, -0.8e-6

    def f(dx, dy):
        return dx + a * dx**2, dy + b * dy**2

    g = advected_flowmap_f(AuxiliarySeed, lon_axis, lat_axis, f, T0, T1)
    gradF = g.deformation_gradient()

    lon_c, lat_c = g.ds["lon_c"], g.ds["lat_c"]
    lon0 = float(g.ds["lon_0"].mean())
    lat0 = float(g.ds["lat_0"].mean())
    X, Y = _lonlat_to_meters(lon_c, lat_c, lon0, lat0)  # centre meters, dims (i, j)
    fxx = 1 + 2 * a * X
    fyy = 1 + 2 * b * Y
    zero = xr.zeros_like(X)
    row_x = xr.concat([fxx, zero], dim="col")
    row_y = xr.concat([zero, fyy], dim="col")
    expected = xr.concat([row_x, row_y], dim="row").assign_coords(
        row=["x", "y"], col=["x", "y"]
    )
    assert float(abs(gradF - expected).max()) < 1e-6

    # sanity: the Jacobian genuinely VARIES across the grid (not the constant-M
    # case), so this test exercises per-point differencing.
    assert float(fxx.max() - fxx.min()) > 0.1


# --- Cauchy-Green ----------------------------------------------------------


def test_cauchy_green_symmetry(lon_axis, lat_axis):
    """C is symmetric: ``C == C`` transposed over ``(row, col)``.

    Compare ``C`` with its ``(row, col)`` transpose via
    ``xr.testing.assert_allclose``. True for any gradF.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    C = g.cauchy_green()

    C_transposed = C.rename({"row": "col", "col": "row"}).transpose(*C.dims)
    xr.testing.assert_allclose(C, C_transposed)


def test_cauchy_green_equals_MT_M(lon_axis, lat_axis):
    """C == M^T M for the linear flow map.

    With ``gradF == M`` everywhere, ``C = (grad F)^T grad F`` equals the constant
    ``M.T @ M`` at every grid point. Use AuxiliarySeed to avoid NaN edges; check
    to ~1e-6.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    C = g.cauchy_green()

    expected = xr.DataArray(
        M.T @ M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
    )
    assert float(abs(C - expected).max()) < 1e-6


# --- eigen-analysis --------------------------------------------------------


def test_cg_eigen_shapes_and_order(lon_axis, lat_axis):
    """cg_eigen returns ``lambda`` (i,j,eig) and ``xi`` (i,j,comp,eig), ascending.

    Assert dims/sizes (``eig`` and ``comp`` size 2) and that eigenvalues are
    ascending along ``eig`` (``lambda.isel(eig=1) >= lambda.isel(eig=0)``).
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
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
    eigen-relation is scale-invariant, so pin normalization separately: the Gram
    matrix ``xi^T xi`` must be the identity over ``eig`` (unit-norm, mutually
    orthogonal).
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
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
    """Eigenvalues equal ``eigvalsh(M^T M)`` for the linear map.

    Compare ``lambda`` (ascending) against ``numpy.linalg.eigvalsh(M.T @ M)`` to
    ~1e-6.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    lam = g.cg_eigen()["lambda"]

    expected = xr.DataArray(
        np.linalg.eigvalsh(M.T @ M), dims="eig", coords={"eig": [0, 1]}
    )
    assert float(abs(lam - expected).max()) < 1e-6


# --- FTLE ------------------------------------------------------------------


def test_ftle_pure_stretch(lon_axis, lat_axis):
    """Pure stretch ``M = diag(a, b)`` gives a constant analytic FTLE.

    Then ``lambda_max = max(a, b)**2`` and
    ``ftle == (1 / |T|) * log(max(a, b))``. Use AuxiliarySeed so the field is
    NaN-free; assert dims ``(i, j)`` and the constant value.
    """
    a, b = 2.0, 3.0
    M_stretch = np.array([[a, 0.0], [0.0, b]])
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M_stretch, T0, T1)
    ftle = g.ftle()

    expected = (1.0 / T_SEC) * np.log(max(a, b))
    assert set(ftle.dims) == {"i", "j"}
    assert "eig" not in ftle.coords  # the eigenvalue pick leaves no scalar coord
    assert float(abs(ftle - expected).max()) < 1e-6


def test_ftle_matches_eigen(lon_axis, lat_axis):
    """ftle is consistent with cg_eigen's largest eigenvalue.

    For a general ``M``, ``ftle == (1 / |T|) * log(sqrt(lambda.isel(eig=1)))``.
    Use AuxiliarySeed so both fields are NaN-free.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
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
    bounds swapped (``t0=T1``, ``t1=T0`` -> ``T < 0``) must reproduce the forward
    field exactly.
    """
    g_fwd = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    g_bwd = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T1, T0)

    # Sanity: the stored windows are equal and opposite.
    assert g_fwd.ds["T"] == -g_bwd.ds["T"]
    assert g_bwd.ds["T"] < np.timedelta64(0, "s")
    # The FTLE fields must match value-for-value. (Subtraction drops the
    # conflicting scalar t0/T coords, leaving a clean (i, j) difference.)
    assert float(abs(g_fwd.ftle() - g_bwd.ftle()).max()) < 1e-12


# --- NaN propagation -------------------------------------------------------


def test_nan_propagates_through_chain(lon_axis, lat_axis):
    """A lost particle (NaN) flows ``gradF -> C -> eigen -> ftle`` and isolates.

    Knock out a single arm of one ``AuxiliarySeed`` cell so its stencil is
    incomplete. That one cell's FTLE must be NaN while every other cell stays
    finite -- the NaN propagates the whole chain and does not leak to neighbours.
    Also guards that ``np.linalg.eigh`` returns NaN rather than raising
    ``LinAlgError`` on a NaN sub-matrix.
    """
    g = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    # Drop the 'south' arm of cell (i=0, j=1) -> NaN advected position there.
    bad = (g.ds["i"] == 0) & (g.ds["j"] == 1) & (g.ds["displacement"] == "south")
    g.ds["lon"] = g.ds["lon"].where(~bad)

    ftle = g.ftle()
    # Exactly one cell is NaN, and it is the corrupted one: the NaN propagated
    # the whole chain and did not leak to any neighbour.
    assert bool(ftle.sel(i=0, j=1).isnull())
    assert int(ftle.isnull().sum()) == 1
