"""Operator tests: gradF -> C -> eigen -> FTLE.

For a constant linear flow map ``F(x) = M @ x`` in the local meters tangent
frame, the deformation gradient ``gradF`` equals ``M`` at every grid point, so
the whole chain has closed-form answers:

- seed a grid via ``cls.from_axes(lon_axis, lat_axis, t0=...)``, where ``t0`` is
  a ``datetime64`` release time recorded on the grid;
- emit its particle set with ``to_parcels_pset()``;
- advect the flat positions through ``M`` about the grid centroid (the
  ``conftest.advected_grid`` helper does this in the meters frame, anchored at
  the same single reference point the implementation uses);
- ingest with ``cls.from_parcels_pset_lon_lat(seed, lon_out, lat_out, t1=...)``,
  where ``t1`` is the ``datetime64`` integration end time; the signed window
  ``T = t1 - t0`` is derived from the seed's ``t0`` (see
  ``plans/timing-design.md``).

A **non-symmetric** ``M = [[2.0, 0.5], [0.0, 3.0]]`` is used for the general
tests so that ``C = M^T M`` is a non-trivial check.  Only the high-level,
label-based xarray API is used in assertions (``.isel`` / ``.sel`` / named dims
/ broadcasting / ``xr.dot`` / ``xr.testing.assert_allclose``), never positional
indexing.
"""

import numpy as np
import xarray as xr

from conftest import advected_grid
from lcs_parcels import AuxiliaryGrid, NeighborGrid

# Release time recorded on the seed and integration end time supplied at ingest;
# the signed window T = T1 - T0 spans one day (|T| = 86400 s). See
# plans/timing-design.md.
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

    Assert the tensor layout contract: ``row`` and ``col`` are dimension
    coordinates of size 2 valued ``['x', 'y']``. This pins the storage
    convention the rest of the operators rely on; ``comp`` is reserved for the
    eigenvector component dim, so the tensor itself carries no ``comp`` coord.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    gradF = g.deformation_gradient()

    assert set(gradF.dims) == {"i", "j", "row", "col"}
    assert gradF.sizes["row"] == 2
    assert gradF.sizes["col"] == 2
    assert list(gradF["row"].values) == ["x", "y"]
    assert list(gradF["col"].values) == ["x", "y"]
    assert "comp" not in gradF.coords


def test_deformation_gradient_equals_M_neighbor(lon_axis, lat_axis):
    """NeighborGrid: gradF == M at every *interior* grid point.

    Neighbour differencing has no stencil at the domain edge, so boundary cells
    are legitimately NaN. Select the interior (``isel(i=slice(1, -1),
    j=slice(1, -1))``) and check each component against ``M`` to ~1e-6; the
    edges carry NaN where their stencil step is missing.
    """
    g = advected_grid(NeighborGrid, lon_axis, lat_axis, M, T0, T1)
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
    """AuxiliaryGrid: gradF == M at *every* grid point, including the boundary.

    The per-point auxiliary stencil makes the gradient well-defined everywhere,
    so there are no NaN edges to exclude. Check each component against ``M``.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    gradF = g.deformation_gradient()

    assert bool(gradF.notnull().all())
    assert float(abs(gradF - M_TENSOR).max()) < 1e-6


# --- Cauchy-Green ----------------------------------------------------------


def test_cauchy_green_symmetry(lon_axis, lat_axis):
    """C is symmetric: ``C == C`` transposed over ``(row, col)``.

    Compare ``C`` with its ``(row, col)`` transpose (rename the two axes, then
    restore the dim order) via ``xr.testing.assert_allclose``. True for any
    gradF, so it does not need the analytic value of ``M``.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    C = g.cauchy_green()

    C_transposed = C.rename({"row": "col", "col": "row"}).transpose(*C.dims)
    xr.testing.assert_allclose(C, C_transposed)


def test_cauchy_green_equals_MT_M(lon_axis, lat_axis):
    """C == M^T M for the linear flow map.

    With ``gradF == M`` everywhere, ``C = (grad F)^T grad F`` must equal the
    constant ``M.T @ M`` at every grid point. Use AuxiliaryGrid to avoid NaN
    edges. Check each component to ~1e-6.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    C = g.cauchy_green()

    expected = xr.DataArray(
        M.T @ M, dims=("row", "col"), coords={"row": ["x", "y"], "col": ["x", "y"]}
    )
    assert float(abs(C - expected).max()) < 1e-6


# --- eigen-analysis --------------------------------------------------------


def test_cg_eigen_shapes_and_order(lon_axis, lat_axis):
    """cg_eigen returns ``lambda`` (i,j,eig) and ``xi`` (i,j,comp,eig), ascending.

    Assert dims/sizes (``eig`` size 2, ``comp`` size 2) and that eigenvalues are
    sorted ascending along ``eig`` (``lambda.isel(eig=1) >= lambda.isel(eig=0)``
    everywhere).
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
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

    Contract ``C``'s ``col`` against ``xi``'s ``comp`` with ``xr.dot`` over the
    shared component dim, relabel the surviving output axis back to ``comp``, and
    compare to ``lambda * xi``. This validates eigenvectors independently of any
    analytic value.

    The eigen-relation ``C xi = lambda xi`` is scale-invariant (it holds for any
    multiple of an eigenvector), so it alone does *not* pin normalization. Pin it
    separately: the Gram matrix ``xi^T xi`` must be the identity over ``eig`` --
    eigenvectors are unit-norm and mutually orthogonal -- which a closed-form 2x2
    solver that forgot to normalize would otherwise fail.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
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

    Compare ``lambda`` (ascending) against ``numpy.linalg.eigvalsh(M.T @ M)`` at
    every grid point to ~1e-6.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    lam = g.cg_eigen()["lambda"]

    expected = xr.DataArray(
        np.linalg.eigvalsh(M.T @ M), dims="eig", coords={"eig": [0, 1]}
    )
    assert float(abs(lam - expected).max()) < 1e-6


# --- FTLE ------------------------------------------------------------------


def test_ftle_pure_stretch(lon_axis, lat_axis):
    """Pure stretch ``M = diag(a, b)`` gives a constant analytic FTLE.

    Then ``lambda_max = max(a, b)**2`` and
    ``ftle == (1 / |T|) * log(sqrt(lambda_max)) == (1 / |T|) * log(max(a, b))``
    at every grid point. Use AuxiliaryGrid so the field is NaN-free, and assert
    dims ``(i, j)`` and the constant value.
    """
    a, b = 2.0, 3.0
    M_stretch = np.array([[a, 0.0], [0.0, b]])
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M_stretch, T0, T1)
    ftle = g.ftle()

    expected = (1.0 / T_SEC) * np.log(max(a, b))
    assert set(ftle.dims) == {"i", "j"}
    assert float(abs(ftle - expected).max()) < 1e-6


def test_ftle_matches_eigen(lon_axis, lat_axis):
    """ftle is consistent with cg_eigen's largest eigenvalue.

    For a general ``M``, ``ftle == (1 / |T|) * log(sqrt(lambda.isel(eig=1)))``.
    Cross-checks the FTLE definition against the eigen step using the *largest*
    eigenvalue. Use AuxiliaryGrid so both fields are NaN-free.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    ftle = g.ftle()

    lam_max = g.cg_eigen()["lambda"].isel(eig=1)
    expected = (1.0 / T_SEC) * np.log(np.sqrt(lam_max))
    xr.testing.assert_allclose(ftle, expected)


def test_ftle_backward_equals_forward(lon_axis, lat_axis):
    """Backward integration (``t1 < t0``, negative ``T``) gives the same FTLE.

    The FTLE divides by ``|T|``, so only the magnitude of the signed window
    enters; the *sign* of ``T`` selects attracting vs. repelling LCS but must not
    change the FTLE value. Ingesting the same advected positions with the time
    bounds swapped (``t0=T1``, ``t1=T0`` -> ``T = T0 - T1 < 0``) must reproduce
    the forward field exactly.
    """
    g_fwd = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    g_bwd = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T1, T0)

    # Sanity: the stored windows are equal and opposite.
    assert g_fwd.ds["T"] == -g_bwd.ds["T"]
    assert g_bwd.ds["T"] < np.timedelta64(0, "s")
    # The FTLE fields must match value-for-value. (Subtraction drops the
    # conflicting scalar t0/T coords, leaving a clean (i, j) difference.)
    assert float(abs(g_fwd.ftle() - g_bwd.ftle()).max()) < 1e-12


# --- NaN propagation -------------------------------------------------------


def test_nan_propagates_through_chain(lon_axis, lat_axis):
    """A lost particle (NaN) flows ``gradF -> C -> eigen -> ftle`` and isolates.

    Knock out a single arm of one ``AuxiliaryGrid`` cell (label-based) so its
    stencil is incomplete. With no special-casing, that one cell's FTLE must be
    NaN while every other cell stays finite -- pinning both that the NaN
    propagates the whole chain *and* that it does not leak to neighbours. This
    also guards the (version-dependent) requirement that ``np.linalg.eigh``
    returns NaN rather than raising ``LinAlgError`` on a NaN sub-matrix.
    """
    g = advected_grid(AuxiliaryGrid, lon_axis, lat_axis, M, T0, T1)
    # Drop the 'south' arm of cell (i=0, j=1) -> NaN advected position there.
    bad = (g.ds["i"] == 0) & (g.ds["j"] == 1) & (g.ds["displacement"] == "south")
    g.ds["lon"] = g.ds["lon"].where(~bad)

    ftle = g.ftle()
    # Exactly one cell is NaN, and it is the corrupted one: the NaN propagated
    # the whole chain and did not leak to any neighbour.
    assert bool(ftle.sel(i=0, j=1).isnull())
    assert int(ftle.isnull().sum()) == 1
