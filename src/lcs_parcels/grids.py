"""Seed and flow-map classes for Lagrangian coherent structure (LCS) diagnostics.

A :class:`Seed` lays out a time-free grid of release positions and emits a
particle set; the advected positions are ingested back into a :class:`FlowMap`,
which computes the deformation gradient, the Cauchy-Green tensor, its
eigen-decomposition and the FTLE. This module contains no Parcels code but just
provides data to construct particle sets and ingests particle positions after
advection::

    seed = NeighborSeed.from_axes(lon=lon, lat=lat)
    lon0, lat0 = seed.to_parcels_pset()
    # ... advect (lon0, lat0) from t0 to t1 ...
    flowmap = seed.pset_to_flowmap(lon=lon1, lat=lat1, t0=t0, t1=t1)
    ftle = flowmap.ftle()            # 1/s, on the (i, j) diagnostic grid
    lcs = flowmap.hyperbolic_lcs()   # or straight to the LCS curves

Pass ``t1`` before ``t0`` for backward integration; a zero window is rejected.
Two stencils for the deformation gradient are two pairs of
classes: :class:`NeighborSeed` / :class:`NeighborFlowMap` difference against
neighbouring grid points, :class:`AuxiliarySeed` / :class:`AuxiliaryFlowMap`
against a four-arm stencil laid around each grid point.

Every object wraps an ``xr.Dataset``, available as ``.ds``. Diagnostics are
reported at the grid points ``lon_grid``/``lat_grid`` (degrees) and every
returned array carries ``long_name`` and ``units``. Lon/lat pairs are
keyword-only throughout. Positions are differenced in metres, each pair in its
own local east/north frame, so the accuracy of a separation depends on how far
apart that pair is and not on how large the domain is or where it sits.
Longitude differences wrap, so the antimeridian is not a special case.
Longitudes are stored in whatever convention they arrive in; only differences
and means are wrapped.

Notation follows Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev.
Fluid Mech. 47:137-162, doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).
"""

from __future__ import annotations

import abc
from typing import Self

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in meters, the sphere all distances are taken on."""

_DEG = np.pi / 180.0
"""Degrees-to-radians factor."""

# --- output metadata -------------------------------------------------------
#
# Attribute sets attached to every coordinate and every returned field, so a
# displayed dataset reads itself and a vanilla plot is labelled from the object.
# A set becomes a constant here when it is attached to more than one object (the
# coordinates, the lon/lat pairs); a set attached to one returned field only
# (`cauchy_green`, `ftle`, ...) is written inline where that field is built.

I_ATTRS = {"long_name": "logical grid index along i"}
J_ATTRS = {"long_name": "logical grid index along j"}
LON_GRID_ATTRS = {
    "long_name": "longitude",
    "units": "degrees_east",
}
LAT_GRID_ATTRS = {
    "long_name": "latitude",
    "units": "degrees_north",
}
LON_0_ATTRS = {
    "long_name": "longitude of the reference release position x_0",
    "units": "degrees_east",
}
LAT_0_ATTRS = {
    "long_name": "latitude of the reference release position x_0",
    "units": "degrees_north",
}
LON_ATTRS = {
    "long_name": "longitude of the advected position F(x_0)",
    "units": "degrees_east",
}
LAT_ATTRS = {
    "long_name": "latitude of the advected position F(x_0)",
    "units": "degrees_north",
}
DISPLACEMENT_ATTRS = {"long_name": "auxiliary stencil arm"}
T0_ATTRS = {"long_name": "release time t0"}
T_ATTRS = {"long_name": "signed integration window T = t1 - t0"}
ROW_ATTRS = {"long_name": "tensor row index"}
COL_ATTRS = {"long_name": "tensor column index"}
COMP_ATTRS = {"long_name": "eigenvector component"}
EIG_ATTRS = {
    "long_name": (
        "Cauchy-Green eigenpair, ascending: "
        "0 is the weak-stretch lambda_1, 1 is lambda_max = lambda_2"
    )
}


def _wrap_lon(dlon):
    """Wrap a longitude *difference* (degrees) into ``[-180, 180]``.

    Applied to differences only, never to a stored position: the package keeps
    longitudes in whatever convention it was handed. Subtracting the nearest
    multiple of 360 rather than shifting by 180 and taking a modulo leaves a
    difference already inside the range bit-for-bit unchanged, so wrapping costs
    no precision on the metre-scale differences the stencils take. Works on plain
    arrays or xarray objects.
    """
    return dlon - 360.0 * np.round(dlon / 360.0)


def _separation_m(*, lon_a, lat_a, lon_b, lat_b):
    """East/north separation of point ``b`` from point ``a``, in meters.

    The pair is differenced in its own local frame: the longitude difference is
    wrapped and scaled by the cosine of the pair's mid-latitude, the latitude
    difference by the Earth radius alone. There is no shared projection and no
    standard parallel, so the frame follows the pair rather than the domain, and
    the antimeridian is not a special case.

    The mid-latitude cosine is a midpoint rule, so it is second-order accurate in
    the *separation of the pair*, not in the size of the domain. For a zonal pair
    the relative error against the great-circle distance is
    ``(dlambda sin(phi))**2 / 24``, so it crosses 1e-6 at ``31.2 km / tan(phi)``:
    54 km at 30 N, 18 km at 60 N, 5.5 km at 80 N, and never at the equator, where
    a parallel is itself a great circle. The auxiliary stencil sits far inside
    that at its default 1 km arms. A neighbour stencil differences over two grid
    cells and can sit outside it, but pays the finite-difference truncation of
    that same span first.

    Returns
    -------
    tuple
        ``(dx, dy)``, the eastward and northward components in meters.
    """
    dlon = _wrap_lon(lon_b - lon_a)
    lat_mid = 0.5 * (lat_a + lat_b)
    dx = EARTH_RADIUS_M * np.cos(lat_mid * _DEG) * dlon * _DEG
    # A position is a pair, so a point NaN in one coordinate is not a point at
    # all. `0.0 * dlon` carries a lost longitude into the north component, which
    # otherwise never touches lon and would report a finite separation for it.
    dy = EARTH_RADIUS_M * ((lat_b - lat_a) + 0.0 * dlon) * _DEG
    return dx, dy


def _circular_mean_lon(lon: xr.DataArray, dim: str) -> xr.DataArray:
    """Mean longitude over ``dim``, taken on the circle.

    Anchored on the first element along ``dim`` and averaged over wrapped offsets
    from it, so a set straddling the antimeridian averages between its members
    and the result stays on the anchor's branch. ``skipna=False``, so a NaN
    member makes the mean NaN.
    """
    anchor = lon.isel({dim: 0}, drop=True)
    return anchor + _wrap_lon(lon - anchor).mean(dim, skipna=False)


def _central_separation_m(lon: xr.DataArray, lat: xr.DataArray, dim: str):
    """Separation of the ``index + 1`` neighbour from the ``index - 1`` one along
    ``dim``, as ``(dx, dy)`` meters.

    ``.shift`` fills NaN past both ends, so the first and last index along ``dim``
    are legitimately NaN: they have no neighbour to difference against.
    """
    return _separation_m(
        lon_a=lon.shift({dim: +1}),
        lat_a=lat.shift({dim: +1}),
        lon_b=lon.shift({dim: -1}),
        lat_b=lat.shift({dim: -1}),
    )


def _arm_separation_m(
    lon: xr.DataArray, lat: xr.DataArray, positive: str, negative: str
):
    """Separation of one auxiliary arm from its opposite, e.g. ``east`` from
    ``west``, as ``(dx, dy)`` meters.

    The arms are selected with ``drop=True``, so the result is back on ``(i, j)``.
    """
    return _separation_m(
        lon_a=lon.sel(displacement=negative, drop=True),
        lat_a=lat.sel(displacement=negative, drop=True),
        lon_b=lon.sel(displacement=positive, drop=True),
        lat_b=lat.sel(displacement=positive, drop=True),
    )


def _assemble_tensor(
    *,
    name: str,
    long_name: str,
    units: str,
    fxx: xr.DataArray,
    fxy: xr.DataArray,
    fyx: xr.DataArray,
    fyy: xr.DataArray,
) -> xr.DataArray:
    """Pack four scalar ``(i, j)`` component fields into a ``(row, col)`` tensor.

    ``row`` and ``col`` become dimension coordinates valued ``['x', 'y']`` with
    ``tensor.sel(row=a, col=b)`` the ``(a, b)`` component, e.g.
    ``gradF.sel(row='y', col='x') = dF_y / dx0_x``. The caller's ``name``,
    ``long_name`` and ``units`` are applied to the assembled result.
    """
    row_x = xr.concat([fxx, fxy], dim="col")
    row_y = xr.concat([fyx, fyy], dim="col")
    tensor = xr.concat([row_x, row_y], dim="row")
    tensor = tensor.assign_coords(
        row=xr.DataArray(["x", "y"], dims="row", attrs=ROW_ATTRS),
        col=xr.DataArray(["x", "y"], dims="col", attrs=COL_ATTRS),
    )
    return tensor.rename(name).assign_attrs(long_name=long_name, units=units)


def _extent(values: xr.DataArray) -> str:
    """``min..max`` of a coordinate in degrees, or ``nan..nan`` if it is all NaN."""
    return f"{float(values.min()):.2f}..{float(values.max()):.2f}"


def _grid_summary(obj: Seed | FlowMap) -> str:
    """Opening of the one-line repr: class, grid shape, lon/lat extent.

    Shared by :meth:`Seed.__repr__` and :meth:`FlowMap.__repr__`, which close it
    (the flow map after appending its timing). Reads the diagnostic grid via the
    ``lon_grid``/``lat_grid`` accessors, so it is stencil-agnostic; ``min``/``max``
    skip NaN.
    """
    lon_grid, lat_grid = obj.lon_grid, obj.lat_grid
    shape = f"{lon_grid.sizes['i']}x{lon_grid.sizes['j']}"
    return (
        f"<{type(obj).__name__} {shape} grid, "
        f"lon {_extent(lon_grid)}, lat {_extent(lat_grid)}"
    )


class Seed(abc.ABC):
    """Time-free wrapper around an ``xr.Dataset`` of seed positions.

    Holds the diagnostic grid-point locations ``lon_grid``/``lat_grid`` and the
    reference release positions ``lon_0``/``lat_0`` (``x_0``, degrees,
    coordinates). No ``t0``, no ``T``, no advected ``lon``/``lat``, no data
    variables; time and advected positions enter at ingest via
    :meth:`pset_to_flowmap`, which produces a :class:`FlowMap`. The dataset is
    held in :attr:`ds`.

    Logical grid dims are ``i, j``; the 2-D ``lon_grid``/``lat_grid`` can
    represent curvilinear grids. For :class:`NeighborSeed` the grid point is the
    release location, so ``lon_0``/``lat_0`` are on ``(i, j)`` and equal
    ``lon_grid``/``lat_grid``; for :class:`AuxiliarySeed` they carry an extra
    ``displacement`` dim, one release position per stencil arm. Concrete
    subclasses differ only in the stencil laid down in :meth:`from_axes`.

    Attributes
    ----------
    ds : xr.Dataset
        Diagnostic grid points ``lon_grid``/``lat_grid`` and reference release
        positions ``lon_0``/``lat_0`` (``x_0``) as coordinates. No data
        variables, no ``t0``/``T``.
    """

    #: The paired :class:`FlowMap` subclass produced by :meth:`pset_to_flowmap`.
    _flowmap_cls: type[FlowMap]

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing time-free dataset of seed positions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j``, the diagnostic grid coordinates
            ``lon_grid``/``lat_grid`` on ``(i, j)``, and the reference coordinates
            ``lon_0``, ``lat_0`` (degrees). Subclasses may require additional
            coordinates/dimensions. No data variables, no ``t0``/``T``.
        """
        self.ds = ds

    @property
    def lon_grid(self) -> xr.DataArray:
        """Longitude of the diagnostic grid points, ``(i, j)``, degrees east."""
        return self.ds["lon_grid"]

    @property
    def lat_grid(self) -> xr.DataArray:
        """Latitude of the diagnostic grid points, ``(i, j)``, degrees north."""
        return self.ds["lat_grid"]

    def __repr__(self) -> str:
        """One-line summary: class, grid shape, lon/lat extent."""
        return f"{_grid_summary(self)}>"

    @classmethod
    @abc.abstractmethod
    def from_axes(cls, *, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a time-free seed from 1-D lon/lat axes.

        The 1-D axes are broadcast into the 2-D diagnostic grid
        ``lon_grid``/``lat_grid`` (degrees), around which the subclass lays down
        its stencil as the reference release positions ``lon_0``/``lat_0``. No
        time is recorded; ``t0`` and ``T`` enter only at :meth:`pset_to_flowmap`.

        Parameters
        ----------
        lon : np.ndarray
            1-D array of longitudes (degrees), length ``Ni``, mapped to dim ``i``.
        lat : np.ndarray
            1-D array of latitudes (degrees), length ``Nj``, mapped to dim ``j``.

        Returns
        -------
        Self
            A seed whose ``.ds`` carries the diagnostic grid
            ``lon_grid``/``lat_grid`` and the reference ``lon_0``/``lat_0`` as
            coordinates, with no data variables and no time.
        """
        raise NotImplementedError("abstract method; implemented by subclasses.")

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten the reference seed positions ``x_0`` to ``(lon, lat)`` lists.

        Emits the reference release positions ``lon_0``/``lat_0``, stacking the
        grid over all of ``lon_0``'s dims into a single ``particle`` index. The
        ``particle`` order is the lossless inverse used by
        :meth:`pset_to_flowmap` to reattach advected positions.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(lon, lat)`` as flat lists of degrees, one entry per particle.
        """
        lon0 = self.ds["lon_0"].reset_coords(drop=True)
        lat0 = self.ds["lat_0"].reset_coords(drop=True)
        dims = lon0.dims
        return (
            list(lon0.stack(particle=dims).values),
            list(lat0.stack(particle=dims).values),
        )

    def pset_to_flowmap(self, *, lon, lat, t0, t1) -> FlowMap:
        """Reattach advected flat lon/lat and produce a :class:`FlowMap`.

        Inverse of :meth:`to_parcels_pset`: the flat advected positions are
        attached to the seed's ``particle`` index as the advected ``lon``/``lat``
        and unstacked back to the grid dims; the reference ``lon_0``/``lat_0``
        (and any auxiliary geometry) are carried through unchanged. Lost
        particles arrive as NaN and propagate. The release time ``t0`` and the
        derived signed window ``T = t1 - t0`` are recorded as scalar coordinates
        on the returned flow map.

        Parameters
        ----------
        lon, lat : array-like
            Advected longitudes/latitudes (degrees), aligned with the order of
            :meth:`to_parcels_pset` output.
        t0 : datetime64-like
            Release time of the seed positions.
        t1 : datetime64-like
            End time of the integration. The signed window ``T = t1 - t0`` sets
            the direction (``t1 < t0`` backward; ``t1 > t0`` forward).
            ``t1`` is not stored (recoverable as ``t0 + T``).

        Returns
        -------
        FlowMap
            The paired concrete flow map carrying the advected positions on the
            original grid plus scalar ``t0`` and signed ``T`` coordinates.

        Raises
        ------
        ValueError
            If ``t1 == t0`` (zero window): the FTLE's ``1 / |T|`` would divide by
            zero.
        """
        dims = self.ds["lon_0"].dims
        lon_arm = (
            self.ds["lon_0"]
            .reset_coords(drop=True)
            .stack(particle=dims)
            .copy(data=np.asarray(lon, dtype=float))
            .unstack("particle")
        )
        lat_arm = (
            self.ds["lat_0"]
            .reset_coords(drop=True)
            .stack(particle=dims)
            .copy(data=np.asarray(lat, dtype=float))
            .unstack("particle")
        )
        t0 = np.datetime64(t0)
        T = np.datetime64(t1) - t0
        if T == np.timedelta64(0, "s"):
            raise ValueError(
                "zero integration window T = t1 - t0; FTLE would divide by zero"
            )
        ds = self.ds.assign(
            lon=lon_arm.assign_attrs(LON_ATTRS),
            lat=lat_arm.assign_attrs(LAT_ATTRS),
        ).assign_coords(
            t0=xr.DataArray(t0, attrs=T0_ATTRS),
            T=xr.DataArray(T, attrs=T_ATTRS),
        )
        return self._flowmap_cls(ds)


class FlowMap(abc.ABC):
    """Wrapper around an advected ``xr.Dataset``; computes the diagnostics.

    Holds the reference release positions ``lon_0``/``lat_0`` (``x_0``,
    coordinates) *and* the advected positions ``lon``/``lat`` (the flow map
    ``F_{t0}^{t1}(x_0)``, data variables), plus a scalar release time ``t0`` and
    the signed window ``T = t1 - t0`` (``timedelta64``). Produced by
    :meth:`Seed.pset_to_flowmap`; the dataset is held in :attr:`ds`.

    Both position pairs share the same dims so ``grad F = d(lon, lat) /
    d(lon_0, lat_0)`` is well-defined: ``(i, j)`` for :class:`NeighborFlowMap`,
    ``(i, j, displacement)`` for :class:`AuxiliaryFlowMap`. Both carry the
    diagnostic grid points ``lon_grid``/``lat_grid`` on ``(i, j)``, which is
    where the diagnostics are reported. Concrete subclasses differ only in how
    ``grad F`` is finite-differenced and how the advected positions collapse
    onto the diagnostic grid (:attr:`grid_image`).

    Attributes
    ----------
    ds : xr.Dataset
        Diagnostic grid points ``lon_grid``/``lat_grid``, reference positions
        ``lon_0``/``lat_0`` (``x_0``), advected positions ``lon``/``lat``
        (``F(x_0)``), the scalar release time ``t0`` and the signed window
        ``T = t1 - t0`` (``timedelta64``).
    """

    #: The paired :class:`Seed` subclass produced by :meth:`to_seed`.
    _seed_cls: type[Seed]

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing advected dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j``, the diagnostic grid coordinates
            ``lon_grid``/``lat_grid`` on ``(i, j)``, reference coordinates
            ``lon_0``/``lat_0``, advected data variables ``lon``/``lat``
            (degrees), and scalar ``t0`` and signed ``T`` coordinates. Subclasses
            may require additional variables/dimensions.
        """
        self.ds = ds

    @property
    def lon_grid(self) -> xr.DataArray:
        """Longitude of the diagnostic grid points, ``(i, j)``, degrees east."""
        return self.ds["lon_grid"]

    @property
    def lat_grid(self) -> xr.DataArray:
        """Latitude of the diagnostic grid points, ``(i, j)``, degrees north."""
        return self.ds["lat_grid"]

    def __repr__(self) -> str:
        """One-line summary: class, grid shape, lon/lat extent, ``t0`` and ``T``."""
        return (
            f"{_grid_summary(self)}, "
            f"t0 {np.datetime64(self.ds['t0'].values, 's')}, "
            f"T {self._window_days():+.1f} days>"
        )

    def _time_direction(self) -> str:
        """``'forward'`` or ``'backward'``, from ``sign(T)``.

        A flow map is integrated one way or the other and knows nothing more
        than that. Which LCS family that yields is a property of the diagnostic,
        so the repelling/attracting wording lives in :meth:`hyperbolic_lcs`.
        """
        if self.ds["T"] > np.timedelta64(0, "s"):
            return "forward"
        return "backward"

    def _window_days(self) -> float:
        """The signed window ``T`` in days."""
        return float(self.ds["T"] / np.timedelta64(1, "D"))

    @property
    @abc.abstractmethod
    def grid_image(self) -> xr.Dataset:
        """The flow map image of the diagnostic grid points, ``F_{t0}^{t1}(x_grid)``.

        The advected positions reduced onto the ``(i, j)`` diagnostic grid: one
        ``lon``/``lat`` pair per grid point, whatever the stencil that grid point
        was released with. Subclasses define the reduction.

        Returns
        -------
        xr.Dataset
            ``lon``/``lat`` (degrees) on dims ``(i, j)``.
        """
        raise NotImplementedError("abstract property; implemented by subclasses.")

    def _integration_seconds(self) -> float:
        """``|T|`` in seconds from the stored signed window ``T`` (``timedelta64``)."""
        return float(np.abs(self.ds["T"] / np.timedelta64(1, "s")))

    def _positions(self) -> tuple[xr.DataArray, ...]:
        """Reference and advected positions in degrees, as
        ``(lon_0, lat_0, lon, lat)``.

        The ``lon_0``/``lat_0`` release-position coords are dropped: everything
        differenced from these is reported at the diagnostic grid point, so it
        must be labelled ``lon_grid``/``lat_grid`` and not by a release
        position -- which for :class:`AuxiliaryFlowMap` is one stencil arm.

        Degrees, not meters: a separation only becomes meters once two positions
        are named, and it is then taken in the local frame of that pair
        (:func:`_separation_m`). The advected pair therefore gets its own cosine,
        not the reference pair's, which is what makes ``grad F`` free of any
        shared frame.
        """
        release = ["lon_0", "lat_0"]
        return (
            self.ds["lon_0"].drop_vars(release),
            self.ds["lat_0"].drop_vars(release),
            self.ds["lon"].drop_vars(release),
            self.ds["lat"].drop_vars(release),
        )

    @abc.abstractmethod
    def deformation_gradient(self) -> xr.DataArray:
        """Deformation gradient grad F of the flow map. Haller (2015) Eq. 9.

        The 2x2 tensor ``grad F = d(lon, lat) / d(lon_0, lat_0)`` per grid point,
        finite-differenced as ``(advected separation) / (initial separation)``,
        each separation taken in meters in its own local east/north frame
        (:func:`_separation_m`): the denominator from the reference
        ``lon_0``/``lat_0``, the numerator from the advected
        ``lon``/``lat``. Subclasses define the stencil: neighbouring grid points
        (:class:`NeighborFlowMap`) or the fixed four-arm auxiliary stencil
        (:class:`AuxiliaryFlowMap`). Cells with a missing stencil point yield NaN.

        Returns
        -------
        xr.DataArray
            grad F with dims ``(i, j, row, col)``; ``row`` and ``col`` are
            dimension coordinates valued ``['x', 'y']``, with
            ``grad F.sel(row=a, col=b) = d F_a / d x0_b`` (dimensionless;
            meters / meters). The tensor carries no ``comp`` coordinate (``comp``
            labels the eigenvector component dim).
        """
        raise NotImplementedError("abstract method; implemented by subclasses.")

    def cauchy_green(self) -> xr.DataArray:
        """Right Cauchy-Green strain tensor ``C = (grad F)^T grad F``.

        Haller (2015) Eq. 6. Symmetric positive-(semi)definite 2x2 tensor per
        grid point, built from :meth:`deformation_gradient`.

        Returns
        -------
        xr.DataArray
            ``C`` with dims ``(i, j, row, col)`` and ``row``/``col`` dimension
            coordinates valued ``['x', 'y']`` (dimensionless).
        """
        gradF = self.deformation_gradient()
        # C_{a,b} = sum_k gradF_{k,a} gradF_{k,b}: contract over the shared output
        # index `row` by label, then relabel the two surviving `col` axes back to
        # (row, col). Pure label-based arithmetic; no positional indexing.
        C = xr.dot(gradF, gradF.rename(col="col_b"), dim="row")
        C = C.rename(col="row", col_b="col").rename("cauchy_green")
        # Before xarray 2025.11, `xr.dot` strips the attrs off every coordinate it
        # carries through. Restore them from the operand.
        C = C.assign_coords(
            {
                name: C[name].assign_attrs(gradF[name].attrs)
                for name in C.coords
                if name in gradF.coords and name not in ("row", "col")
            }
        )
        # Separate call: `row` and `col` are new axes here, not restored ones.
        C = C.assign_coords(
            row=xr.DataArray(["x", "y"], dims="row", attrs=ROW_ATTRS),
            col=xr.DataArray(["x", "y"], dims="col", attrs=COL_ATTRS),
        )
        return C.assign_attrs(
            long_name="right Cauchy-Green strain tensor C = (grad F)^T grad F",
            units="1",
        )

    def cg_eigen(self) -> xr.Dataset:
        """Eigen-decomposition of the Cauchy-Green tensor ``C``.

        Haller (2015) Eq. 7. Solves ``C xi_i = lambda_i xi_i`` with
        ``0 < lambda_1 <= lambda_2`` and orthonormal eigenvectors
        ``xi_1 perp xi_2``.

        Returns
        -------
        xr.Dataset
            Variables ``lambda`` with dims ``(i, j, eig)`` (eigenvalues in
            ascending order, dimensionless) and ``xi`` with dims
            ``(i, j, comp, eig)`` (orthonormal eigenvectors, component coordinate
            ``comp = ['x', 'y']``).
        """
        C = self.cauchy_green()
        # eigh works on the trailing two axes and returns ascending eigenvalues
        # `w[..., k]` with eigenvector `v[..., :, k]`; apply_ufunc moves the
        # (row, col) core dims last so the first output axis of `v` is the vector
        # component and the second selects which eigenpair.
        lam, vec = xr.apply_ufunc(
            np.linalg.eigh,
            C,
            input_core_dims=[["row", "col"]],
            output_core_dims=[["eig"], ["comp", "eig"]],
        )
        eig = xr.DataArray([0, 1], dims="eig", attrs=EIG_ATTRS)
        comp = xr.DataArray(["x", "y"], dims="comp", attrs=COMP_ATTRS)
        lam = lam.assign_coords(eig=eig).assign_attrs(
            long_name="eigenvalue lambda of the Cauchy-Green tensor",
            units="1",
        )
        vec = vec.assign_coords(comp=comp, eig=eig).assign_attrs(
            long_name="eigenvector xi of the Cauchy-Green tensor",
            units="1",
        )
        return xr.Dataset({"lambda": lam, "xi": vec})

    def ftle(self) -> xr.DataArray:
        """Finite-time Lyapunov exponent (FTLE). Haller (2015) Sec. 4.1.

        Computes ``Lambda = (1 / |T|) * log(sqrt(lambda_max))`` using the
        *largest* eigenvalue of ``C`` from :meth:`cg_eigen` and ``|T|`` in
        seconds from the recorded signed window.

        Returns
        -------
        xr.DataArray
            FTLE field with dims ``(i, j)`` in units of 1/second.
        """
        # drop=True removes the scalar `eig` coord the selection leaves, so the
        # FTLE carries only its documented (i, j) dims.
        lambda_max = self.cg_eigen()["lambda"].isel(eig=1, drop=True)
        t_sec = self._integration_seconds()
        # (1 / |T|) log sqrt(lambda_max) = (1 / |T|) * 0.5 * log(lambda_max).
        ftle = (1.0 / t_sec) * 0.5 * np.log(lambda_max)
        return ftle.rename("ftle").assign_attrs(
            long_name="finite-time Lyapunov exponent",
            units="1/s",
        )

    def image(self, *, lon0: xr.DataArray, lat0: xr.DataArray) -> xr.Dataset:
        """Advected positions ``F_{t0}^{t1}(x_0)`` at arbitrary reference points.

        Interpolates the stored advected-position field ``lon``/``lat`` at the
        reference locations ``lon0``/``lat0`` (degrees), i.e. where those
        material points sit at ``t1``. Evolving an extracted material curve is
        exactly this -- feed a curve's own ``lon``/``lat`` (its ``x_0``) and read
        back its later position, ``M(t1) = F_{t0}^{t1}(M(t0))`` (Haller 2015
        Eq. 5).

        Vectorized and dim-preserving: ``lon0``/``lat0`` are ``DataArray``s on
        any shared dims -- e.g. the ``(line, point)`` grid of
        :func:`~lcs_parcels.shrink_lines` -- and the output carries those dims.
        Points off the grid, in a NaN (land/edge) cell, or NaN themselves map to
        NaN, so an evolved curve terminates exactly where the flow map is
        undefined.

        Rectilinear grids only, like :func:`~lcs_parcels.shrink_lines`: the
        advected field (:attr:`grid_image`) is read on the axis-aligned
        diagnostic grid ``lon_grid``/``lat_grid`` (``lon_grid`` varying along
        ``i``, ``lat_grid`` along ``j``), and the ``lon_grid`` axis must be
        monotonic -- a domain crossing the antimeridian is seeded ``170, 175,
        180, 185``, not ``170, 175, 180, -175``.

        The advected longitudes themselves may arrive on any branch: they are
        re-anchored on the branch of the grid point they came from before the
        interpolation, so an advection that returns positions wrapped to
        ``[-180, 180)`` is read correctly. The returned longitudes are on that
        same branch, which is the one ``lon0`` was given in.

        Parameters
        ----------
        lon0, lat0 : xr.DataArray
            Reference positions ``x_0`` (degrees) to map, sharing dims.

        Returns
        -------
        xr.Dataset
            ``lon``/``lat`` (degrees) on the dims of ``lon0``/``lat0`` -- the
            same structure a :func:`~lcs_parcels.shrink_lines` curve has, so an
            evolved curve is drop-in plottable and can itself be re-fed. The
            requested reference positions ride along as the ``lon_0``/``lat_0``
            coords.
        """
        # Interpolation is arithmetic on the advected longitudes, so they have to
        # be on one branch first: an advection that hands positions back wrapped
        # to [-180, 180) tears from 179.9 to -179.9 between two adjacent grid
        # points, and a linear interpolant reads that tear as a 40 000 km jump.
        # Re-anchoring each image on the branch of the grid point it came from
        # removes the tear whenever the displacement is under 180 degrees, which
        # is every advection short of half the globe.
        grid_image = self.grid_image
        grid_image = grid_image.assign(
            lon=(
                self.lon_grid + _wrap_lon(grid_image["lon"] - self.lon_grid)
            ).assign_attrs(LON_ATTRS)
        )
        # Relabel the logical (i, j) index axes by their geographic values so the
        # interpolation runs against lon/lat directly.
        advected = (
            grid_image.reset_coords(drop=True)
            .assign_coords(
                i=self.lon_grid.isel(j=0, drop=True).values,
                j=self.lat_grid.isel(i=0, drop=True).values,
            )
            .rename(i="lon_grid", j="lat_grid")
        )
        image = advected.interp(
            lon_grid=lon0,
            lat_grid=lat0,
            kwargs={"bounds_error": False, "fill_value": np.nan},
        )
        # The interpolation carries the requested reference positions through
        # under the axis names it interpolated along; they are reference
        # positions x_0, not diagnostic grid points, so they come back as
        # lon_0/lat_0. Any lon_0/lat_0 the inputs brought along as coords of
        # their own is dropped first, so the returned pair is always the points
        # actually interpolated at.
        image = image.drop_vars(["lon_0", "lat_0"], errors="ignore").rename(
            lon_grid="lon_0", lat_grid="lat_0"
        )
        # Rebuilt rather than stamped through ``assign``/``assign_coords``: those
        # merge, and a merge cannot tell whether an incoming ``lon`` is the data
        # variable or the dimension of the same name. Callers passing lon0/lat0
        # straight off a CF dataset -- the most natural way to call this -- have
        # exactly that collision, so the merging forms raise ``MergeError`` on
        # them. Naming the data variables and the coordinates separately says
        # which is which, and never merges.
        return xr.Dataset(
            {
                "lon": image["lon"].assign_attrs(LON_ATTRS),
                "lat": image["lat"].assign_attrs(LAT_ATTRS),
            },
            coords={
                "lon_0": image["lon_0"].assign_attrs(LON_0_ATTRS),
                "lat_0": image["lat_0"].assign_attrs(LAT_0_ATTRS),
            },
        )

    def hyperbolic_lcs(
        self,
        *,
        window_m: float | None = None,
        quantile: float | None = None,
        min_anisotropy: float | None = None,
        step_m: float | None = None,
        line_length_m: float | None = None,
    ) -> xr.Dataset:
        """Hyperbolic LCS of this flow map: FTLE, ridge seeds, and shrink lines.

        Runs the three-step workflow in one call: compute the FTLE field
        (:meth:`ftle`), pick seed points at its strong local maxima
        (:func:`~lcs_parcels.ftle_ridge_seeds`), and integrate the shrink lines
        through them (:func:`~lcs_parcels.shrink_lines`). The FTLE is computed
        once and handed to the ridge finder; to pick ridges from a smoothed or
        masked field instead, drive the three steps yourself.

        A *forward* flow map (``T > 0``) yields repelling LCS, a *backward* one
        (``T < 0``) attracting LCS, by the forward-backward duality (Haller &
        Sapsis 2011, https://doi.org/10.1063/1.3579597); the sign of the stored
        window decides, and the returned dataset says which it holds.

        Every call recomputes :meth:`ftle`, which is the expensive part of the
        chain. Re-tuning the ridge or integration parameters against a fixed
        field is therefore a case for driving :meth:`ftle`,
        :func:`~lcs_parcels.ftle_ridge_seeds` and
        :func:`~lcs_parcels.shrink_lines` yourself, computing the field once.

        Parameters
        ----------
        window_m, quantile : float, optional
            Ridge-selection parameters, passed to
            :func:`~lcs_parcels.ftle_ridge_seeds`.
        min_anisotropy, step_m, line_length_m : float, optional
            Integration parameters, passed to
            :func:`~lcs_parcels.shrink_lines`.

        Returns
        -------
        xr.Dataset
            ``lon``/``lat`` (degrees) on dims ``(line, point)`` -- the LCS curves,
            NaN past termination -- together with the ``ftle`` field (1/s) on
            ``(i, j)`` that the seeds were picked from, so the curves can be
            plotted over it without recomputing.
        """
        # This method is a layering inversion: `grids` is the lower layer, and
        # here it reaches up into `tensorlines`, which imports from it. The
        # deferred import is the standard remedy, taken deliberately because the
        # one-call ergonomics of this method are worth the inversion.
        from lcs_parcels.tensorlines import ftle_ridge_seeds, shrink_lines

        # Forwarding a None would push the sentinel into the public signatures of
        # ftle_ridge_seeds and shrink_lines, which would each then have to
        # resolve it; dropping the unset arguments here keeps exactly one home
        # for every default.
        def _filter_kwargs(**kwargs) -> dict[str, float]:
            return {k: v for k, v in kwargs.items() if v is not None}

        ftle = self.ftle()
        seed_lon, seed_lat = ftle_ridge_seeds(
            ftle, **_filter_kwargs(window_m=window_m, quantile=quantile)
        )
        lines = shrink_lines(
            self,
            seed_lon=seed_lon,
            seed_lat=seed_lat,
            **_filter_kwargs(
                min_anisotropy=min_anisotropy,
                step_m=step_m,
                line_length_m=line_length_m,
            ),
        )
        # The forward-backward duality is the diagnostic's, not the flow map's.
        direction = self._time_direction()
        kind = "repelling" if direction == "forward" else "attracting"
        lines = lines.assign(
            lon=lines["lon"].assign_attrs(long_name=f"longitude along the {kind} LCS"),
            lat=lines["lat"].assign_attrs(long_name=f"latitude along the {kind} LCS"),
        )
        return lines.assign(ftle=ftle).assign_attrs(
            long_name=(
                f"{kind} LCS: shrink lines of the {direction} flow map, "
                "with the FTLE field their seeds were picked from"
            )
        )

    def to_seed(self) -> Seed:
        """Drop the advected positions and time, recovering a time-free seed.

        Lossless inverse of :meth:`Seed.pset_to_flowmap`: removes the advected
        ``lon``/``lat`` and the scalar ``t0``/``T`` coords, leaving the reference
        positions (and any auxiliary geometry). Re-emitting reproduces the same
        flat particle set.

        Returns
        -------
        Seed
            The paired concrete time-free seed.
        """
        ds = self.ds.drop_vars(["lon", "lat", "t0", "T"])
        return self._seed_cls(ds)


class NeighborSeed(Seed):
    """Time-free seed whose stencil is the neighbouring grid points.

    The release point is the grid point itself, so ``.ds`` carries the reference
    positions ``lon_0``/``lat_0`` on ``(i, j)`` -- no displacement dim -- equal
    to the diagnostic grid ``lon_grid``/``lat_grid``. The paired
    :class:`NeighborFlowMap` differences ``grad F`` against neighbouring grid
    points ``(i +/- 1, j +/- 1)``, coupling the diagnostic resolution to the seed
    grid resolution.
    """

    @classmethod
    def from_axes(cls, *, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a neighbour-stencil seed from 1-D lon/lat axes.

        See :meth:`Seed.from_axes`. The 1-D axes are broadcast into axis-aligned
        (rectilinear) 2-D fields on ``(i, j)`` (lon varies along ``i``, lat along
        ``j``) -- the layout the paired :class:`NeighborFlowMap` gradient
        assumes -- and stored both as the diagnostic grid
        ``lon_grid``/``lat_grid`` and as the reference release positions
        ``lon_0``/``lat_0``, which coincide for this stencil.
        """
        # Broadcast the 1-D axes into axis-aligned 2-D fields on (i, j) with the
        # high-level API; lon varies along i, lat along j.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon2d, lat2d = xr.broadcast(lon_axis, lat_axis)

        ds = xr.Dataset(
            coords={
                "i": xr.DataArray(
                    np.arange(lon_axis.sizes["i"]), dims="i", attrs=I_ATTRS
                ),
                "j": xr.DataArray(
                    np.arange(lat_axis.sizes["j"]), dims="j", attrs=J_ATTRS
                ),
                # Diagnostic grid points; stored rather than reconstructed from
                # lon_0/lat_0, which happen to coincide for this stencil.
                "lon_grid": lon2d.assign_attrs(LON_GRID_ATTRS),
                "lat_grid": lat2d.assign_attrs(LAT_GRID_ATTRS),
                # Reference initial positions x_0: the grid points themselves.
                "lon_0": lon2d.assign_attrs(LON_0_ATTRS),
                "lat_0": lat2d.assign_attrs(LAT_0_ATTRS),
            },
        )
        return cls(ds)


class AuxiliarySeed(Seed):
    """Time-free seed with a fixed four-arm auxiliary displacement stencil.

    The reference release positions ``lon_0(i, j, displacement)`` /
    ``lat_0(i, j, displacement)`` (coordinates, degrees) are the explicit per-arm
    positions -- exactly what :meth:`Seed.to_parcels_pset` emits. The diagnostic
    grid points ``lon_grid(i, j)`` / ``lat_grid(i, j)`` (coordinates) are the arm
    centres, on which the diagnostics are reported.

    Each grid point carries four arms ``east, north, west, south`` at offsets
    ``east = (+s, 0)``, ``north = (0, +s)``, ``west = (-s, 0)``,
    ``south = (0, -s)`` for ``s = aux_separation_m`` -- no centre point, no
    diagonals. The arms are placed in each grid point's own local east/north
    frame (see :func:`_separation_m`), so the gradient step is ``s`` rather than
    the seed grid spacing, at every latitude. The paired
    :class:`AuxiliaryFlowMap` differences ``grad F`` across the four arms
    (east-west, north-south).
    """

    @classmethod
    def from_axes(
        cls,
        *,
        lon: np.ndarray,
        lat: np.ndarray,
        aux_separation_m: float = 1_000.0,
    ) -> Self:
        """Build an auxiliary-stencil seed from 1-D lon/lat axes.

        See :meth:`Seed.from_axes`. The diagnostic grid points ``lon_grid`` /
        ``lat_grid`` are placed on ``(i, j)`` from the axes, then the four-arm
        ``displacement = ['east', 'north', 'west', 'south']`` stencil is laid out
        around each of them at offsets ``east = (+s, 0)``, ``north = (0, +s)``,
        ``west = (-s, 0)``, ``south = (0, -s)`` meters (``s = aux_separation_m``)
        and stored as the reference release positions ``lon_0`` / ``lat_0`` on
        ``(i, j, displacement)``.

        Parameters
        ----------
        lon, lat : np.ndarray
            1-D longitude/latitude axes (degrees); see :meth:`Seed.from_axes`.
        aux_separation_m : float, optional
            Auxiliary separation ``s`` (meters) applied to every arm; sets the
            finite-difference step (the reference arm span is ``2s``). Chosen
            small relative to the flow scale.

        Raises
        ------
        ValueError
            If ``s`` spans 90 degrees of longitude or more at any grid point.
            That happens closer to a pole than ``2 s / pi``, about 0.64 times the
            arm separation itself: 640 m from it for the default 1 km arms, but
            32 km for 50 km arms.
        """
        # Broadcast the 1-D axes into curvilinear 2-D fields on (i, j); lon
        # varies along i, lat along j. These are the diagnostic grid points.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon_grid, lat_grid = xr.broadcast(lon_axis, lat_axis)

        # Four-arm stencil offsets in meters.
        s = float(aux_separation_m)
        displacement = ["east", "north", "west", "south"]
        off_x = xr.DataArray(
            [+s, 0.0, -s, 0.0],
            dims="displacement",
            coords={"displacement": displacement},
        )
        off_y = xr.DataArray(
            [0.0, +s, 0.0, -s],
            dims="displacement",
            coords={"displacement": displacement},
        )

        # Place the arms in each grid point's own local east/north frame, so the
        # east-west and north-south spans are exactly 2s at every latitude.
        # lon_grid (i, j) broadcasts with the (displacement,) offset into the arm
        # positions (i, j, displacement); the east and west arms sit at
        # lat_grid, so cos(lat_grid) is exactly their mid-latitude cosine.
        deg_per_m = 1.0 / (EARTH_RADIUS_M * np.cos(lat_grid * _DEG) * _DEG)
        # `s` degrees of longitude grows without bound as cos(lat) -> 0, and past
        # 180 it aliases through the wrap into an arm on the far side of the
        # pole -- a wrong gradient rather than a NaN. There is no east at the
        # pole, so this is rejected rather than approximated.
        if not bool((np.abs(s * deg_per_m) < 90.0).all()):
            raise ValueError(
                f"aux_separation_m={s} spans 90 degrees or more of longitude at "
                f"latitude {float(np.abs(lat_grid).max())}; the east-west arms are "
                "not local there. Use a smaller separation or keep the seed off "
                "the pole."
            )
        lon_0 = lon_grid + off_x * deg_per_m
        lat_0 = lat_grid + off_y / (EARTH_RADIUS_M * _DEG)

        ds = xr.Dataset(
            coords={
                "i": xr.DataArray(
                    np.arange(lon_axis.sizes["i"]), dims="i", attrs=I_ATTRS
                ),
                "j": xr.DataArray(
                    np.arange(lat_axis.sizes["j"]), dims="j", attrs=J_ATTRS
                ),
                "displacement": xr.DataArray(
                    displacement, dims="displacement", attrs=DISPLACEMENT_ATTRS
                ),
                # Diagnostic grid points (no displacement dim).
                "lon_grid": lon_grid.assign_attrs(LON_GRID_ATTRS),
                "lat_grid": lat_grid.assign_attrs(LAT_GRID_ATTRS),
                # Explicit per-arm reference release positions x_0.
                "lon_0": lon_0.assign_attrs(LON_0_ATTRS),
                "lat_0": lat_0.assign_attrs(LAT_0_ATTRS),
            },
        )
        return cls(ds)


class NeighborFlowMap(FlowMap):
    """Advected flow map whose stencil is the neighbouring grid points.

    ``.ds`` carries the reference and advected positions on ``(i, j)`` (no
    displacement dim), so each grid point has exactly one advected position.
    ``grad F`` is differenced against neighbouring grid points
    ``(i +/- 1, j +/- 1)``, so boundary cells lacking a neighbour are NaN. See
    Haller (2015) Eq. 9 and the paired :class:`NeighborSeed`.

    Axis-aligned grids only: the neighbour gradient divides each tensor column by
    a single axis step and drops the off-diagonal metric terms
    (``d lon_0 / d j``, ``d lat_0 / d i``), so it is correct only when ``lon_0``
    varies along ``i`` and ``lat_0`` along ``j``. :meth:`Seed.from_axes` always
    builds such a grid, but :meth:`FlowMap.__init__` accepts any dataset, so the
    limitation is latent. Use :class:`AuxiliaryFlowMap` for curvilinear grids.
    """

    @property
    def grid_image(self) -> xr.Dataset:
        """The advected positions, already one per grid point. See
        :attr:`FlowMap.grid_image`."""
        return self.ds[["lon", "lat"]]

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced against neighbouring grid points ``(i +/- 1, j +/- 1)``.

        Numerator: the separation of the advected neighbour positions;
        denominator: the initial neighbour separation, each in its own local
        east/north frame (:func:`_separation_m`). Boundary cells lacking a
        neighbour yield NaN. Each column is divided by a single axis step
        (``dx0`` along ``i``, ``dy0`` along ``j``), so this assumes an
        axis-aligned grid (see the class docstring). See
        :meth:`FlowMap.deformation_gradient`.
        """
        lon_0, lat_0, lon, lat = self._positions()

        # lon_0 varies along i and lat_0 along j, so the denominators are the
        # pure east- and north-separations of the two neighbours. The names are
        # spans over two cells, not one-cell steps.
        span_x, _ = _central_separation_m(lon_0, lat_0, "i")
        _, span_y = _central_separation_m(lon_0, lat_0, "j")
        dx_i, dy_i = _central_separation_m(lon, lat, "i")
        dx_j, dy_j = _central_separation_m(lon, lat, "j")
        return _assemble_tensor(
            name="deformation_gradient",
            long_name="deformation gradient grad F of the flow map",
            units="1",
            fxx=dx_i / span_x,
            fxy=dx_j / span_y,
            fyx=dy_i / span_x,
            fyy=dy_j / span_y,
        )


class AuxiliaryFlowMap(FlowMap):
    """Advected flow map differenced across the fixed four-arm auxiliary stencil.

    ``.ds`` carries the reference and advected arm positions on
    ``(i, j, displacement)`` plus the diagnostic grid points
    ``lon_grid``/``lat_grid`` on ``(i, j)``. The per-point stencil makes
    ``grad F`` well-defined at every grid point, including the boundary. See
    Haller (2015) Eq. 9 and the paired :class:`AuxiliarySeed`.
    """

    @property
    def grid_image(self) -> xr.Dataset:
        """The centroid of the four advected arms -- the arms sit ~metres apart,
        so this is the flow map image of the grid point. The longitudes are
        averaged on the circle (:func:`_circular_mean_lon`), so four arms
        straddling the antimeridian average between themselves. ``skipna=False``
        so a lost (NaN) arm makes the whole grid point NaN, matching the
        deformation-gradient path. See :attr:`FlowMap.grid_image`."""
        advected = self.ds[["lon", "lat"]].drop_vars(["lon_0", "lat_0"])
        return xr.Dataset(
            {
                "lon": _circular_mean_lon(advected["lon"], "displacement").assign_attrs(
                    LON_ATTRS
                ),
                "lat": advected["lat"]
                .mean("displacement", skipna=False)
                .assign_attrs(LAT_ATTRS),
            }
        )

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced across the four-arm auxiliary stencil.

        ``grad F = d(lon, lat) / d(lon_0, lat_0)`` over the ``displacement`` dim:
        east minus west for the ``x`` derivative, north minus south for ``y``.
        Both the advected separation (numerator) and the reference arm separation
        (denominator, the ``2s`` span) are read in their own local east/north
        frame (:func:`_separation_m`). Well-defined at every grid point,
        including the boundary. See :meth:`FlowMap.deformation_gradient`.
        """
        lon_0, lat_0, lon, lat = self._positions()

        arm_span_x, _ = _arm_separation_m(lon_0, lat_0, "east", "west")
        _, arm_span_y = _arm_separation_m(lon_0, lat_0, "north", "south")
        dx_ew, dy_ew = _arm_separation_m(lon, lat, "east", "west")
        dx_ns, dy_ns = _arm_separation_m(lon, lat, "north", "south")
        return _assemble_tensor(
            name="deformation_gradient",
            long_name="deformation gradient grad F of the flow map",
            units="1",
            fxx=dx_ew / arm_span_x,
            fxy=dx_ns / arm_span_y,
            fyx=dy_ew / arm_span_x,
            fyy=dy_ns / arm_span_y,
        )


# --- paired seed <-> flow-map wiring ---------------------------------------
#
# Each concrete seed knows the flow map it produces, and each flow map knows the
# seed it collapses back to; explicit class attributes, not inheritance.
NeighborSeed._flowmap_cls = NeighborFlowMap
AuxiliarySeed._flowmap_cls = AuxiliaryFlowMap
NeighborFlowMap._seed_cls = NeighborSeed
AuxiliaryFlowMap._seed_cls = AuxiliarySeed
