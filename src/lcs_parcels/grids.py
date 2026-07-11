"""Seed and flow-map classes for Lagrangian coherent structure (LCS) diagnostics.

The diagnostic layer on top of trajectory integration. It contains no Parcels
code: it emits particle sets and ingests their advected positions. Notation
follows Haller (2015), *Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech.
47:137-162, doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).

Seed / flow-map split
---------------------
Two sibling families of composition wrappers around an ``xr.Dataset`` (held in
``.ds``; neither subclasses ``xr.Dataset``):

- a :class:`Seed` is time-free: it holds only the reference release positions
  ``lon_0``/``lat_0`` (initial conditions ``x_0``, coordinates) and -- for
  :class:`AuxiliarySeed` -- the per-arm stencil geometry and grid-point centres.
  No ``t0``, no ``T``, no advected ``lon``/``lat``, no data variables. It emits
  a particle set (:meth:`Seed.to_parcels_pset`) and ingests advected positions
  (:meth:`Seed.pset_to_flowmap`).
- a :class:`FlowMap` holds the reference positions *and* the advected positions
  ``lon``/``lat`` (the flow map ``F_{t0}^{t1}(x_0)``, data variables), plus a
  scalar release time ``t0`` and the signed window ``T = t1 - t0``. It computes
  the diagnostics (deformation gradient, Cauchy-Green tensor, eigen-analysis,
  FTLE).

Each family has two concrete subclasses for the two finite-difference stencils
(``Neighbor*`` and ``Auxiliary*``); the stencil is a distinct type, never
inferred from the dataset dims.

Timing convention
-----------------
Time enters only at ingest. :meth:`Seed.pset_to_flowmap` takes the release time
``t0`` and end time ``t1``; the signed window ``T = t1 - t0`` is derived and
stored as a scalar coordinate (``t1`` is recoverable as ``t0 + T``). Direction
is ``sign(T)``: ``t1 < t0`` backward/attracting, ``t1 > t0`` forward/repelling.
A zero window (``t1 == t0``) is rejected. See ``plans/seed-flowmap-design.md``.

Position convention
-------------------
Reference positions ``lon_0``/``lat_0`` (``x_0``, coordinates) are what
:meth:`Seed.to_parcels_pset` emits and the deformation gradient differences
against; advected positions ``lon``/``lat`` (``F_{t0}^{t1}(x_0)``, data
variables on a :class:`FlowMap`) are where particles end up. The deformation
gradient is ``grad F = d(lon, lat) / d(lon_0, lat_0)``, so both pairs share the
same dims: ``(i, j)`` for :class:`NeighborSeed` / :class:`NeighborFlowMap`;
``(i, j, displacement)`` for :class:`AuxiliarySeed` / :class:`AuxiliaryFlowMap`,
which additionally keep the diagnostic grid-point centres ``lon_c``/``lat_c`` on
``(i, j)``.

Sphere metric convention
------------------------
Separations are formed in a single local-tangent meters frame anchored at the
grid centroid ``lon_ref = lon_0.mean()``, ``lat_ref = lat_0.mean()``. With ``R``
the Earth radius and ``deg = pi / 180``:
``X = R cos(phi_ref) (lambda - lambda_ref) deg`` and
``Y = R (phi - phi_ref) deg``. The cosine factor uses the one reference latitude
``phi_ref`` for every point, not a per-point ``cos(phi)``. The metric only
converts lon/lat separations to meters; advected separations come from the
ingested positions, not from ``R`` and ``phi``. Both flow-map subclasses use the
module-level :func:`_to_meters`.
"""

from __future__ import annotations

import abc
from typing import Self

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in meters, used for the local-tangent meters convention."""

_DEG = np.pi / 180.0
"""Degrees-to-radians factor for the local-tangent meters convention."""


def _lonlat_to_meters(lon, lat, lon_ref: float, lat_ref: float):
    """Project lon/lat (degrees) into the local-tangent meters frame.

    Uses the single reference latitude ``lat_ref`` for the cosine factor
    (``X = R cos(phi_ref) (lambda - lambda_ref) deg``, ``Y = R (phi - phi_ref) deg``).
    Works on plain arrays or xarray objects.
    """
    c = np.cos(lat_ref * _DEG)
    x = EARTH_RADIUS_M * c * (lon - lon_ref) * _DEG
    y = EARTH_RADIUS_M * (lat - lat_ref) * _DEG
    return x, y


def _reference_lonlat(lon_0: xr.DataArray, lat_0: xr.DataArray) -> tuple[float, float]:
    """The single reference point ``(lon_ref, lat_ref)`` = means of ``lon_0``/``lat_0``."""
    return float(lon_0.mean()), float(lat_0.mean())


def _to_meters(lon: xr.DataArray, lat: xr.DataArray, lon_0: xr.DataArray, lat_0: xr.DataArray):
    """Project ``lon``/``lat`` into the meters frame anchored at the centroid of
    ``lon_0``/``lat_0`` (one ``cos(phi_ref)``, not a per-point cosine).
    """
    lon_ref, lat_ref = _reference_lonlat(lon_0, lat_0)
    return _lonlat_to_meters(lon, lat, lon_ref, lat_ref)


def _assemble_tensor(
    fxx: xr.DataArray,
    fxy: xr.DataArray,
    fyx: xr.DataArray,
    fyy: xr.DataArray,
) -> xr.DataArray:
    """Pack four scalar ``(i, j)`` component fields into a ``(row, col)`` tensor.

    ``row`` and ``col`` become dimension coordinates valued ``['x', 'y']`` with
    ``tensor.sel(row=a, col=b)`` the ``(a, b)`` component, e.g.
    ``gradF.sel(row='y', col='x') = dF_y / dx0_x``. The fields are renamed to a
    common name so :func:`xarray.concat` does not drop one.
    """
    fxx, fxy = fxx.rename("tensor"), fxy.rename("tensor")
    fyx, fyy = fyx.rename("tensor"), fyy.rename("tensor")
    row_x = xr.concat([fxx, fxy], dim="col")
    row_y = xr.concat([fyx, fyy], dim="col")
    tensor = xr.concat([row_x, row_y], dim="row")
    return tensor.assign_coords(row=["x", "y"], col=["x", "y"])


class Seed(abc.ABC):
    """Time-free wrapper around an ``xr.Dataset`` of seed positions.

    Holds only the reference release positions ``lon_0``/``lat_0`` (``x_0``,
    degrees, coordinates) and -- for :class:`AuxiliarySeed` -- the per-arm
    stencil geometry and grid-point centres. No ``t0``, no ``T``, no advected
    ``lon``/``lat``, no data variables; time and advected positions enter at
    ingest via :meth:`pset_to_flowmap`, which produces a :class:`FlowMap`. The
    dataset is held in :attr:`ds`.

    Logical grid dims are ``i, j``; the 2-D ``lon_0``/``lat_0`` can represent
    curvilinear grids. For :class:`NeighborSeed` the grid point is the
    release/diagnostic location, so ``lon_0``/``lat_0`` are on ``(i, j)``; for
    :class:`AuxiliarySeed` they carry an extra ``displacement`` dim and the
    centres ``lon_c``/``lat_c`` are on ``(i, j)``. Concrete subclasses differ
    only in the stencil laid down in :meth:`from_axes`.

    Attributes
    ----------
    ds : xr.Dataset
        Reference release positions ``lon_0``/``lat_0`` (``x_0``) as coordinates,
        plus any per-class auxiliary geometry. No data variables, no ``t0``/``T``.
    """

    #: The paired :class:`FlowMap` subclass produced by :meth:`pset_to_flowmap`.
    _flowmap_cls: type[FlowMap]

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing time-free dataset of seed positions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j`` and reference coordinates ``lon_0``,
            ``lat_0`` (degrees). Subclasses may require additional
            coordinates/dimensions. No data variables, no ``t0``/``T``.
        """
        self.ds = ds

    @classmethod
    @abc.abstractmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a time-free seed from 1-D lon/lat axes.

        The 1-D axes are broadcast into 2-D reference fields ``lon_0``/``lat_0``
        (degrees). No time is recorded; ``t0`` and ``T`` enter only at
        :meth:`pset_to_flowmap`.

        Parameters
        ----------
        lon : np.ndarray
            1-D array of longitudes (degrees), length ``Ni``, mapped to dim ``i``.
        lat : np.ndarray
            1-D array of latitudes (degrees), length ``Nj``, mapped to dim ``j``.

        Returns
        -------
        Self
            A seed whose ``.ds`` carries the reference ``lon_0``/``lat_0`` (and
            any per-class auxiliary geometry) as coordinates, with no data
            variables and no time.
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

    def pset_to_flowmap(self, lon, lat, *, t0, t1) -> FlowMap:
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
            the direction (``t1 < t0`` backward/attracting; ``t1 > t0``
            forward/repelling). ``t1`` is not stored (recoverable as ``t0 + T``).

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
        ds = self.ds.assign(lon=lon_arm, lat=lat_arm).assign_coords(t0=t0, T=T)
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
    ``(i, j, displacement)`` for :class:`AuxiliaryFlowMap` (with diagnostic
    centres ``lon_c``/``lat_c`` on ``(i, j)``). Concrete subclasses differ only
    in how ``grad F`` is finite-differenced.

    Attributes
    ----------
    ds : xr.Dataset
        Reference positions ``lon_0``/``lat_0`` (``x_0``), advected positions
        ``lon``/``lat`` (``F(x_0)``), the scalar release time ``t0``, the signed
        window ``T = t1 - t0`` (``timedelta64``), and any per-class auxiliary
        state.
    """

    #: The paired :class:`Seed` subclass produced by :meth:`to_seed`.
    _seed_cls: type[Seed]

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing advected dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j``, reference coordinates ``lon_0``/``lat_0``,
            advected data variables ``lon``/``lat`` (degrees), and scalar ``t0``
            and signed ``T`` coordinates. Subclasses may require additional
            variables/dimensions.
        """
        self.ds = ds

    def _integration_seconds(self) -> float:
        """``|T|`` in seconds from the stored signed window ``T`` (``timedelta64``)."""
        return float(np.abs(self.ds["T"] / np.timedelta64(1, "s")))

    @abc.abstractmethod
    def deformation_gradient(self) -> xr.DataArray:
        """Deformation gradient grad F of the flow map. Haller (2015) Eq. 9.

        The 2x2 tensor ``grad F = d(lon, lat) / d(lon_0, lat_0)`` per grid point,
        finite-differenced as ``(advected separation) / (initial separation)``,
        both taken in the meters frame (:func:`_to_meters`): the denominator from
        the reference ``lon_0``/``lat_0``, the numerator from the advected
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
        return C.rename(col="row", col_b="col")

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
        lam = lam.assign_coords(eig=[0, 1])
        vec = vec.assign_coords(comp=["x", "y"], eig=[0, 1])
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
        lambda_max = self.cg_eigen()["lambda"].isel(eig=1)
        t_sec = self._integration_seconds()
        # (1 / |T|) log sqrt(lambda_max) = (1 / |T|) * 0.5 * log(lambda_max).
        return (1.0 / t_sec) * 0.5 * np.log(lambda_max)

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

    The release point is the grid point itself, so ``.ds`` carries only the
    reference positions ``lon_0``/``lat_0`` on ``(i, j)`` -- no displacement dim.
    The paired :class:`NeighborFlowMap` differences ``grad F`` against
    neighbouring grid points ``(i +/- 1, j +/- 1)``, coupling the diagnostic
    resolution to the seed grid resolution. This is the SPASSO approach; see
    ``src/Diagnostics.py`` at https://github.com/OceanCruises/SPASSO.
    """

    @classmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a neighbour-stencil seed from 1-D lon/lat axes.

        See :meth:`Seed.from_axes`. The 1-D axes are broadcast into axis-aligned
        (rectilinear) 2-D reference fields ``lon_0(i, j)`` / ``lat_0(i, j)`` (lon
        varies along ``i``, lat along ``j``) -- the layout the paired
        :class:`NeighborFlowMap` gradient assumes.
        """
        # Broadcast the 1-D axes into axis-aligned 2-D fields on (i, j) with the
        # high-level API; lon varies along i, lat along j.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon2d, lat2d = xr.broadcast(lon_axis, lat_axis)

        ds = xr.Dataset(
            coords={
                "i": np.arange(lon_axis.sizes["i"]),
                "j": np.arange(lat_axis.sizes["j"]),
                # Reference initial positions x_0; the grid diagnostics live here.
                "lon_0": lon2d,
                "lat_0": lat2d,
            },
        )
        return cls(ds)


class AuxiliarySeed(Seed):
    """Time-free seed with a fixed four-arm auxiliary displacement stencil.

    The reference release positions ``lon_0(i, j, displacement)`` /
    ``lat_0(i, j, displacement)`` (coordinates, degrees) are the explicit per-arm
    positions -- exactly what :meth:`Seed.to_parcels_pset` emits, so the dataset
    is self-sufficient. The grid-point centres ``lon_c(i, j)`` / ``lat_c(i, j)``
    (coordinates) -- the points on which the diagnostics are reported -- are kept
    explicitly. There is no stored ``dx``/``dy``: the stencil lives in the
    reference positions themselves.

    Each grid point carries four arms ``east, north, west, south`` at offsets
    ``east = (+s, 0)``, ``north = (0, +s)``, ``west = (-s, 0)``,
    ``south = (0, -s)`` for ``s = aux_separation_m`` -- no centre point, no
    diagonals. The arms are placed in the single grid local-tangent meters frame
    (see :func:`_to_meters`), decoupling the gradient step from the seed grid
    resolution. The paired :class:`AuxiliaryFlowMap` differences ``grad F``
    across the four arms (east-west, north-south).
    """

    @classmethod
    def from_axes(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        aux_separation_m: float = 1_000.0,
    ) -> Self:
        """Build an auxiliary-stencil seed from 1-D lon/lat axes.

        See :meth:`Seed.from_axes`. The grid-point centres ``lon_c`` / ``lat_c``
        are placed on ``(i, j)`` from the axes, then the four-arm
        ``displacement = ['east', 'north', 'west', 'south']`` stencil is laid out
        around each centre at offsets ``east = (+s, 0)``, ``north = (0, +s)``,
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
        """
        # Broadcast the 1-D axes into curvilinear 2-D centre fields on (i, j);
        # lon varies along i, lat along j. These are the diagnostic centres.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon_c, lat_c = xr.broadcast(lon_axis, lat_axis)

        # Four-arm stencil offsets in meters.
        s = float(aux_separation_m)
        displacement = ["east", "north", "west", "south"]
        off_x = xr.DataArray(
            [+s, 0.0, -s, 0.0], dims="displacement", coords={"displacement": displacement}
        )
        off_y = xr.DataArray(
            [0.0, +s, 0.0, -s], dims="displacement", coords={"displacement": displacement}
        )

        # Place the arms in the single grid reference frame (reference latitude =
        # grid centroid), matching _to_meters. lon_c (i, j) broadcasts with the
        # (displacement,) offset into the arm positions (i, j, displacement).
        lat_ref = float(lat_c.mean())
        c = np.cos(lat_ref * _DEG)
        lon_0 = lon_c + off_x / (EARTH_RADIUS_M * c * _DEG)
        lat_0 = lat_c + off_y / (EARTH_RADIUS_M * _DEG)

        ds = xr.Dataset(
            coords={
                "i": np.arange(lon_axis.sizes["i"]),
                "j": np.arange(lat_axis.sizes["j"]),
                "displacement": displacement,
                # Explicit per-arm reference release positions x_0.
                "lon_0": lon_0,
                "lat_0": lat_0,
                # Diagnostic grid-point centres (no displacement dim).
                "lon_c": lon_c,
                "lat_c": lat_c,
            },
        )
        return cls(ds)


class NeighborFlowMap(FlowMap):
    """Advected flow map whose stencil is the neighbouring grid points.

    ``.ds`` carries the reference and advected positions on ``(i, j)`` (no
    displacement dim). ``grad F`` is differenced against neighbouring grid points
    ``(i +/- 1, j +/- 1)``, so boundary cells lacking a neighbour are NaN. See
    Haller (2015) Eq. 9 and the paired :class:`NeighborSeed`.

    Axis-aligned grids only: the neighbour gradient divides each tensor column by
    a single axis step and drops the off-diagonal metric terms
    (``d lon_0 / d j``, ``d lat_0 / d i``), so it is correct only when ``lon_0``
    varies along ``i`` and ``lat_0`` along ``j``. :meth:`Seed.from_axes` always
    builds such a grid, but :meth:`FlowMap.__init__` accepts any dataset, so the
    limitation is latent. Use :class:`AuxiliaryFlowMap` for curvilinear grids.
    """

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced against neighbouring grid points ``(i +/- 1, j +/- 1)``.

        Numerator: the separation of the advected neighbour positions;
        denominator: the initial neighbour separation, both in the meters frame
        (:func:`_to_meters`). Boundary cells lacking a neighbour yield NaN. Each
        column is divided by a single axis step (``dx0`` along ``i``, ``dy0``
        along ``j``), so this assumes an axis-aligned grid (see the class
        docstring). See :meth:`FlowMap.deformation_gradient`.
        """
        x_adv, y_adv = _to_meters(
            self.ds["lon"], self.ds["lat"], self.ds["lon_0"], self.ds["lat_0"]
        )
        x_ref, y_ref = _to_meters(
            self.ds["lon_0"], self.ds["lat_0"], self.ds["lon_0"], self.ds["lat_0"]
        )

        def central_diff(field: xr.DataArray, dim: str) -> xr.DataArray:
            # Neighbour difference (index + 1) - (index - 1); .shift fills NaN
            # past both ends, so the domain edges are legitimately NaN.
            return field.shift({dim: -1}) - field.shift({dim: +1})

        # lon_0 varies along i, lat_0 along j, so these are the pure x- and y-
        # reference steps in meters.
        dx0 = central_diff(x_ref, "i")
        dy0 = central_diff(y_ref, "j")
        return _assemble_tensor(
            fxx=central_diff(x_adv, "i") / dx0,
            fxy=central_diff(x_adv, "j") / dy0,
            fyx=central_diff(y_adv, "i") / dx0,
            fyy=central_diff(y_adv, "j") / dy0,
        )


class AuxiliaryFlowMap(FlowMap):
    """Advected flow map differenced across the fixed four-arm auxiliary stencil.

    ``.ds`` carries the reference and advected arm positions on
    ``(i, j, displacement)`` plus the diagnostic centres ``lon_c``/``lat_c`` on
    ``(i, j)``. The per-point stencil makes ``grad F`` well-defined at every grid
    point, including the boundary. See Haller (2015) Eq. 9 and the paired
    :class:`AuxiliarySeed`.
    """

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced across the four-arm auxiliary stencil.

        ``grad F = d(lon, lat) / d(lon_0, lat_0)`` over the ``displacement`` dim:
        east minus west for the ``x`` derivative, north minus south for ``y``.
        Both the advected separation (numerator) and the reference arm separation
        (denominator, the ``2s`` span) are read in the meters frame
        (:func:`_to_meters`). Well-defined at every grid point, including the
        boundary. See :meth:`FlowMap.deformation_gradient`.
        """
        x_adv, y_adv = _to_meters(
            self.ds["lon"], self.ds["lat"], self.ds["lon_0"], self.ds["lat_0"]
        )
        x_ref, y_ref = _to_meters(
            self.ds["lon_0"], self.ds["lat_0"], self.ds["lon_0"], self.ds["lat_0"]
        )

        def arm_diff(field: xr.DataArray, hi: str, lo: str) -> xr.DataArray:
            # Difference two opposing arms; the scalar `displacement` label is
            # dropped on subtraction so the result is back on (i, j).
            return field.sel(displacement=hi) - field.sel(displacement=lo)

        den_x = arm_diff(x_ref, "east", "west")
        den_y = arm_diff(y_ref, "north", "south")
        return _assemble_tensor(
            fxx=arm_diff(x_adv, "east", "west") / den_x,
            fxy=arm_diff(x_adv, "north", "south") / den_y,
            fyx=arm_diff(y_adv, "east", "west") / den_x,
            fyy=arm_diff(y_adv, "north", "south") / den_y,
        )


# --- paired seed <-> flow-map wiring ---------------------------------------
#
# Each concrete seed knows the flow map it produces, and each flow map knows the
# seed it collapses back to; explicit class attributes, not inheritance.
NeighborSeed._flowmap_cls = NeighborFlowMap
AuxiliarySeed._flowmap_cls = AuxiliaryFlowMap
NeighborFlowMap._seed_cls = NeighborSeed
AuxiliaryFlowMap._seed_cls = AuxiliarySeed
