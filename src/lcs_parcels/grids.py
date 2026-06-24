"""Particle-grid classes for Lagrangian coherent structure (LCS) diagnostics.

This module is the diagnostic layer that sits on top of trajectory integration.
It contains no Parcels code: it only emits particle sets and ingests their
advected positions. Conventions and notation follow Haller (2015),
*Lagrangian Coherent Structures*, Annu. Rev. Fluid Mech. 47:137-162,
doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322).

The two finite-difference strategies for the deformation gradient grad F are
modeled as two explicit subclasses, NeighborGrid and AuxiliaryGrid, rather than
inferred at runtime from the dataset dimensions. Each class is a composition
wrapper around an xr.Dataset (held in ``.ds``); none subclasses xr.Dataset.

Sphere metric convention
------------------------
Haller's math is Cartesian; our grids are lon/lat. The deformation gradient is
formed from separations in a local tangent frame in meters, using
``dx = R cos(phi) dlambda`` and ``dy = R dphi`` where ``R`` is the Earth radius,
``phi`` is latitude (radians) and ``lambda`` is longitude (radians). In the
tiny-separation regime a flat-tangent ``cos(phi)`` approximation is adequate.
"""

from __future__ import annotations

import abc
from typing import Self

import numpy as np
import xarray as xr

EARTH_RADIUS_M = 6_371_000.0
"""Mean Earth radius in meters, used for the local-tangent meters convention."""


class ParticleGrid(abc.ABC):
    """Composition wrapper around an ``xr.Dataset`` of seed positions.

    The wrapped dataset is held in :attr:`ds` (this class does *not* subclass
    ``xr.Dataset``, which xarray discourages). The dataset has logical grid
    dimensions ``i, j`` and data variables ``lon(i, j)`` and ``lat(i, j)``
    (degrees). Two-dimensional lon/lat support curvilinear / non-rectangular
    grids. These positions are the initial conditions ``x_0`` on which all
    diagnostics are defined.

    Concrete subclasses differ only in how the deformation gradient grad F is
    finite-differenced (see :class:`NeighborGrid` and :class:`AuxiliaryGrid`).

    All internal access uses the high-level, label-based xarray API (``.isel``,
    ``.sel``, named dims, broadcasting, ``.where``), never positional indexing.

    Attributes
    ----------
    ds : xr.Dataset
        Seed positions and any per-class auxiliary state. After ingesting
        advected positions, also carries the integration time ``T`` (seconds,
        including sign) needed for the FTLE.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing dataset of seed positions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j`` and variables ``lon(i, j)``,
            ``lat(i, j)`` in degrees. Subclasses may require additional
            variables/dimensions (see their docstrings).
        """
        self.ds = ds

    @classmethod
    @abc.abstractmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a seed grid from 1-D lon/lat axes.

        The 1-D axes are broadcast into curvilinear 2-D fields ``lon(i, j)`` and
        ``lat(i, j)`` (degrees), so that downstream code never special-cases
        rectangular grids.

        Parameters
        ----------
        lon : np.ndarray
            1-D array of longitudes (degrees), length ``Ni``, mapped to dim ``i``.
        lat : np.ndarray
            1-D array of latitudes (degrees), length ``Nj``, mapped to dim ``j``.

        Returns
        -------
        Self
            A grid whose ``.ds`` has ``lon`` and ``lat`` with dims ``(i, j)`` and
            shape ``(Ni, Nj)``.
        """
        raise NotImplementedError("from_axes is not implemented (scaffolding only).")

    @abc.abstractmethod
    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten seed positions to plain ``(lon, lat)`` lists for Parcels.

        Stacks the grid over the particle dimension(s) into a single
        ``particle`` index (``.stack(particle=('i', 'j'))`` for
        :class:`NeighborGrid`, additionally over ``('di', 'dj')`` for
        :class:`AuxiliaryGrid`) and returns plain Python lists. The ``particle``
        MultiIndex is the lossless inverse used by
        :meth:`from_parcels_pset_lon_lat` to reattach advected positions.

        Returns
        -------
        tuple[list[float], list[float]]
            ``(lon, lat)`` as flat lists of degrees, one entry per particle.
        """
        raise NotImplementedError(
            "to_parcels_pset is not implemented (scaffolding only)."
        )

    @classmethod
    @abc.abstractmethod
    def from_parcels_pset_lon_lat(
        cls, seed: "ParticleGrid", lon, lat, *, T: float
    ) -> Self:
        """Reattach advected flat lon/lat onto a seed grid and record ``T``.

        Inverse of :meth:`to_parcels_pset`: the flat advected positions are
        attached to the ``particle`` MultiIndex of ``seed`` and unstacked back to
        the grid dims. Lost particles arrive as NaN and propagate naturally.
        Multiple release times ``t0`` and integration times ``T`` are just extra
        broadcast dimensions handled by xarray.

        Parameters
        ----------
        seed : ParticleGrid
            The grid that produced the particle set; supplies the ``particle``
            MultiIndex used to unstack the flat results back to ``(i, j)``
            (and ``(di, dj)`` for :class:`AuxiliaryGrid`).
        lon, lat : array-like
            Advected longitudes/latitudes (degrees), aligned with the order of
            :meth:`to_parcels_pset` output.
        T : float
            Integration time in seconds, including sign (the particle set owns
            integration direction). Recorded for use by :meth:`ftle`.

        Returns
        -------
        Self
            A grid whose ``.ds`` holds the advected positions on the original
            grid and records ``T``.
        """
        raise NotImplementedError(
            "from_parcels_pset_lon_lat is not implemented (scaffolding only)."
        )

    @abc.abstractmethod
    def deformation_gradient(self) -> xr.DataArray:
        """Deformation gradient grad F of the flow map. Haller (2015) Eq. 9.

        The 2x2 tensor ``grad F = d x(t1) / d x_0`` per grid point, finite-
        differenced in the local-tangent meters frame
        (``dx = R cos(phi) dlambda``, ``dy = R dphi``). Subclasses define the
        stencil: neighboring grid points (:class:`NeighborGrid`) or a per-point
        auxiliary displacement grid (:class:`AuxiliaryGrid`). Cells with a
        missing stencil point yield NaN.

        Returns
        -------
        xr.DataArray
            grad F with dims ``(i, j, row, col)`` and a component coordinate
            ``comp = ['x', 'y']`` labeling both ``row`` and ``col``
            (dimensionless; meters / meters).
        """
        raise NotImplementedError(
            "deformation_gradient is not implemented (scaffolding only)."
        )

    def cauchy_green(self) -> xr.DataArray:
        """Right Cauchy-Green strain tensor ``C = (grad F)^T grad F``.

        Haller (2015) Eq. 6. Symmetric positive-(semi)definite 2x2 tensor per
        grid point, built from :meth:`deformation_gradient`.

        Returns
        -------
        xr.DataArray
            ``C`` with dims ``(i, j, row, col)`` and component coordinate
            ``comp = ['x', 'y']`` (dimensionless).
        """
        raise NotImplementedError(
            "cauchy_green is not implemented (scaffolding only)."
        )

    def cg_eigen(self) -> xr.Dataset:
        """Eigen-decomposition of the Cauchy-Green tensor ``C``.

        Haller (2015) Eq. 7. Solves ``C xi_i = lambda_i xi_i`` with
        ``0 < lambda_1 <= lambda_2`` and orthonormal eigenvectors
        ``xi_1 perp xi_2``. Implementable either via
        ``xr.apply_ufunc(np.linalg.eigh, C, input_core_dims=[['row', 'col']], ...)``
        or a closed-form 2x2 symmetric solver in pure xarray arithmetic (xarray
        has no native eig); see ``docs/notation.md``.

        Returns
        -------
        xr.Dataset
            Variables ``lambda`` with dims ``(i, j, eig)`` (eigenvalues in
            ascending order, dimensionless) and ``xi`` with dims
            ``(i, j, comp, eig)`` (orthonormal eigenvectors, component coordinate
            ``comp = ['x', 'y']``).
        """
        raise NotImplementedError("cg_eigen is not implemented (scaffolding only).")

    def ftle(self) -> xr.DataArray:
        """Finite-time Lyapunov exponent (FTLE). Haller (2015) Sec. 4.1.

        Computes ``Lambda = (1 / |T|) * log(sqrt(lambda_max))`` using the
        *largest* eigenvalue of ``C`` from :meth:`cg_eigen` and the recorded
        integration time ``T`` (seconds).

        Returns
        -------
        xr.DataArray
            FTLE field with dims ``(i, j)`` in units of 1/second.
        """
        raise NotImplementedError("ftle is not implemented (scaffolding only).")


class NeighborGrid(ParticleGrid):
    """Stencil = neighboring grid points ``(i +/- 1, j +/- 1)``.

    The deformation gradient is differenced against neighboring seed grid points;
    no auxiliary displacement grid is added, so ``.ds`` carries no extra
    dimensions beyond ``i, j``. This couples the diagnostic resolution to the
    seed grid resolution. See Haller (2015) Eq. 9 for the deformation gradient.
    """

    @classmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build a neighbor-stencil seed grid from 1-D lon/lat axes.

        See :meth:`ParticleGrid.from_axes`.
        """
        raise NotImplementedError("from_axes is not implemented (scaffolding only).")

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten seed positions over ``('i', 'j')``.

        See :meth:`ParticleGrid.to_parcels_pset`.
        """
        raise NotImplementedError(
            "to_parcels_pset is not implemented (scaffolding only)."
        )

    @classmethod
    def from_parcels_pset_lon_lat(
        cls, seed: "ParticleGrid", lon, lat, *, T: float
    ) -> Self:
        """Reattach advected lon/lat onto the ``('i', 'j')`` MultiIndex.

        See :meth:`ParticleGrid.from_parcels_pset_lon_lat`.
        """
        raise NotImplementedError(
            "from_parcels_pset_lon_lat is not implemented (scaffolding only)."
        )

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced against neighboring grid points ``(i +/- 1, j +/- 1)``.

        Separations are taken in the local-tangent meters frame
        (``dx = R cos(phi) dlambda``, ``dy = R dphi``). Boundary cells lacking a
        neighbor yield NaN. See :meth:`ParticleGrid.deformation_gradient`.
        """
        raise NotImplementedError(
            "deformation_gradient is not implemented (scaffolding only)."
        )


class AuxiliaryGrid(ParticleGrid):
    """Stencil = per-point auxiliary displacement grid. Haller (2015) Eq. 9.

    Adds a per-point displacement stencil with dims ``(di, dj)`` and
    displacement variables ``dx(di, dj)`` and ``dy(di, dj)`` in **meters**
    (Haller's auxiliary grid). This decouples the diagnostic / gradient step from
    the seed grid resolution. The displacements are applied in the local-tangent
    meters frame (``dx = R cos(phi) dlambda``, ``dy = R dphi``) to produce the
    auxiliary seed positions surrounding each grid point.
    """

    @classmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray) -> Self:
        """Build an auxiliary-stencil seed grid from 1-D lon/lat axes.

        See :meth:`ParticleGrid.from_axes`. Concrete implementations also
        populate the ``(di, dj)`` displacement stencil (``dx``, ``dy`` in
        meters).
        """
        raise NotImplementedError("from_axes is not implemented (scaffolding only).")

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten seed positions over ``('i', 'j', 'di', 'dj')``.

        See :meth:`ParticleGrid.to_parcels_pset`.
        """
        raise NotImplementedError(
            "to_parcels_pset is not implemented (scaffolding only)."
        )

    @classmethod
    def from_parcels_pset_lon_lat(
        cls, seed: "ParticleGrid", lon, lat, *, T: float
    ) -> Self:
        """Reattach advected lon/lat onto the ``('i', 'j', 'di', 'dj')`` MultiIndex.

        See :meth:`ParticleGrid.from_parcels_pset_lon_lat`.
        """
        raise NotImplementedError(
            "from_parcels_pset_lon_lat is not implemented (scaffolding only)."
        )

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced across the per-point auxiliary stencil.

        Uses the auxiliary displacements ``dx(di, dj)``, ``dy(di, dj)`` (meters)
        in the local-tangent frame to finite-difference the flow map at each grid
        point, decoupling the gradient step from the seed grid spacing. Points
        with a missing auxiliary neighbor yield NaN. See
        :meth:`ParticleGrid.deformation_gradient` and Haller (2015) Eq. 9.
        """
        raise NotImplementedError(
            "deformation_gradient is not implemented (scaffolding only)."
        )
