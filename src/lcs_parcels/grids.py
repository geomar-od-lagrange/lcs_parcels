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

Timing convention
-----------------
A seed grid *owns* its release time ``t0`` (recorded by ``from_axes``). Ingest
(``from_parcels_pset_lon_lat``) is given only the end time ``t1``; the signed
integration window ``T = t1 - t0`` is derived and stored. Direction is implied
by ``sign(T)`` (``t1 < t0`` → backward → attracting LCS; ``t1 > t0`` → forward →
repelling LCS); there is no separate flag and ``t1`` itself is not stored. See
``plans/timing-design.md``.

Position convention
-------------------
Every grid carries two position pairs on ``('i', 'j')``: the *reference* initial
positions ``lon_0``/``lat_0`` (the initial conditions ``x_0``, held as
coordinates) and the *advected* positions ``lon``/``lat`` (the flow map
``F_{t0}^{t1}(x_0)``, i.e. where the particle actually is, held as data
variables). ``from_axes`` seeds both equal -- the identity
``F_{t0}^{t0}(x_0) = x_0`` -- and ``from_parcels_pset_lon_lat`` overwrites only
the advected ``lon``/``lat`` with the ingested Parcels output. The deformation
gradient is then ``grad F = d(lon, lat) / d(lon_0, lat_0)`` -- advected
separations (numerator) over reference separations (denominator) -- so both pairs
must live on the same grid.

Sphere metric convention
------------------------
Haller's math is Cartesian; our grids are lon/lat. Separations are formed in a
local tangent frame in meters, using ``dx = R cos(phi) dlambda`` and
``dy = R dphi`` where ``R`` is the Earth radius, ``lambda`` is longitude and
``phi`` is latitude (both radians). In the tiny-separation regime a flat-tangent
``cos(phi)`` approximation is adequate. This metric only converts lon/lat
separations to meters; the *advected* separations that form the numerator of
grad F are measured from the ingested Parcels output positions, not recomputed
analytically from ``R`` and ``phi``.
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
    dimensions ``i, j``; the reference initial positions ``lon_0(i, j)`` and
    ``lat_0(i, j)`` (degrees) are coordinates, and the advected positions
    ``lon(i, j)`` and ``lat(i, j)`` are data variables. Two-dimensional lon/lat
    support curvilinear / non-rectangular grids. The reference positions are the
    initial conditions ``x_0`` on which all diagnostics are defined; the advected
    positions are the flow map ``F_{t0}^{t1}(x_0)`` (see the module-level
    *Position convention*).

    Concrete subclasses differ only in how the deformation gradient grad F is
    finite-differenced (see :class:`NeighborGrid` and :class:`AuxiliaryGrid`).

    All internal access uses the high-level, label-based xarray API (``.isel``,
    ``.sel``, named dims, broadcasting, ``.where``), never positional indexing.

    Attributes
    ----------
    ds : xr.Dataset
        Reference initial positions ``lon_0``/``lat_0`` (``x_0``), advected
        positions ``lon``/``lat`` (``F(x_0)``), the release time ``t0``
        (recorded at seeding), the signed integration window ``T = t1 - t0``
        (``timedelta64``; zero on a seed, derived from the ingest end time ``t1``
        and needed for the FTLE), and any per-class auxiliary state.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """Wrap an existing dataset of seed positions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with dims ``i, j``, reference coordinates ``lon_0(i, j)``,
            ``lat_0(i, j)`` and advected data variables ``lon(i, j)``,
            ``lat(i, j)`` (degrees), plus the release time ``t0`` and signed
            window ``T``. Subclasses may require additional variables/dimensions
            (see their docstrings).
        """
        self.ds = ds

    @classmethod
    @abc.abstractmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray, *, t0: np.datetime64) -> Self:
        """Build a seed grid from 1-D lon/lat axes, released at ``t0``.

        The 1-D axes are broadcast into curvilinear 2-D reference fields
        ``lon_0(i, j)`` and ``lat_0(i, j)`` (degrees), so that downstream code
        never special-cases rectangular grids. The advected ``lon``/``lat`` are
        seeded equal to the reference (the identity flow map
        ``F_{t0}^{t0}(x_0) = x_0``) and the signed window ``T`` is zero until
        ingest. The release time ``t0`` is recorded on ``.ds`` so the grid
        *owns* its own ``t0``; ingest then needs only the end time ``t1`` (see
        :meth:`from_parcels_pset_lon_lat` and ``plans/timing-design.md``).

        Parameters
        ----------
        lon : np.ndarray
            1-D array of longitudes (degrees), length ``Ni``, mapped to dim ``i``.
        lat : np.ndarray
            1-D array of latitudes (degrees), length ``Nj``, mapped to dim ``j``.
        t0 : np.datetime64
            Release time of the seed positions (scalar, or array for an ensemble
            of releases). Recorded as a coordinate on ``.ds``.

        Returns
        -------
        Self
            A grid whose ``.ds`` has reference ``lon_0``/``lat_0`` and advected
            ``lon``/``lat`` with dims ``(i, j)`` and shape ``(Ni, Nj)``, and
            records ``t0`` and ``T = 0``.
        """
        raise NotImplementedError("from_axes is not implemented (scaffolding only).")

    @abc.abstractmethod
    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten the reference seed positions ``x_0`` to plain ``(lon, lat)``
        lists for Parcels.

        Emits the *reference* positions ``lon_0``/``lat_0`` (where particles are
        released), not the advected positions. Stacks the grid over the particle
        dimension(s) into a single ``particle`` index
        (``.stack(particle=('i', 'j'))`` for :class:`NeighborGrid`, additionally
        over ``('displacement',)`` for :class:`AuxiliaryGrid`) and returns plain
        Python lists. The ``particle`` MultiIndex is the lossless inverse used by
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
        cls,
        seed: "ParticleGrid",
        lon,
        lat,
        *,
        t1: np.datetime64,
    ) -> Self:
        """Reattach advected flat lon/lat onto a seed grid and record ``t0``, ``T``.

        Inverse of :meth:`to_parcels_pset`: the flat advected positions are
        attached to the ``particle`` MultiIndex of ``seed`` as the advected
        ``lon``/``lat`` and unstacked back to the grid dims; the reference
        ``lon_0``/``lat_0`` carried from ``seed`` are left untouched.
        Lost particles arrive as NaN and propagate naturally.
        Multiple release times ``t0`` and integration windows ``T`` are just
        extra broadcast dimensions handled by xarray. See
        ``plans/timing-design.md``.

        The seed already owns its release time ``t0`` (recorded by
        :meth:`from_axes`), so only the end time ``t1`` is supplied here; the
        signed window ``T = t1 - t0`` is derived and stored. The caller never
        passes ``T`` directly.

        Parameters
        ----------
        seed : ParticleGrid
            The grid that produced the particle set; supplies the ``particle``
            MultiIndex used to unstack the flat results back to ``(i, j)`` (and
            the ``displacement`` arm for :class:`AuxiliaryGrid`), and the
            release time ``t0``.
        lon, lat : array-like
            Advected longitudes/latitudes (degrees), aligned with the order of
            :meth:`to_parcels_pset` output.
        t1 : np.datetime64
            End time of the integration (scalar, or array for an ensemble). The
            signed window ``T = t1 - seed.t0`` sets the integration direction
            (``t1 < t0`` → backward → attracting LCS; ``t1 > t0`` → forward →
            repelling LCS) and is recorded as a coordinate, used by
            :meth:`ftle`. ``t1`` itself is not stored.

        Returns
        -------
        Self
            A grid whose ``.ds`` holds the advected positions as ``lon``/``lat``
            on the original grid and records ``t0`` and the derived signed ``T``
            as coordinates.
        """
        raise NotImplementedError(
            "from_parcels_pset_lon_lat is not implemented (scaffolding only)."
        )

    @abc.abstractmethod
    def deformation_gradient(self) -> xr.DataArray:
        """Deformation gradient grad F of the flow map. Haller (2015) Eq. 9.

        The 2x2 tensor ``grad F = d(lon, lat) / d(lon_0, lat_0)`` per grid point,
        estimated by finite differences as
        ``(advected separation) / (initial separation)``:

        - **denominator** — the *initial* separation of the reference
          ``lon_0``/``lat_0`` between stencil points, a controlled quantity taken
          in the local-tangent meters frame (``dx = R cos(phi) dlambda``,
          ``dy = R dphi``; ``lambda`` = longitude, ``phi`` = latitude);
        - **numerator** — the corresponding separation of the *advected*
          positions ``lon``/``lat`` (the ingested Parcels outputs, converted to
          meters with the same metric). It is **not** recomputed analytically
          from ``R`` and ``phi``; only the advected positions carry the flow-map
          information.

        Subclasses define the stencil: neighboring grid points
        (:class:`NeighborGrid`) or the fixed four-arm auxiliary stencil
        (:class:`AuxiliaryGrid`). Cells with a missing stencil point yield NaN.

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
        signed integration window ``T`` (``timedelta64``, derived at ingest as
        ``t1 - t0`` and converted to seconds; the conversion must be
        calendar-aware for non-standard model calendars).

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

    This is the SPASSO / d'Ovidio approach to FTLE: particles are seeded once on
    a single regular grid (one per output cell) and the flow-map gradient is
    taken by neighbour-differencing those positions (``np.gradient`` over the
    grid), with no controlled auxiliary separation. See SPASSO,
    ``src/Diagnostics.py`` (https://github.com/OceanCruises/SPASSO).
    """

    @classmethod
    def from_axes(cls, lon: np.ndarray, lat: np.ndarray, *, t0: np.datetime64) -> Self:
        """Build a neighbor-stencil seed grid from 1-D lon/lat axes.

        See :meth:`ParticleGrid.from_axes`.
        """
        # Broadcast the 1-D axes into curvilinear 2-D fields on (i, j) with the
        # high-level API; lon varies along i, lat along j.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon2d, lat2d = xr.broadcast(lon_axis, lat_axis)

        t0 = np.datetime64(t0)
        ds = xr.Dataset(
            data_vars={
                # Advected positions F(x_0). At seeding the flow map is the
                # identity F_{t0}^{t0}(x_0) = x_0, so they equal the reference.
                "lon": lon2d,
                "lat": lat2d,
            },
            coords={
                "i": np.arange(lon_axis.sizes["i"]),
                "j": np.arange(lat_axis.sizes["j"]),
                # Reference initial positions x_0; the grid diagnostics live on.
                "lon_0": lon2d,
                "lat_0": lat2d,
                "t0": t0,
                # Signed integration window; zero for the un-advected seed.
                "T": t0 - t0,
            },
        )
        return cls(ds)

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten the reference seed positions over ``('i', 'j')``.

        See :meth:`ParticleGrid.to_parcels_pset`.
        """
        stacked = self.ds.stack(particle=("i", "j"))
        return list(stacked["lon_0"].values), list(stacked["lat_0"].values)

    @classmethod
    def from_parcels_pset_lon_lat(
        cls,
        seed: "ParticleGrid",
        lon,
        lat,
        *,
        t1: np.datetime64,
    ) -> Self:
        """Reattach advected lon/lat onto the ``('i', 'j')`` MultiIndex.

        See :meth:`ParticleGrid.from_parcels_pset_lon_lat`.
        """
        # Reattach the flat advected positions to the seed's particle index as
        # the flow map F(x_0), leaving the reference positions x_0 untouched.
        ds = (
            seed.ds.stack(particle=("i", "j"))
            .assign(
                lon=("particle", np.asarray(lon, dtype=float)),
                lat=("particle", np.asarray(lat, dtype=float)),
            )
            .unstack("particle")
        )
        # Seed owns t0; derive and store the signed window T = t1 - t0.
        return cls(ds.assign_coords(T=np.datetime64(t1) - seed.ds["t0"]))

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced against neighbouring grid points ``(i +/- 1, j +/- 1)``.

        The numerator is the separation of the *advected* neighbour positions
        (from the ingested outputs); the denominator is the *initial* neighbour
        separation in the local-tangent meters frame (``dx = R cos(phi) dlambda``,
        ``dy = R dphi``). Boundary cells lacking a neighbour yield NaN. See
        :meth:`ParticleGrid.deformation_gradient`.
        """
        raise NotImplementedError(
            "deformation_gradient is not implemented (scaffolding only)."
        )


class AuxiliaryGrid(ParticleGrid):
    """Stencil = fixed four-arm auxiliary displacement grid. Haller (2015) Eq. 9.

    Each grid point carries a controlled auxiliary stencil of four neighbours --
    ``east, north, west, south`` -- on a single ``displacement`` dimension
    (coordinate ``displacement = ['east', 'north', 'west', 'south']``), with
    offsets ``dx(displacement)`` and ``dy(displacement)`` in **meters**. The
    stencil is *fixed at construction* (:meth:`from_axes`), never inferred or
    left arbitrary: there is no center point (it would duplicate the grid
    position) and no diagonal corners (unused by central differencing), so the
    emitted particle set is the minimal four arms per grid point. This decouples
    the gradient step from the seed grid resolution. The arms are placed in the
    local-tangent meters frame (``dx = R cos(phi) dlambda``, ``dy = R dphi``)
    around each grid point.
    """

    @classmethod
    def from_axes(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        t0: np.datetime64,
        aux_separation_m: float = 1_000.0,
    ) -> Self:
        """Build an auxiliary-stencil seed grid from 1-D lon/lat axes.

        See :meth:`ParticleGrid.from_axes`. Also populates the fixed four-arm
        ``displacement = ['east', 'north', 'west', 'south']`` stencil with
        offsets ``dx``, ``dy`` (meters): ``east = (+s, 0)``, ``north = (0, +s)``,
        ``west = (-s, 0)``, ``south = (0, -s)`` for ``s = aux_separation_m``. The
        shape is enforced here, not left to the caller.

        Parameters
        ----------
        lon, lat : np.ndarray
            1-D longitude/latitude axes (degrees); see
            :meth:`ParticleGrid.from_axes`.
        t0 : np.datetime64
            Release time recorded on ``.ds``; see :meth:`ParticleGrid.from_axes`.
        aux_separation_m : float, optional
            Controlled auxiliary separation ``s`` (meters) applied to every arm;
            this is the finite-difference denominator. Chosen small relative to
            the flow scale.
        """
        raise NotImplementedError("from_axes is not implemented (scaffolding only).")

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten seed positions over ``('i', 'j', 'displacement')``.

        See :meth:`ParticleGrid.to_parcels_pset`.
        """
        raise NotImplementedError(
            "to_parcels_pset is not implemented (scaffolding only)."
        )

    @classmethod
    def from_parcels_pset_lon_lat(
        cls,
        seed: "ParticleGrid",
        lon,
        lat,
        *,
        t1: np.datetime64,
    ) -> Self:
        """Reattach advected lon/lat onto the ``('i', 'j', 'displacement')`` MultiIndex.

        See :meth:`ParticleGrid.from_parcels_pset_lon_lat`.
        """
        raise NotImplementedError(
            "from_parcels_pset_lon_lat is not implemented (scaffolding only)."
        )

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced across the fixed four-arm auxiliary stencil.

        Central differences of the *advected* arm positions over the
        ``displacement`` dim -- east minus west for the ``x`` derivative, north
        minus south for the ``y`` derivative (numerator, from the ingested
        outputs) -- divided by the controlled initial separations ``dx``, ``dy``
        (denominator, meters). The per-point stencil makes grad F well-defined at
        every grid point, including the boundary. See
        :meth:`ParticleGrid.deformation_gradient` and Haller (2015) Eq. 9.
        """
        raise NotImplementedError(
            "deformation_gradient is not implemented (scaffolding only)."
        )
