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
by ``sign(T)`` (``t1 < t0`` -> backward -> attracting LCS; ``t1 > t0`` -> forward
-> repelling LCS); there is no separate flag and ``t1`` itself is not stored. See
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
must share the same dims. ``lon_0``/``lat_0`` are the *release* positions (what
:meth:`to_parcels_pset` emits); for :class:`NeighborGrid` the release point is the
grid point itself, so ``lon_0``/``lat_0`` are on ``('i', 'j')``. For
:class:`AuxiliaryGrid` the release points are the four stencil arms, so
``lon_0``/``lat_0`` and the advected ``lon``/``lat`` all carry the extra
``displacement`` dim, i.e. ``('i', 'j', 'displacement')``, and the grid-point
*centres* on which the diagnostics are reported are kept explicitly as a separate
pair ``lon_c``/``lat_c`` on ``('i', 'j')``; see its docstring.

Sphere metric convention
------------------------
Haller's math is Cartesian; our grids are lon/lat. Separations are formed in a
*single* local tangent frame in meters, anchored at the grid centroid -- the one
reference longitude/latitude ``lon_ref = lon_0.mean()``,
``lat_ref = lat_0.mean()``. With ``R`` the Earth radius and ``deg = pi / 180``,
positions convert as ``X = R cos(phi_ref) (lambda - lambda_ref) deg`` and
``Y = R (phi - phi_ref) deg`` (``lambda`` longitude, ``phi`` latitude). The
cosine factor uses that one grid reference latitude ``phi_ref`` for every point,
*not* a per-point ``cos(phi)``: a single shared frame is what makes the
off-diagonal deformation-gradient terms exact -- a per-point cosine would corrupt
them by a cosine ratio -- and in the tiny-separation regime the flat-tangent
approximation is adequate (a convention, not a correctness blocker). This metric
only converts lon/lat separations to meters; the *advected* separations that form
the numerator of grad F are measured from the ingested Parcels output positions,
not recomputed analytically from ``R`` and ``phi``. Both subclasses share the
identical metric code (:meth:`ParticleGrid._to_meters`).
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

    Uses a *single* grid reference latitude ``lat_ref`` for the cosine factor
    (``X = R cos(phi_ref) (lambda - lambda_ref) deg``, ``Y = R (phi - phi_ref) deg``),
    so the whole grid shares one frame; see the module-level *Sphere metric
    convention*. Works on plain arrays or label-based xarray objects (the
    arithmetic is broadcasting only).
    """
    c = np.cos(lat_ref * _DEG)
    x = EARTH_RADIUS_M * c * (lon - lon_ref) * _DEG
    y = EARTH_RADIUS_M * (lat - lat_ref) * _DEG
    return x, y


def _meters_to_lonlat(x, y, lon_ref: float, lat_ref: float):
    """Inverse of :func:`_lonlat_to_meters`, sharing the one reference latitude."""
    c = np.cos(lat_ref * _DEG)
    lon = lon_ref + x / (EARTH_RADIUS_M * c * _DEG)
    lat = lat_ref + y / (EARTH_RADIUS_M * _DEG)
    return lon, lat


def _assemble_tensor(
    fxx: xr.DataArray,
    fxy: xr.DataArray,
    fyx: xr.DataArray,
    fyy: xr.DataArray,
) -> xr.DataArray:
    """Pack four scalar ``(i, j)`` component fields into a ``(row, col)`` tensor.

    ``row`` and ``col`` become dimension coordinates valued ``['x', 'y']`` with
    ``tensor.sel(row=a, col=b)`` holding the ``(a, b)`` component, e.g.
    ``gradF.sel(row='y', col='x') = dF_y / dx0_x``. The component fields are
    renamed to a common name so :func:`xarray.concat` does not drop one.
    """
    fxx, fxy = fxx.rename("tensor"), fxy.rename("tensor")
    fyx, fyy = fyx.rename("tensor"), fyy.rename("tensor")
    row_x = xr.concat([fxx, fxy], dim="col")
    row_y = xr.concat([fyx, fyy], dim="col")
    tensor = xr.concat([row_x, row_y], dim="row")
    return tensor.assign_coords(row=["x", "y"], col=["x", "y"])


class ParticleGrid(abc.ABC):
    """Composition wrapper around an ``xr.Dataset`` of seed positions.

    The wrapped dataset is held in :attr:`ds` (this class does *not* subclass
    ``xr.Dataset``, which xarray discourages). The dataset has logical grid
    dimensions ``i, j``; the reference initial (release) positions
    ``lon_0``/``lat_0`` (degrees) are coordinates and the advected positions
    ``lon``/``lat`` are data variables, both pairs sharing the same dims.
    Two-dimensional lon/lat support curvilinear / non-rectangular grids. The
    reference positions are the initial conditions ``x_0``; the advected positions
    are the flow map ``F_{t0}^{t1}(x_0)``. For :class:`NeighborGrid` both pairs are
    on ``(i, j)`` and the grid point is itself the diagnostic location; for
    :class:`AuxiliaryGrid` both pairs carry an extra ``displacement`` dim (the
    stencil arms) and the diagnostic *centres* ``lon_c``/``lat_c`` are kept on
    ``(i, j)`` (see the module-level *Position convention* and the subclass
    docstrings).

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

    # --- shared local-tangent metric ---------------------------------------
    #
    # NeighborGrid and AuxiliaryGrid use these *identical* helpers so that both
    # read positions in the one frame anchored at the grid centroid (see the
    # module-level *Sphere metric convention*).

    def _reference_lonlat(self) -> tuple[float, float]:
        """The single grid reference point ``(lon_ref, lat_ref)`` = centroid means."""
        return float(self.ds["lon_0"].mean()), float(self.ds["lat_0"].mean())

    def _to_meters(self, lon: xr.DataArray, lat: xr.DataArray):
        """Project ``lon``/``lat`` into the grid's single local-tangent meters frame."""
        lon_ref, lat_ref = self._reference_lonlat()
        return _lonlat_to_meters(lon, lat, lon_ref, lat_ref)

    def _integration_seconds(self) -> float:
        """``|T|`` in seconds from the stored signed window ``T`` (``timedelta64``).

        Standard ``datetime64`` division suffices here; a calendar-aware
        conversion would be needed for non-standard model calendars.
        """
        return float(np.abs(self.ds["T"] / np.timedelta64(1, "s")))

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
        raise NotImplementedError("abstract method; implemented by subclasses.")

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
        raise NotImplementedError("abstract method; implemented by subclasses.")

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
            (``t1 < t0`` -> backward -> attracting LCS; ``t1 > t0`` -> forward
            -> repelling LCS) and is recorded as a coordinate, used by
            :meth:`ftle`. ``t1`` itself is not stored.

        Returns
        -------
        Self
            A grid whose ``.ds`` holds the advected positions as ``lon``/``lat``
            on the original grid and records ``t0`` and the derived signed ``T``
            as coordinates.
        """
        raise NotImplementedError("abstract method; implemented by subclasses.")

    @abc.abstractmethod
    def deformation_gradient(self) -> xr.DataArray:
        """Deformation gradient grad F of the flow map. Haller (2015) Eq. 9.

        The 2x2 tensor ``grad F = d(lon, lat) / d(lon_0, lat_0)`` per grid point,
        estimated by finite differences as
        ``(advected separation) / (initial separation)``:

        - **denominator** — the *initial* separation of the reference
          ``lon_0``/``lat_0`` between stencil points, a controlled quantity taken
          in the shared single-reference-latitude meters frame
          (:meth:`_to_meters`; one grid ``cos(phi_ref)``, not a per-point cosine);
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
            grad F with dims ``(i, j, row, col)``; ``row`` and ``col`` are
            dimension coordinates valued ``['x', 'y']``, with
            ``grad F.sel(row=a, col=b) = d F_a / d x0_b`` (dimensionless;
            meters / meters). There is no separate ``comp`` coordinate on the
            tensor (``comp`` labels the eigenvector component dim).
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
        *largest* eigenvalue of ``C`` from :meth:`cg_eigen` and the recorded
        signed integration window ``T`` (``timedelta64``, derived at ingest as
        ``t1 - t0`` and converted to seconds; the conversion must be
        calendar-aware for non-standard model calendars).

        Returns
        -------
        xr.DataArray
            FTLE field with dims ``(i, j)`` in units of 1/second.
        """
        lambda_max = self.cg_eigen()["lambda"].isel(eig=1)
        t_sec = self._integration_seconds()
        # (1 / |T|) log sqrt(lambda_max) = (1 / |T|) * 0.5 * log(lambda_max).
        return (1.0 / t_sec) * 0.5 * np.log(lambda_max)


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
        separation in the shared single-reference-latitude meters frame
        (:meth:`ParticleGrid._to_meters`). Boundary cells lacking a neighbour
        yield NaN. See :meth:`ParticleGrid.deformation_gradient`.
        """
        x_adv, y_adv = self._to_meters(self.ds["lon"], self.ds["lat"])
        x_ref, y_ref = self._to_meters(self.ds["lon_0"], self.ds["lat_0"])

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


class AuxiliaryGrid(ParticleGrid):
    """Stencil = fixed four-arm auxiliary displacement grid. Haller (2015) Eq. 9.

    Data model. The reference release positions
    ``lon_0(i, j, displacement)`` / ``lat_0(i, j, displacement)`` (coordinates,
    degrees) are the explicit per-arm positions -- exactly what
    :meth:`to_parcels_pset` emits, so the dataset is self-sufficient (no metric
    convention is needed to recover where particles were released). The advected
    arm positions ``lon(i, j, displacement)`` / ``lat(i, j, displacement)`` (data
    variables; equal to ``lon_0`` / ``lat_0`` on a seed) share those dims, so the
    deformation gradient is the plain ``grad F = d(lon, lat) / d(lon_0, lat_0)``
    differenced over ``displacement``. The grid-point **centres**
    ``lon_c(i, j)`` / ``lat_c(i, j)`` (coordinates) are kept explicitly -- the
    points on which the diagnostics (FTLE, eigenpairs) are reported, and the
    natural anchor for downstream LCS work. ``t0`` and the signed window ``T`` are
    coordinates. There is no stored ``dx`` / ``dy``: the stencil lives in the
    reference positions themselves.

    Each grid point carries a controlled auxiliary stencil of four neighbours --
    ``east, north, west, south`` -- with offsets ``east = (+s, 0)``,
    ``north = (0, +s)``, ``west = (-s, 0)``, ``south = (0, -s)`` for
    ``s = aux_separation_m``. The stencil is *fixed at construction*
    (:meth:`from_axes`), never inferred or left arbitrary: there is no center
    point (it would duplicate the grid position) and no diagonal corners (unused
    by central differencing), so the emitted particle set is the minimal four
    arms per grid point. This decouples the gradient step from the seed grid
    resolution. Arms are placed by offsetting each centre in the *single* grid
    local-tangent meters frame (one grid reference latitude, not a per-point
    ``cos(phi)``; see :meth:`ParticleGrid._to_meters` and the module-level
    *Sphere metric convention*).
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

        See :meth:`ParticleGrid.from_axes`. The grid-point centres ``lon_c`` /
        ``lat_c`` are placed on ``(i, j)`` from the axes, then the fixed four-arm
        ``displacement = ['east', 'north', 'west', 'south']`` stencil is laid out
        around each centre at offsets ``east = (+s, 0)``, ``north = (0, +s)``,
        ``west = (-s, 0)``, ``south = (0, -s)`` meters (``s = aux_separation_m``)
        and stored *explicitly* as the reference release positions
        ``lon_0`` / ``lat_0`` on ``(i, j, displacement)``. The shape is enforced
        here, not left to the caller.

        Parameters
        ----------
        lon, lat : np.ndarray
            1-D longitude/latitude axes (degrees); see
            :meth:`ParticleGrid.from_axes`.
        t0 : np.datetime64
            Release time recorded on ``.ds``; see :meth:`ParticleGrid.from_axes`.
        aux_separation_m : float, optional
            Controlled auxiliary separation ``s`` (meters) applied to every arm;
            it sets the finite-difference step (the reference arm span is ``2s``).
            Chosen small relative to the flow scale.
        """
        # Broadcast the 1-D axes into curvilinear 2-D centre fields on (i, j);
        # lon varies along i, lat along j. These are the diagnostic centres.
        lon_axis = xr.DataArray(np.asarray(lon, dtype=float), dims="i")
        lat_axis = xr.DataArray(np.asarray(lat, dtype=float), dims="j")
        lon_c, lat_c = xr.broadcast(lon_axis, lat_axis)

        # Fixed four-arm stencil offsets in meters; shape enforced here.
        s = float(aux_separation_m)
        displacement = ["east", "north", "west", "south"]
        off_x = xr.DataArray(
            [+s, 0.0, -s, 0.0], dims="displacement", coords={"displacement": displacement}
        )
        off_y = xr.DataArray(
            [0.0, +s, 0.0, -s], dims="displacement", coords={"displacement": displacement}
        )

        # Place the arms about each centre in the single grid reference frame (one
        # reference latitude = grid centroid), so meters <-> degrees here matches
        # ParticleGrid._to_meters exactly. lon_c (i, j) broadcasts with the
        # (displacement,) offset into the explicit arm positions (i, j, displacement).
        lon_ref = float(lon_c.mean())
        lat_ref = float(lat_c.mean())
        c = np.cos(lat_ref * _DEG)
        lon_0 = lon_c + off_x / (EARTH_RADIUS_M * c * _DEG)
        lat_0 = lat_c + off_y / (EARTH_RADIUS_M * _DEG)

        t0 = np.datetime64(t0)
        ds = xr.Dataset(
            data_vars={
                # Advected arm positions F(x_0); identity arms on a seed.
                "lon": lon_0,
                "lat": lat_0,
            },
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
                "t0": t0,
                # Signed integration window; zero for the un-advected seed.
                "T": t0 - t0,
            },
        )
        return cls(ds)

    def to_parcels_pset(self) -> tuple[list[float], list[float]]:
        """Flatten the RELEASE arm positions over ``('i', 'j', 'displacement')``.

        Emits the *reference* arm positions ``lon_0``/``lat_0`` directly -- they
        are stored explicitly, so no metric reconstruction is needed and this is
        trivially robust on an already-ingested grid (the reference positions are
        never overwritten). See :meth:`ParticleGrid.to_parcels_pset`.
        """
        dims = ("i", "j", "displacement")
        lon_arm = self.ds["lon_0"].reset_coords(drop=True).stack(particle=dims)
        lat_arm = self.ds["lat_0"].reset_coords(drop=True).stack(particle=dims)
        return list(lon_arm.values), list(lat_arm.values)

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

        Only the *coord-stripped* advected ``lon``/``lat`` are stacked and
        reattached: stacking the raw seed variables would broadcast the centre
        coordinates ``lon_c(i, j)`` / ``lat_c(i, j)`` up to
        ``(i, j, displacement)`` and corrupt the schema. After ingest the centres
        stay ``(i, j)``, the reference arms ``lon_0``/``lat_0`` (carried from the
        seed, untouched) stay ``(i, j, displacement)``, and the advected arms are
        overwritten on ``(i, j, displacement)``. See
        :meth:`ParticleGrid.from_parcels_pset_lon_lat`.
        """
        dims = ("i", "j", "displacement")
        lon_arm = (
            seed.ds["lon"]
            .reset_coords(drop=True)
            .stack(particle=dims)
            .copy(data=np.asarray(lon, dtype=float))
            .unstack("particle")
        )
        lat_arm = (
            seed.ds["lat"]
            .reset_coords(drop=True)
            .stack(particle=dims)
            .copy(data=np.asarray(lat, dtype=float))
            .unstack("particle")
        )
        # Reference arms lon_0/lat_0 and centres lon_c/lat_c are carried from the
        # seed untouched; only the advected arms and the derived window change.
        ds = seed.ds.assign(lon=lon_arm, lat=lat_arm).assign_coords(
            T=np.datetime64(t1) - seed.ds["t0"]
        )
        return cls(ds)

    def deformation_gradient(self) -> xr.DataArray:
        """grad F differenced across the fixed four-arm auxiliary stencil.

        The plain ``grad F = d(lon, lat) / d(lon_0, lat_0)`` over the
        ``displacement`` dim: east minus west for the ``x`` derivative, north
        minus south for the ``y`` derivative. Both the advected separation
        (numerator) and the *reference* arm separation (denominator) are read from
        positions in the shared single-reference-latitude meters frame, so the
        denominator is the explicit ``2s`` reference span -- no separately stored
        offsets. The per-point stencil makes grad F well-defined at every grid
        point, including the boundary. See
        :meth:`ParticleGrid.deformation_gradient` and Haller (2015) Eq. 9.
        """
        x_adv, y_adv = self._to_meters(self.ds["lon"], self.ds["lat"])
        x_ref, y_ref = self._to_meters(self.ds["lon_0"], self.ds["lat_0"])

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
