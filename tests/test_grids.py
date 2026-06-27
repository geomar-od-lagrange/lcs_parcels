"""Shape / dim / coord contracts for the Seed and FlowMap families.

A ``Seed`` (``NeighborSeed`` / ``AuxiliarySeed``) is TIME-FREE: ``from_axes``
takes no ``t0``, and the resulting dataset is all-coordinates -- no ``t0``, no
``T``, and no advected ``lon``/``lat`` data vars. Time and the advected
positions enter only at ingest, ``seed.pset_to_flowmap(lon, lat, *, t0, t1)``,
which returns a ``FlowMap`` carrying scalar ``t0``/``T`` coords and ``lon``/
``lat`` data vars.
"""

import numpy as np

from lcs_parcels import AuxiliarySeed, NeighborSeed

# Release/end times supplied only at ingest (the seed itself is time-free).
T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")


# --- seed shape ------------------------------------------------------------


def test_neighbor_seed_from_axes_dims(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    # curvilinear reference lon_0/lat_0 on logical dims (i, j)
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}
    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0

    # NeighborSeed carries no displacement stencil dim
    assert "displacement" not in ds.dims


def test_neighbor_seed_lon_lat_values(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    # 2D reference lon_0/lat_0 broadcast from the 1D axes (label-based access).
    lon_iv = lon_axis[2]
    lat_jv = lat_axis[3]
    assert float(ds["lon_0"].isel(i=2, j=3)) == lon_iv
    assert float(ds["lat_0"].isel(i=2, j=3)) == lat_jv
    # lon_0 varies along i and is constant along j; lat_0 the reverse.
    assert float(ds["lon_0"].isel(i=2, j=0)) == float(ds["lon_0"].isel(i=2, j=4))
    assert float(ds["lat_0"].isel(i=0, j=3)) == float(ds["lat_0"].isel(i=3, j=3))


def test_auxiliary_seed_from_axes_dims(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon_axis, lat_axis)
    ds = seed.ds

    assert ds.sizes["i"] == lon_axis.size
    assert ds.sizes["j"] == lat_axis.size

    # AuxiliarySeed adds a fixed four-arm stencil on a single `displacement` dim
    # (no center, no diagonals; the shape is enforced, not arbitrary).
    assert ds.sizes["displacement"] == 4
    assert list(ds["displacement"].values) == ["east", "north", "west", "south"]

    # The reference release positions x_0 are the explicit per-arm positions:
    # lon_0/lat_0 carry the displacement dim (this is what to_parcels_pset emits,
    # so the dataset is self-sufficient -- no metric needed to recover them).
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat_0"].dims) == {"i", "j", "displacement"}

    # The grid-point centres on which diagnostics are reported are kept explicitly
    # on (i, j) (needed for downstream LCS work).
    assert set(ds["lon_c"].dims) == {"i", "j"}
    assert set(ds["lat_c"].dims) == {"i", "j"}

    # A seed is time-free and carries no advected positions: no t0, no T, and no
    # lon/lat data vars (only the reference + centre coords).
    assert "t0" not in ds.coords
    assert "T" not in ds.coords
    assert "lon" not in ds.variables
    assert "lat" not in ds.variables
    assert len(ds.data_vars) == 0


def test_only_auxiliary_has_stencil(lon_axis, lat_axis):
    ns = NeighborSeed.from_axes(lon_axis, lat_axis)
    aus = AuxiliarySeed.from_axes(lon_axis, lat_axis)

    # NeighborSeed has no auxiliary stencil: no displacement dim, no centres, and
    # its reference positions are the grid points themselves on (i, j).
    assert "displacement" not in ns.ds.dims
    assert "lon_c" not in ns.ds.coords and "lat_c" not in ns.ds.coords
    assert set(ns.ds["lon_0"].dims) == {"i", "j"}

    # AuxiliarySeed carries the four-arm displacement stencil (in its explicit
    # reference positions) and the separate diagnostic centres.
    assert aus.ds.sizes["displacement"] == 4
    assert set(aus.ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert "lon_c" in aus.ds.coords and "lat_c" in aus.ds.coords


# --- flow-map shape --------------------------------------------------------


def test_neighbor_flowmap_shape(lon_axis, lat_axis):
    seed = NeighborSeed.from_axes(lon_axis, lat_axis)
    fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t0=T0, t1=T1)
    ds = fm.ds

    # The flow map adds the advected positions as its only data vars, on (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j"}
    assert set(ds["lat"].dims) == {"i", "j"}

    # Reference positions are carried through unchanged on (i, j).
    assert set(ds["lon_0"].dims) == {"i", "j"}
    assert set(ds["lat_0"].dims) == {"i", "j"}

    # Scalar t0 (release time) and signed window T = t1 - t0 land as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == T0
    assert ds["T"] == (T1 - T0)


def test_auxiliary_flowmap_shape(lon_axis, lat_axis):
    seed = AuxiliarySeed.from_axes(lon_axis, lat_axis)
    fm = seed.pset_to_flowmap(*seed.to_parcels_pset(), t0=T0, t1=T1)
    ds = fm.ds

    # Advected arm positions carry the displacement dim on top of (i, j).
    assert "lon" in ds.data_vars
    assert "lat" in ds.data_vars
    assert set(ds["lon"].dims) == {"i", "j", "displacement"}
    assert set(ds["lat"].dims) == {"i", "j", "displacement"}

    # Reference arms + centres carried through unchanged.
    assert set(ds["lon_0"].dims) == {"i", "j", "displacement"}
    assert set(ds["lon_c"].dims) == {"i", "j"}
    assert set(ds["lat_c"].dims) == {"i", "j"}

    # Scalar t0 / signed T as coords.
    assert ds["t0"].ndim == 0
    assert ds["T"].ndim == 0
    assert ds["t0"] == T0
    assert ds["T"] == (T1 - T0)
