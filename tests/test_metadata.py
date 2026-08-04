"""Output metadata: every returned array is self-describing.

A displayed dataset and a vanilla ``.plot()`` take their axis labels, titles and
colorbar captions from ``name``/``long_name``/``units``, so every coordinate,
data variable and returned field must carry them, and two distinct quantities
must never come back under the same ``name``.
"""

import numpy as np
import pytest
from conftest import advected_flowmap

from lcs_parcels import AuxiliarySeed, NeighborSeed, shrink_lines

T0 = np.datetime64("2020-01-01")
T1 = np.datetime64("2020-01-02")
M = np.array([[2.0, 0.5], [0.0, 3.0]])  # generic (sheared) linear map

SEED_CLASSES = [NeighborSeed, AuxiliarySeed]

# Coordinates whose values are labels or indices, for which a unit is
# meaningless; they carry a long_name only.
UNITLESS = {
    "i",
    "j",
    "displacement",
    "row",
    "col",
    "comp",
    "eig",
    "line",
    "point",
    "t0",
    "T",
}


def assert_labelled(obj, expected_units=None):
    """Every entry of `obj` (Dataset or DataArray) carries its metadata."""
    arrays = obj.data_vars.values() if hasattr(obj, "data_vars") else [obj]
    for array in arrays:
        assert array.name is not None
        assert array.attrs.get("long_name")
        assert array.attrs.get("units")
        if expected_units is not None:
            assert array.attrs["units"] == expected_units[str(array.name)]
    for name, coord in obj.coords.items():
        assert coord.attrs.get("long_name"), f"{name} has no long_name"
        if str(name) not in UNITLESS:
            assert coord.attrs.get("units"), f"{name} has no units"


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_seed_coords_are_labelled(seed_cls, lon_axis, lat_axis):
    seed = seed_cls.from_axes(lon=lon_axis, lat=lat_axis)
    assert_labelled(seed.ds)


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_flowmap_dataset_is_labelled(seed_cls, lon_axis, lat_axis):
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, T0, T1)
    assert_labelled(
        fm.ds, expected_units={"lon": "degrees_east", "lat": "degrees_north"}
    )


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_grid_image_is_labelled(seed_cls, lon_axis, lat_axis):
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, T0, T1)
    assert_labelled(
        fm.grid_image, expected_units={"lon": "degrees_east", "lat": "degrees_north"}
    )


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_diagnostics_are_labelled(seed_cls, lon_axis, lat_axis):
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, T0, T1)
    assert_labelled(fm.deformation_gradient())
    assert_labelled(fm.cauchy_green())
    assert_labelled(fm.cg_eigen())
    assert_labelled(fm.ftle())


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_dimensionless_diagnostics_and_ftle_units(seed_cls, lon_axis, lat_axis):
    """Tensors and eigenpairs are dimensionless; the FTLE is SI (1/s)."""
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, T0, T1)
    assert fm.deformation_gradient().attrs["units"] == "1"
    assert fm.cauchy_green().attrs["units"] == "1"
    eigen = fm.cg_eigen()
    assert eigen["lambda"].attrs["units"] == "1"
    assert eigen["xi"].attrs["units"] == "1"
    assert fm.ftle().attrs["units"] == "1/s"


@pytest.mark.parametrize("seed_cls", SEED_CLASSES)
def test_distinct_quantities_have_distinct_names(seed_cls, lon_axis, lat_axis):
    """No two different diagnostics come back under the same name."""
    fm = advected_flowmap(seed_cls, lon_axis, lat_axis, M, T0, T1)
    eigen = fm.cg_eigen()
    names = [
        fm.deformation_gradient().name,
        fm.cauchy_green().name,
        eigen["lambda"].name,
        eigen["xi"].name,
        fm.ftle().name,
    ]
    assert len(set(names)) == len(names)


def test_eig_coord_says_which_eigenvalue_is_which(lon_axis, lat_axis):
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    long_name = fm.cg_eigen()["eig"].attrs["long_name"]
    assert "lambda_1" in long_name
    assert "lambda_max" in long_name


def test_image_output_is_labelled(lon_axis, lat_axis):
    fm = advected_flowmap(NeighborSeed, lon_axis, lat_axis, M, T0, T1)
    out = fm.image(lon0=fm.ds["lon_0"], lat0=fm.ds["lat_0"])
    assert_labelled(out, expected_units={"lon": "degrees_east", "lat": "degrees_north"})


def test_shrink_lines_output_is_labelled(lon_axis, lat_axis):
    fm = advected_flowmap(AuxiliarySeed, lon_axis, lat_axis, M, T0, T1)
    lines = shrink_lines(
        fm, seed_lon=lon_axis[1:3], seed_lat=lat_axis[1:3], n_steps=3, step_m=1_000.0
    )
    assert_labelled(
        lines, expected_units={"lon": "degrees_east", "lat": "degrees_north"}
    )
