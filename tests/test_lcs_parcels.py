"""Package-level sanity checks: import, version, and class hierarchy."""

import inspect

import pytest

import lcs_parcels
from lcs_parcels import AuxiliaryGrid, NeighborGrid, ParticleGrid


def test_version():
    assert lcs_parcels.__version__ == "0.1.0"


def test_public_classes_importable():
    # The three public classes are exported from the top-level package.
    assert inspect.isclass(ParticleGrid)
    assert inspect.isclass(NeighborGrid)
    assert inspect.isclass(AuxiliaryGrid)


def test_class_hierarchy():
    # Both concrete grids are subclasses of the abstract base.
    assert issubclass(NeighborGrid, ParticleGrid)
    assert issubclass(AuxiliaryGrid, ParticleGrid)
    # They are distinct, first-class types (no default / fallback).
    assert NeighborGrid is not AuxiliaryGrid


def test_particle_grid_is_abstract():
    # ParticleGrid is an ABC: it must not be instantiable on its own via the
    # from_axes seeding path used for the concrete grids.
    import numpy as np

    lon = np.linspace(-2.0, 1.0, 4)
    lat = np.linspace(10.0, 14.0, 5)
    with pytest.raises(TypeError):
        ParticleGrid.from_axes(lon, lat)
