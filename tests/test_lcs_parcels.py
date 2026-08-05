"""Package-level sanity checks: import, version, and class hierarchies."""

import inspect

import packaging.version
import pytest
import xarray as xr

import lcs_parcels
from lcs_parcels import (
    AuxiliaryFlowMap,
    AuxiliarySeedGrid,
    FlowMap,
    NeighborFlowMap,
    NeighborSeedGrid,
    SeedGrid,
)


def test_version():
    # A version string is exported; don't pin the value (it goes stale on bumps).
    assert isinstance(lcs_parcels.__version__, str) and lcs_parcels.__version__


def test_version_is_pep440():
    """The version parses as PEP 440.

    hatch-vcs derives it from the git tag, so a tag PEP 440 cannot express, or a
    checkout with no tags at all, shows up here rather than at upload time.
    Comparing ``__version__`` against the distribution metadata is no longer a
    test: it *is* that metadata.
    """
    packaging.version.Version(lcs_parcels.__version__)


def test_public_classes_importable():
    # Both ABCs and all four concrete classes are exported from the top level.
    for cls in (
        SeedGrid,
        NeighborSeedGrid,
        AuxiliarySeedGrid,
        FlowMap,
        NeighborFlowMap,
        AuxiliaryFlowMap,
    ):
        assert inspect.isclass(cls)


def test_seed_hierarchy():
    # Both concrete seeds subclass the abstract SeedGrid base.
    assert issubclass(NeighborSeedGrid, SeedGrid)
    assert issubclass(AuxiliarySeedGrid, SeedGrid)
    # They are distinct, first-class types (no default / fallback).
    assert NeighborSeedGrid is not AuxiliarySeedGrid


def test_flowmap_hierarchy():
    # Both concrete flow maps subclass the abstract FlowMap base.
    assert issubclass(NeighborFlowMap, FlowMap)
    assert issubclass(AuxiliaryFlowMap, FlowMap)
    assert NeighborFlowMap is not AuxiliaryFlowMap


def test_families_are_disjoint():
    # SeedGrid and FlowMap are sibling families, not an inheritance pair: a FlowMap
    # is not a kind of SeedGrid (it emits nothing) and vice versa.
    assert not issubclass(FlowMap, SeedGrid)
    assert not issubclass(SeedGrid, FlowMap)


def test_seed_is_abstract():
    # SeedGrid is an ABC: its abstract method (from_axes) makes it non-instantiable,
    # so only the concrete seeds can be built.
    with pytest.raises(TypeError):
        SeedGrid(xr.Dataset())


def test_flowmap_is_abstract():
    # FlowMap is an ABC: its abstract method (deformation_gradient) makes it
    # non-instantiable, so only the concrete flow maps can be built.
    with pytest.raises(TypeError):
        FlowMap(xr.Dataset())
