"""Package-level sanity checks: import, version, and class hierarchies."""

import importlib.metadata
import inspect

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


def test_version_matches_the_installed_distribution():
    """``__version__`` equals what the installed distribution reports.

    PEP 440 normalises the version on the way into the metadata, so a
    zero-padded CalVer date in the source (``2026.08.04.1``) is published as
    ``2026.8.4.1`` and the two disagree for anyone who installs. Pins no value,
    so a release bump needs no edit here.
    """
    assert lcs_parcels.__version__ == importlib.metadata.version("lcs_parcels")


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
