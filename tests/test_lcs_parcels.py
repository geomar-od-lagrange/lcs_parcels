"""Package-level sanity checks: import, version, and class hierarchies."""

import inspect

import pytest
import xarray as xr

import lcs_parcels
from lcs_parcels import (
    AuxiliaryFlowMap,
    AuxiliarySeed,
    FlowMap,
    NeighborFlowMap,
    NeighborSeed,
    Seed,
)


def test_version():
    # A version string is exported; don't pin the value (it goes stale on bumps).
    assert isinstance(lcs_parcels.__version__, str) and lcs_parcels.__version__


def test_public_classes_importable():
    # Both ABCs and all four concrete classes are exported from the top level.
    for cls in (
        Seed,
        NeighborSeed,
        AuxiliarySeed,
        FlowMap,
        NeighborFlowMap,
        AuxiliaryFlowMap,
    ):
        assert inspect.isclass(cls)


def test_seed_hierarchy():
    # Both concrete seeds subclass the abstract Seed base.
    assert issubclass(NeighborSeed, Seed)
    assert issubclass(AuxiliarySeed, Seed)
    # They are distinct, first-class types (no default / fallback).
    assert NeighborSeed is not AuxiliarySeed


def test_flowmap_hierarchy():
    # Both concrete flow maps subclass the abstract FlowMap base.
    assert issubclass(NeighborFlowMap, FlowMap)
    assert issubclass(AuxiliaryFlowMap, FlowMap)
    assert NeighborFlowMap is not AuxiliaryFlowMap


def test_families_are_disjoint():
    # Seed and FlowMap are sibling families, not an inheritance pair: a FlowMap
    # is not a kind of Seed (it emits nothing) and vice versa.
    assert not issubclass(FlowMap, Seed)
    assert not issubclass(Seed, FlowMap)


def test_seed_is_abstract():
    # Seed is an ABC: its abstract method (from_axes) makes it non-instantiable,
    # so only the concrete seeds can be built.
    with pytest.raises(TypeError):
        Seed(xr.Dataset())


def test_flowmap_is_abstract():
    # FlowMap is an ABC: its abstract method (deformation_gradient) makes it
    # non-instantiable, so only the concrete flow maps can be built.
    with pytest.raises(TypeError):
        FlowMap(xr.Dataset())
