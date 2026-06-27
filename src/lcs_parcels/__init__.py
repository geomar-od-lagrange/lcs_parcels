"""LCS-Parcels: Lagrangian coherent structure diagnostics on top of Parcels.

The diagnostic layer (deformation gradient, Cauchy-Green tensor, eigen-analysis,
FTLE) following Haller (2015),
doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322). This package contains no
Parcels code: it only emits particle sets and ingests their advected positions.
"""

from __future__ import annotations

from lcs_parcels.grids import (
    AuxiliaryFlowMap,
    AuxiliarySeed,
    FlowMap,
    NeighborFlowMap,
    NeighborSeed,
    Seed,
)

__version__ = "0.1.0"

# EARTH_RADIUS_M is an internal constant of the local-tangent meters metric; it
# stays reachable as lcs_parcels.grids.EARTH_RADIUS_M but is not part of the
# public surface.
__all__ = [
    "AuxiliaryFlowMap",
    "AuxiliarySeed",
    "FlowMap",
    "NeighborFlowMap",
    "NeighborSeed",
    "Seed",
]
