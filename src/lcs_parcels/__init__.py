"""LCS-Parcels: Lagrangian coherent structure diagnostics on top of Parcels.

The diagnostic layer (deformation gradient, Cauchy-Green tensor, eigen-analysis,
FTLE) and the geometric layer (hyperbolic LCS as shrink lines, ``shrink_lines`` /
``ftle_ridge_seeds``) follow Haller (2015),
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
from lcs_parcels.tensorlines import ftle_ridge_seeds, shrink_lines

__version__ = "2026.07.17.1"

# EARTH_RADIUS_M is an internal constant, not part of the public surface.
__all__ = [
    "AuxiliaryFlowMap",
    "AuxiliarySeed",
    "FlowMap",
    "NeighborFlowMap",
    "NeighborSeed",
    "Seed",
    "ftle_ridge_seeds",
    "shrink_lines",
]
