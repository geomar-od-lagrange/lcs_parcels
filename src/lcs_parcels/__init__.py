"""LCS-Parcels: Lagrangian coherent structure diagnostics on top of Parcels.

The diagnostic layer (deformation gradient, Cauchy-Green tensor, eigen-analysis,
FTLE) following Haller (2015),
doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322). This package contains no
Parcels code: it only emits particle sets and ingests their advected positions.
"""

from __future__ import annotations

from lcs_parcels.grids import (
    EARTH_RADIUS_M,
    AuxiliaryGrid,
    NeighborGrid,
    ParticleGrid,
)

__version__ = "0.1.0"

__all__ = [
    "EARTH_RADIUS_M",
    "AuxiliaryGrid",
    "NeighborGrid",
    "ParticleGrid",
]
