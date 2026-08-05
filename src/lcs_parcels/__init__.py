"""LCS-Parcels: Lagrangian coherent structure diagnostics on top of Parcels.

The diagnostic layer (deformation gradient, Cauchy-Green tensor, eigen-analysis,
FTLE) and the geometric layer (hyperbolic LCS as shrink lines, ``shrink_lines`` /
``ftle_ridge_seeds``) follow Haller (2015),
doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322). This package contains no
Parcels code. It only emits particle sets and ingests their advected positions.
"""

from __future__ import annotations

import importlib.metadata

from lcs_parcels.grids import (
    AuxiliaryFlowMap,
    AuxiliarySeedGrid,
    FlowMap,
    NeighborFlowMap,
    NeighborSeedGrid,
    SeedGrid,
)
from lcs_parcels.tensorlines import ftle_ridge_seeds, shrink_lines

#: Read from the installed distribution, whose version hatch-vcs derived from the
#: git tag at build time. Nothing in the source tree carries a version literal.
__version__ = importlib.metadata.version("lcs_parcels")

__all__ = [
    "AuxiliaryFlowMap",
    "AuxiliarySeedGrid",
    "FlowMap",
    "NeighborFlowMap",
    "NeighborSeedGrid",
    "SeedGrid",
    "ftle_ridge_seeds",
    "shrink_lines",
]
