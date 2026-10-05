"""LCS-Parcels: Lagrangian coherent structure diagnostics on top of Parcels.

The diagnostic layer (deformation gradient, Cauchy-Green tensor, eigen-analysis,
FTLE) and the geometric layer (hyperbolic LCS as shrink lines, ``shrink_lines`` /
``ftle_ridge_seeds``) follow Haller (2015),
doi:10.1146/annurev-fluid-010313-141322
(https://doi.org/10.1146/annurev-fluid-010313-141322). Elliptic LCS as closed
shear lines (``closed_shear_lines``) follow Haller & Beron-Vera (2013),
doi:10.1017/jfm.2013.391 (https://doi.org/10.1017/jfm.2013.391). This package contains no
Parcels code. It only emits particle sets and ingests their advected positions.
"""

from __future__ import annotations

import importlib.metadata

from lcs_parcels.elliptic import (
    closed_shear_lines,
    elliptic_centres,
    outermost_shear_lines,
    stretch_range,
)
from lcs_parcels.grids import (
    AuxiliaryFlowMap,
    AuxiliarySeedGrid,
    FlowMap,
    NeighborFlowMap,
    NeighborSeedGrid,
    SeedGrid,
)
from lcs_parcels.tensorlines import ftle_ridge_seeds, prune_shrink_lines, shrink_lines

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
    "closed_shear_lines",
    "elliptic_centres",
    "ftle_ridge_seeds",
    "outermost_shear_lines",
    "prune_shrink_lines",
    "shrink_lines",
    "stretch_range",
]
