# API reference

Generated from the docstrings, so it cannot drift from the code. For prose that
says which of these to reach for and why, read [the API guide](api.md); for the
symbols they are written in, [the notation](notation.md).

## Seeds

A `Seed` is time-free: it lays out release positions and emits a particle set.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   lcs_parcels.Seed
   lcs_parcels.NeighborSeed
   lcs_parcels.AuxiliarySeed
```

## Flow maps

A `FlowMap` holds the advected positions and computes the diagnostics.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   lcs_parcels.FlowMap
   lcs_parcels.NeighborFlowMap
   lcs_parcels.AuxiliaryFlowMap
```

## Tensor lines

Hyperbolic LCS as strain tensor lines, taking the gridded outputs of a
`FlowMap`.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   lcs_parcels.ftle_ridge_seeds
   lcs_parcels.shrink_lines
```

## Modules

The two modules' own docstrings. Their members are the classes and functions
above, documented there rather than a second time here.

```{eval-rst}
.. automodule:: lcs_parcels.grids
   :no-members:

.. automodule:: lcs_parcels.tensorlines
   :no-members:
```
