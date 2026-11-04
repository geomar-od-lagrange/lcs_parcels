# Examples

The notebooks are listed in increasing scope and are read in that order. The
Cabo Verde ones build on each other, and `cape_cauldron_vortices` builds on
`cabo_verde_ftle`. `example_grid_pset` stands on its own and can be read at any point.

- [`get_data`](get_data.ipynb) downloads the CMEMS subsets the notebooks read,
  the hourly Cabo Verde currents and the daily GLORYS12 Cape Cauldron currents.
  Run it once, before them.
- [`example_grid_pset`](example_grid_pset.ipynb) illustrates the seed grid and
  flow map structures without any advection.
- [`cabo_verde_ftle`](cabo_verde_ftle.ipynb) seeds a grid, advects it through
  CMEMS currents with Parcels, and maps the FTLE.
- [`cabo_verde_lcs`](cabo_verde_lcs.ipynb) turns the same flow maps into
  repelling and attracting LCS as strain tensor lines.
- [`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb) advects an
  extracted LCS as a material curve.
- [`cape_cauldron_vortices`](cape_cauldron_vortices.ipynb) finds coherent
  Lagrangian vortex boundaries as closed shear lines over the Agulhas-ring
  corridor in GLORYS12 currents, with `elliptic_centres`, `stretch_range`,
  `closed_shear_lines`, `outermost_shear_lines`, `FlowMap.polar_rotation` and
  `FlowMap.elliptic_lcs`.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.
