# Examples

The notebooks are listed in increasing scope and are read in that order. The
Cabo Verde ones build on each other. `cape_cauldron_vortices` builds on
`cabo_verde_ftle`, and `cape_cauldron_gled` builds on `cape_cauldron_vortices`.
`example_grid_pset` stands on its own and can be read at any point.

- [`get_data`](get_data.ipynb) downloads the CMEMS subsets the notebooks read,
  the hourly Cabo Verde currents, the GLORYS12 and DUACS Cape Cauldron fields,
  and the GLED records. Run it once, before them.
- [`example_grid_pset`](example_grid_pset.ipynb) illustrates the seed grid and
  flow map structures without any advection.
- [`cabo_verde_ftle`](cabo_verde_ftle.ipynb) seeds a grid, advects it through
  CMEMS currents with Parcels, and maps the FTLE.
- [`cabo_verde_lcs`](cabo_verde_lcs.ipynb) turns the same flow maps into
  repelling and attracting LCS as strain tensor lines.
- [`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb) advects an
  extracted LCS as a material curve.
- [`cape_cauldron_vortices`](cape_cauldron_vortices.ipynb) finds coherent
  Lagrangian vortex boundaries as closed shear lines of the Cauchy-Green tensor
  over the Agulhas-ring corridor in GLORYS12 currents from 1 June 2018, with
  `elliptic_centres`, `closed_shear_lines` and `outermost_shear_lines`, and
  checks a boundary's stretching against its $\lambda$.
- [`cape_cauldron_gled`](cape_cauldron_gled.ipynb) runs `elliptic_lcs` on the
  DUACS, GLORYS12 and GLORYS12 geostrophic currents from 1 June 2018 over 10
  to 30 days, and compares the boundaries with the GLED v1.0 coherent-eddy
  records for that window.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.
