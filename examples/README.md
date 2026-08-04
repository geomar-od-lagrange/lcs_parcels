# Examples

Example notebooks plus [`get_data`](get_data.ipynb), which needs to run once to
fetch data other notebooks need.

- [`cabo_verde_ftle`](cabo_verde_ftle.ipynb) seeds a grid, advects it through
  CMEMS currents with Parcels, maps the FTLE,
- [`cabo_verde_lcs`](cabo_verde_lcs.ipynb) turns the same flow maps into
  repelling and attracting LCS as strain tensor lines,
- [`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb) advects an
  extracted LCS as a material curve,
- [`example_grid_pset`](example_grid_pset.ipynb) illustrates the seed and the
  flow map structures with no actual advection.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.
