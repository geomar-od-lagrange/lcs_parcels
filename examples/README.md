# Examples

Example notebooks, plus [`get_data`](get_data.ipynb), which you run once to
download the CMEMS subset the three Cabo Verde notebooks read.

The three Cabo Verde notebooks build on each other and are read in the order
listed; `example_grid_pset` stands on its own and can be read at any point.

- [`cabo_verde_ftle`](cabo_verde_ftle.ipynb) seeds a grid, advects it through
  CMEMS currents with Parcels, and maps the FTLE.
- [`cabo_verde_lcs`](cabo_verde_lcs.ipynb) turns the same flow maps into
  repelling and attracting LCS as strain tensor lines.
- [`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb) advects an
  extracted LCS as a material curve.
- [`example_grid_pset`](example_grid_pset.ipynb) illustrates the seed and flow
  map structures without any advection, and stands on its own.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.
