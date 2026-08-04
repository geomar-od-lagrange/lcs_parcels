# Examples

Four notebooks, written to be read top to bottom. Start with
[`cabo_verde_ftle`](cabo_verde_ftle.ipynb) — seed a grid, advect it through CMEMS
currents with Parcels, map the FTLE. Then
[`cabo_verde_lcs`](cabo_verde_lcs.ipynb), which turns the same flow maps into
repelling and attracting LCS as strain tensor lines, and
[`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb), which advects an
extracted LCS as a material curve. [`example_grid_pset`](example_grid_pset.ipynb)
stands apart: a tour of the seed and flow-map structures with no Parcels and no
data download, readable at any point.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.

The three Cabo Verde notebooks run on one CMEMS subset: hourly surface velocity
(`uo`, `vo`) around Cabo Verde from the product
`cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`. `cabo_verde_ftle` opens it with
`copernicusmarine` — its parameter and currents cells are the exact request, and
it needs CMEMS credentials. Save that dataset once to
`examples/data/cabo_verde_currents_hourly.nc`, the path the two LCS notebooks
read. `examples/data/` is not in the repository and nothing here downloads the
file for you.
