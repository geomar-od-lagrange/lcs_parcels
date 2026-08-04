# Examples

Four notebooks, written to be read top to bottom, plus
[`get_data`](get_data.ipynb), which you run once to fetch the currents three of
them need. Start with
[`cabo_verde_ftle`](cabo_verde_ftle.ipynb) — seed a grid, advect it through CMEMS
currents with Parcels, map the FTLE. Then
[`cabo_verde_lcs`](cabo_verde_lcs.ipynb), which turns the same flow maps into
repelling and attracting LCS as strain tensor lines, and
[`cabo_verde_lcs_evolution`](cabo_verde_lcs_evolution.ipynb), which advects an
extracted LCS as a material curve. [`example_grid_pset`](example_grid_pset.ipynb)
stands apart: a tour of the seed and flow-map structures with no Parcels and no
data download, readable at any point.

All three Cabo Verde notebooks read one local file,
`examples/data/cabo_verde_currents_hourly.nc`: hourly surface velocity (`uo`,
`vo`) around Cabo Verde from the product
`cmems_mod_glo_phy_anfc_0.083deg_PT1H-m`. [`get_data`](get_data.ipynb) is what
writes it — run that notebook once, with CMEMS credentials, before the three.
It is plumbing rather than a demonstration; nothing else downloads anything.
`examples/data/` is gitignored and never committed, so a fresh clone starts by
running `get_data`.

Each notebook is a jupytext triplet. The `.py` (py:percent) is the source of
truth; the `.md` and `.ipynb` are generated with `jupytext --sync`, and the
`.ipynb` is committed executed, so the rendered notebook shows real output.
Edit the `.py`, never the `.md` or `.ipynb`.
