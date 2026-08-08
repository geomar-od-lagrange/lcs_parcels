"""Sphinx configuration for the LCS-Parcels documentation site.

MyST reads the `docs/*.md` sources as they are, so the four written documents
are the same files whether read on the forge or on the built site. Nothing here
maintains a second copy of any prose: the home page includes `README.md`, the
examples land on `examples/README.md` and the executed notebooks themselves, and
the API reference is generated from the docstrings. What is duplicated is
generated at build time and gitignored, so no copy is maintained by hand.
"""

import importlib.metadata
import pathlib
import shutil

project = "LCS-Parcels"
author = "Henry Owusu Adjei, Willi Rath"
copyright = "2026, GEOMAR"
release = importlib.metadata.version("lcs_parcels")
version = release

extensions = [
    "myst_nb",  # supersedes myst_parser; reads Markdown *and* notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # the docstrings are numpydoc style
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# `dollarmath` is what makes the `$...$` the docs are written with render; the
# rest are used by the written documents (tables, footnotes) or make links
# between them easier to write.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "attrs_inline",
]
myst_heading_anchors = 3  # so `notation.md#the-local-east-north-frame` resolves

# The notebooks are committed executed and are the record of a real CMEMS run.
# Re-running them here would need credentials and would replace outputs the
# repository deliberately keeps.
nb_execution_mode = "off"

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def _stage_examples() -> None:
    """Copy `examples/README.md` and the executed notebooks into the source tree.

    Sphinx resolves a symlink to its real path, so a link inside
    `examples/README.md` would be looked up next to the original rather than
    next to the staged copy. Copying instead keeps the README's own relative
    links (`[cabo_verde_ftle](cabo_verde_ftle.ipynb)`, which is what makes it
    work on the forge) resolving here too.

    Only the README and the `.ipynb` are staged. The `.py` and `.md` of each
    jupytext triplet would be read as further copies of the same notebook, and
    `data/` is the downloaded CMEMS subset. `docs/examples/` is generated, and
    gitignored.
    """
    source = pathlib.Path(__file__).parent.parent / "examples"
    staged = pathlib.Path(__file__).parent / "examples"
    shutil.rmtree(staged, ignore_errors=True)
    staged.mkdir()
    for path in [source / "README.md", *sorted(source.glob("*.ipynb"))]:
        shutil.copy2(path, staged / path.name)


_stage_examples()

# Warnings are failures. The written documents cross-reference each other by
# anchor throughout, and a link that stops resolving should fail the `docs` CI
# job rather than ship.
nitpicky = False
suppress_warnings = ["mystnb.unknown_mime_type"]

# No `html_theme`: Sphinx's default (alabaster) applies, and it ships with
# Sphinx, so the docs environment carries no theme dependency of its own.
html_title = f"LCS-Parcels {release}"

# The repository link in the sidebar, on every page. Alabaster renders these
# through `about.html`, which its default `sidebars` already includes, so there
# is no `html_sidebars` to set. The values are the lowercase strings alabaster
# compares against, not booleans, and `github_banner` stays unset: that key is
# the corner ribbon, and any value other than `"false"` turns it on.
html_theme_options = {
    "github_user": "geomar-od-lagrange",
    "github_repo": "lcs_parcels",
    "github_button": "true",
    "github_type": "star",
    "github_count": "false",
    "extra_nav_links": {
        "Source on GitHub": "https://github.com/geomar-od-lagrange/lcs_parcels",
        "Issues": "https://github.com/geomar-od-lagrange/lcs_parcels/issues",
    },
}
