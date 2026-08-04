"""Every example runs against the present API.

`AGENTS.md` treats a broken example as a failing test rather than a TODO; these
are that test. The `.py` half of each jupytext triplet is an ordinary Python
script -- `# %%` is a comment -- so executing it is enough to catch a changed
signature, dim name or data layout, and needs no notebook machinery.

What this does not cover: the display path. A cell ending in a bare `ds` renders
a repr under Jupyter and is a no-op under the interpreter, so a broken repr
survives here. The committed `.ipynb`, executed at authoring time, is what
covers that.

Each example is skipped rather than failed when its prerequisites are absent, so
the same test file is correct in the minimal default environment (where only the
Parcels-free example can run) and in `examples` (where all of them can).
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CURRENTS = EXAMPLES / "data" / "cabo_verde_currents_hourly.nc"

# Parcels advection over an hourly field is slow; this is a hang guard, not a
# performance budget.
TIMEOUT_S = 3600


def _installed(module):
    return importlib.util.find_spec(module) is not None


def _have_cmems_credentials():
    if os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"):
        return True
    return (
        Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"
    ).exists()


needs_currents = pytest.mark.skipif(
    not (_installed("parcels") and CURRENTS.exists()),
    reason=f"needs Parcels and {CURRENTS.name}; run `pixi run -e examples get-data` once",
)

needs_cmems = pytest.mark.skipif(
    not (
        _installed("copernicusmarine")
        and (CURRENTS.exists() or _have_cmems_credentials())
    ),
    reason="needs copernicusmarine, plus either the downloaded subset or CMEMS credentials",
)


def run_example(name):
    """Execute one example's .py source, failing with its output on a non-zero exit."""
    script = EXAMPLES / f"{name}.py"
    assert script.exists(), f"{script} does not exist"

    completed = subprocess.run(
        [sys.executable, script.name],
        cwd=EXAMPLES,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            f"{script.name} exited {completed.returncode}\n\n"
            f"--- stdout (tail) ---\n{completed.stdout[-4000:]}\n\n"
            f"--- stderr (tail) ---\n{completed.stderr[-4000:]}"
        )


def test_example_grid_pset():
    """The Parcels-free tour of the seed and flow-map structures."""
    run_example("example_grid_pset")


@needs_cmems
def test_get_data():
    """Fetching the CMEMS subset, or -- once it is there -- skipping the fetch."""
    run_example("get_data")


@needs_currents
def test_cabo_verde_ftle():
    run_example("cabo_verde_ftle")


@needs_currents
def test_cabo_verde_lcs():
    run_example("cabo_verde_lcs")


@needs_currents
def test_cabo_verde_lcs_evolution():
    run_example("cabo_verde_lcs_evolution")
