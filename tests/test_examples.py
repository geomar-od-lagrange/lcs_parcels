"""Every example runs against the present API.

`AGENTS.md` treats a broken example as a failing test rather than a TODO; these
are that test. The `.py` half of each jupytext triplet is an ordinary Python
script -- `# %%` is a comment -- so executing it is enough to catch a changed
signature, dim name or data layout, and needs no notebook machinery.

What this does not cover: the display path. A cell ending in a bare `ds` renders
a repr under Jupyter and is a no-op under the interpreter, so a broken repr
survives here. The committed `.ipynb`, executed at authoring time, is what
covers that.

Prerequisites are checked *inside* each test, never with `skipif`. A module-level
`skipif` freezes its condition at import, which silently un-gates the Cabo Verde
examples on any run that downloads the subset during the session: the file does
not exist at collection, so the skip is decided before `get_data` can create it.

Set `LCS_REQUIRE_EXAMPLES=1` to turn every skip here into a failure. CI sets it
for the job that has CMEMS credentials, where a skip means the gate silently did
nothing rather than that the runner lacks the data.
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


def _unavailable(reason):
    """Skip, unless the caller has declared the prerequisites must be there."""
    if os.environ.get("LCS_REQUIRE_EXAMPLES"):
        pytest.fail(f"LCS_REQUIRE_EXAMPLES is set but {reason}")
    pytest.skip(reason)


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


@pytest.fixture(scope="session")
def currents():
    """The CMEMS subset the Cabo Verde examples read, downloading it if need be.

    A fixture rather than a check per test, so the three examples do not depend
    on running in file order to see a file `test_get_data` created.
    """
    if not CURRENTS.exists():
        if not _installed("copernicusmarine"):
            _unavailable("copernicusmarine is not installed")
        if not _have_cmems_credentials():
            _unavailable(f"{CURRENTS.name} is absent and there are no CMEMS credentials")
        run_example("get_data")
    if not CURRENTS.exists():
        pytest.fail(f"get_data.py ran but did not produce {CURRENTS}")
    if not _installed("parcels"):
        _unavailable("Parcels is not installed")
    return CURRENTS


def test_example_grid_pset():
    """The Parcels-free tour of the seed and flow-map structures."""
    run_example("example_grid_pset")


def test_get_data():
    """Fetching the CMEMS subset, or -- once it is there -- skipping the fetch."""
    if not _installed("copernicusmarine"):
        _unavailable("copernicusmarine is not installed")
    if not (CURRENTS.exists() or _have_cmems_credentials()):
        _unavailable("no downloaded subset and no CMEMS credentials")
    run_example("get_data")


def test_cabo_verde_ftle(currents):
    run_example("cabo_verde_ftle")


def test_cabo_verde_lcs(currents):
    run_example("cabo_verde_lcs")


def test_cabo_verde_lcs_evolution(currents):
    run_example("cabo_verde_lcs_evolution")
