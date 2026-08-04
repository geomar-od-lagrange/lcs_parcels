"""Broken examples are treated as failing tests.

The `.py` third of each jupytext triplet is an ordinary Python script which we
run to verify the notebooks execute end to end.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_example(name):
    """Execute one example's .py source, failing with its output on a non-zero exit."""
    completed = subprocess.run(
        [sys.executable, f"{name}.py"],
        cwd=EXAMPLES,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            f"{name}.py exited {completed.returncode}\n\n"
            f"--- stdout (tail) ---\n{completed.stdout[-4000:]}\n\n"
            f"--- stderr (tail) ---\n{completed.stderr[-4000:]}"
        )


@pytest.mark.minimal
def test_example_grid_pset():
    run_example("example_grid_pset")


@pytest.mark.needs_creds
def test_get_data():
    run_example("get_data")


@pytest.mark.needs_creds
def test_cabo_verde_ftle():
    run_example("cabo_verde_ftle")


@pytest.mark.needs_creds
def test_cabo_verde_lcs():
    run_example("cabo_verde_lcs")


@pytest.mark.needs_creds
def test_cabo_verde_lcs_evolution():
    run_example("cabo_verde_lcs_evolution")
