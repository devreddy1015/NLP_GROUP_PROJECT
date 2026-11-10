import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = "scripts/run_experiments.sh"


def planned(*args, **env):
    out = subprocess.run(["bash", SCRIPT, "--dry-run", *args], capture_output=True, text=True, check=True,
                         env={**os.environ, **env}).stdout
    return [line for line in out.splitlines() if "src.models.transformer" in line]


def test_dry_run_lists_the_48_planned_runs():
    runs = planned()
    assert len(runs) == 48 and len(set(runs)) == 48
    assert sum("--model xlmr" in r for r in runs) == 3 and sum("full-minus-N" in r for r in runs) == 24
    assert sum("mixed/full-k" in r for r in runs) == 12 and sum("full-40k" in r for r in runs) == 3
    assert all("--seed 42" in r or "--seed 43" in r or "--seed 44" in r for r in runs)


def test_the_real_test_set_is_scored_only_in_final_runs():
    assert not any("realtest" in r or "real/test.jsonl" in r for r in planned())
    final = planned("--final", "e2")
    assert len(final) == 6 and all("data/real/test.jsonl" in r and "realtest" in r for r in final)


def test_experiments_can_be_run_alone_or_several_at_once():
    assert len(planned("e3")) == 24 and len(planned("e4")) == 15 and len(planned("xlmr")) == 3
    assert len(planned("e2", "e3")) == 30 and len(planned("all")) == 48


def test_a_finished_run_is_skipped(tmp_path):
    (tmp_path / "20261120-muril-synclean-realdev-s42.json").write_text("{}")
    runs = planned("e2", DATE="20261120", RESULTS=str(tmp_path))
    assert len(runs) == 5 and not any("synclean-realdev-s42" in r for r in runs)
    assert all(f"--results-dir {tmp_path}" in r for r in runs)
    assert len(planned("e2", "--final", DATE="20261120", RESULTS=str(tmp_path))) == 6  # a final run trains again


@pytest.mark.skipif(Path("data/real/dev.jsonl").exists(), reason="real dev data is present")
def test_it_refuses_to_start_without_real_data():
    done = subprocess.run(["bash", SCRIPT, "e2"], capture_output=True, text=True)
    assert done.returncode == 1 and "data/real/dev.jsonl" in done.stderr
