import json
from pathlib import Path

import pytest

from scripts.make_tables import (
    compare_table, config_key, configs_table, load, main, normaliser_table, runs_table, shared_table,
)
from src.data import read_jsonl
from src.eval import evaluate, write_predictions, write_results
from src.models.zeroshot import only

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """Results as the model scripts write them: our system with two seeds, a worse one, an existing tagger, a JSON system."""
    d = tmp_path_factory.mktemp("results")
    recs = read_jsonl(HAND20)
    gold = [r["tags"] for r in recs]
    no_pin = [[t if "PIN" not in t else "O" for t in tags] for tags in gold]
    no_city = [[t if "CITY" not in t else "O" for t in tags] for tags in gold]

    def save(run_id, model, records, pred, seed, config=None):
        write_results(evaluate(records, pred, run_id=run_id, model=model, data="fixtures/handlabelled20", seed=seed,
                               config=config, bootstrap_n=10), d)
        write_predictions(records, pred, run_id, d)

    save("20261111-ours-syn-hand20-s1", "ours", recs, gold, 1)
    save("20261112-ours-syn-hand20-s2", "ours", recs, no_pin, 2)  # a second seed, run a day later
    save("20261111-worse-syn-hand20-s1", "worse", recs, no_city, 1)
    masked = [{**r, "tags": only(r["tags"], {"CITY", "PIN"})} for r in recs]
    save("20261111-theirs-hand20-s0", "theirs", masked, [only(p, {"CITY", "PIN"}) for p in no_city], 0,
         config={"shared_labels": ["CITY", "PIN"]})
    write_results({"run_id": "20261111-jsonsys-hand20-s0", "model": "jsonsys", "data": "fixtures/handlabelled20", "seed": 0,
                   "n": 20, "json_valid": 1.0, "config": {"shared_labels": ["CITY", "PIN"], "either": ["CITY", "DISTRICT"]},
                   "field": {"micro": {"p": 0.5, "r": 0.5, "f1": 0.5}, "per_field": {}}}, d)
    write_results({"run_id": "20261111-normalizer-hand20-s0", "model": "normalizer", "data": "handlabelled20", "seed": 0, "n": 20,
                   "pin_check": {"valid": 15, "missing": 4, "not_found": 1, "mismatch": 0},
                   "filled_accuracy": {"district": {"n": 3, "correct": 2, "accuracy": 2 / 3},
                                       "state": {"n": 0, "correct": 0, "accuracy": None}}}, d)
    return d


def test_config_key_drops_the_date_and_the_seed():
    assert config_key("20261111-muril-synfull-realdev-s42") == config_key("20261120-muril-synfull-realdev-s43") == "muril-synfull-realdev"


def test_runs_table_has_one_row_per_results_file(results_dir):
    table = runs_table(load(results_dir))
    assert table.count("\n| 2026") == 5
    assert "| 20261111-ours-syn-hand20-s1 | ours | fixtures/handlabelled20 | 20 | 1.000 " in table
    assert "| 20261111-jsonsys-hand20-s0 | jsonsys | fixtures/handlabelled20 | 20 | – " in table  # no spans, so no span F1


def test_configs_table_averages_the_seeds_of_one_configuration(results_dir):
    table = configs_table(load(results_dir))
    row = next(line for line in table.splitlines() if line.startswith("| ours-syn-hand20 "))
    both = [json.loads((results_dir / f"{rid}.json").read_text())["span"]["micro"]["f1"]
            for rid in ("20261111-ours-syn-hand20-s1", "20261112-ours-syn-hand20-s2")]
    assert "| 1, 2 |" in row and f"{sum(both) / 2:.3f} ± " in row
    assert "| worse-syn-hand20 | worse | fixtures/handlabelled20 | 1 |" in table and "± –" in table  # one seed: no spread


def test_shared_table_rescores_our_systems_on_the_labels_the_other_has(results_dir):
    table = shared_table(load(results_dir), results_dir)
    theirs = json.loads((results_dir / "20261111-theirs-hand20-s0.json").read_text())["field"]["micro"]["f1"]
    assert f"| 20261111-theirs-hand20-s0 | CITY, PIN | {theirs:.3f} | 20261111-ours-syn-hand20-s1 | 1.000 |" in table
    # "worse" dropped every city, exactly what "theirs" did, so on CITY and PIN alone the two score the same
    assert f"| 20261111-theirs-hand20-s0 | CITY, PIN | {theirs:.3f} | 20261111-worse-syn-hand20-s1 | {theirs:.3f} |" in table
    assert "| 20261111-jsonsys-hand20-s0 | CITY, PIN | 0.500 | 20261111-ours-syn-hand20-s1 | 1.000 |" in table
    assert "theirs-hand20-s0 | CITY, PIN | " + f"{theirs:.3f} | 20261111-jsonsys" not in table  # only our systems are rescored


def test_compare_table_reports_a_difference_or_no_measurable_effect(results_dir):
    pairs = [("ours vs worse", "ours-syn-hand20", "worse-syn-hand20"), ("worse vs itself", "worse-syn-hand20", "worse-syn-hand20"),
             ("not run", "ours-syn-hand20", "missing-syn-hand20")]
    table = compare_table(load(results_dir), results_dir, pairs, n=50)
    rows = {line.split(" | ")[0].strip("| "): line for line in table.splitlines() if line.startswith("| ") and "---" not in line}
    assert "first is higher" in rows["ours vs worse"] and "| 1 |" not in rows["ours vs worse"]  # ours has two seeds
    assert "no measurable effect" in rows["worse vs itself"] and "0.000" in rows["worse vs itself"]
    assert "not run yet" in rows["not run"]


def test_normaliser_table_shows_the_pin_checks_and_the_fills(results_dir):
    table = normaliser_table(load(results_dir))
    assert "| 20261111-normalizer-hand20-s0 | handlabelled20 | 20 | 15 | 4 | 1 | 0 | 2 of 3 | – |" in table


def test_main_writes_the_five_tables(results_dir, tmp_path):
    main(results_dir, tmp_path, n=20)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["compare.md", "configs.md", "normaliser.md", "runs.md", "shared.md"]
    assert all("scripts/make_tables.py" in p.read_text() for p in tmp_path.iterdir())  # each says where it came from
