import json
from pathlib import Path

import pytest

from scripts import make_tables
from scripts.make_tables import (
    compare_table, config_key, configs_table, load, main, normaliser_table, runs_table, shared_table,
)
from src.data import read_jsonl, write_jsonl
from src.eval import evaluate, write_predictions, write_results
from src.models.zeroshot import only

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")
OURS, WORSE, THEIRS = "20261111-ours-syn-hand20-s1", "20261111-worse-syn-hand20-s1", "20261111-theirs-hand20-s0"


def f1(results_dir, run_id, kind="span"):
    return json.loads((results_dir / f"{run_id}.json").read_text())[kind]["micro"]["f1"]


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory):
    """Results as the model scripts write them: our system with two seeds, a worse one, an existing tagger, a JSON system."""
    d = tmp_path_factory.mktemp("results")
    recs = read_jsonl(HAND20)
    gold = [r["tags"] for r in recs]
    no_pin = [[t if "PIN" not in t else "O" for t in tags] for tags in gold]
    no_city = [[t if "CITY" not in t else "O" for t in tags] for tags in gold]

    def save(run_id, model, records, pred, seed, config=None, data="fixtures/handlabelled20"):
        write_results(evaluate(records, pred, run_id=run_id, model=model, data=data, seed=seed, config=config, bootstrap_n=10), d)
        write_predictions(records, pred, run_id, d)

    save("20261101-ours-syn-hand20-s1", "ours", recs, no_city, 1)  # an earlier run of the same configuration and seed
    save(OURS, "ours", recs, gold, 1)
    save("20261112-ours-syn-hand20-s2", "ours", recs, no_pin, 2)  # a second seed, run a day later
    save(WORSE, "worse", recs, no_city, 1)
    save("20261111-ours-tune-a-syn-hand20-s1", "ours", recs, gold, 1)  # a tuning run
    save("20261111-ours-syn-elsewhere-s1", "ours", recs, gold, 1, data="elsewhere/handlabelled20")  # same file name, other data
    masked = [{**r, "tags": only(r["tags"], {"CITY", "PIN"})} for r in recs]
    save(THEIRS, "theirs", masked, [only(p, {"CITY", "PIN"}) for p in no_city], 0, config={"shared_labels": ["CITY", "PIN"]})
    write_results({"run_id": "20261111-jsonsys-hand20-s0", "model": "jsonsys", "data": "fixtures/handlabelled20", "seed": 0,
                   "n": 20, "json_valid": 1.0, "field": {"micro": {"p": 0.5, "r": 0.5, "f1": 0.5}, "per_field": {}},
                   "config": {"shared_labels": ["CITY", "PIN"], "either": ["CITY", "DISTRICT"],
                              "mapping": {"city": "CITY", "pin": "PIN"}}}, d)
    (d / "20261111-jsonsys-hand20-s0.pred.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "output": '{"city": "Nowhere"}'}) + "\n" for r in recs), encoding="utf-8")
    write_results({"run_id": "20261111-normalizer-hand20-s0", "model": "normalizer", "data": "handlabelled20", "seed": 0, "n": 20,
                   "pin_check": {"valid": 15, "missing": 4, "not_found": 1, "mismatch": 0},
                   "filled_accuracy": {"district": {"n": 3, "correct": 2, "accuracy": 2 / 3},
                                       "state": {"n": 0, "correct": 0, "accuracy": None}}}, d)
    return d


def test_config_key_drops_the_date_and_the_seed():
    assert config_key("20261111-muril-synfull-realdev-s42") == config_key("20261120-muril-synfull-realdev-s43") == "muril-synfull-realdev"


def test_load_keeps_only_the_newest_run_of_a_configuration_and_seed(results_dir):
    ids = [r["run_id"] for r in load(results_dir)]
    assert OURS in ids and "20261101-ours-syn-hand20-s1" not in ids and len(ids) == len(set(ids)) == 8


def test_runs_table_has_one_row_per_results_file(results_dir):
    table = runs_table(load(results_dir))
    assert table.count("\n| 2026") == 7  # the normaliser has its own table
    assert f"| {OURS} | ours | fixtures/handlabelled20 | 20 | 1.000 " in table
    assert "| 20261111-jsonsys-hand20-s0 | jsonsys | fixtures/handlabelled20 | 20 | – " in table  # no spans, so no span F1
    assert "Span macro F1, labels with 30+ spans" in table.splitlines()[0]


def test_configs_table_averages_the_seeds_of_one_configuration(results_dir):
    table = configs_table(load(results_dir))
    row = next(line for line in table.splitlines() if line.startswith("| ours-syn-hand20 "))
    both = [f1(results_dir, OURS), f1(results_dir, "20261112-ours-syn-hand20-s2")]
    assert "| 1, 2 |" in row and f"{sum(both) / 2:.3f} ± " in row  # the superseded run is not a third seed
    assert "| worse-syn-hand20 | worse | fixtures/handlabelled20 | 1 |" in table and "± –" in table  # one seed: no spread


def test_shared_table_rescores_our_systems_on_the_labels_the_other_has(results_dir):
    table = shared_table(load(results_dir), results_dir, n=50)
    theirs = f1(results_dir, THEIRS, "field")
    assert f"| {THEIRS} | CITY, PIN | {theirs:.3f} | {OURS} | 1.000 | {1 - theirs:+.3f} | " in table
    ours_row = next(line for line in table.splitlines() if THEIRS in line and OURS in line)
    assert "ours is higher" in ours_row
    # "worse" dropped every city, exactly what "theirs" did, so on CITY and PIN alone the two are the same
    worse_row = next(line for line in table.splitlines() if THEIRS in line and WORSE in line)
    assert f"| {theirs:.3f} | {WORSE} | {theirs:.3f} | +0.000 | " in worse_row and "no measurable effect" in worse_row
    json_row = next(line for line in table.splitlines() if "jsonsys" in line and OURS in line)
    assert "| CITY, PIN | 0.500 | " in json_row and "| 1.000 | " in json_row and "ours is higher" in json_row
    assert "tune" not in table and "elsewhere" not in table  # tuning runs and runs on other data are not compared
    assert f"{THEIRS} | CITY, PIN | {theirs:.3f} | 20261111-jsonsys" not in table  # only our systems are rescored


def test_shared_table_refuses_results_that_are_older_than_the_gold_file(results_dir):
    stale = [{**r, "n": 19} if r["run_id"] == THEIRS else r for r in load(results_dir)]
    with pytest.raises(ValueError, match="19"):
        shared_table(stale, results_dir, n=10)


def test_compare_table_reports_a_difference_or_no_measurable_effect(results_dir):
    pairs = [("ours vs worse", "ours-syn-hand20", "worse-syn-hand20"), ("worse vs ours", "worse-syn-hand20", "ours-syn-hand20"),
             ("worse vs itself", "worse-syn-hand20", "worse-syn-hand20"), ("not run", "ours-syn-hand20", "missing-syn-hand20")]
    table = compare_table(load(results_dir), results_dir, pairs, n=50)
    rows = {line.split(" | ")[0].strip("| "): line for line in table.splitlines() if line.startswith("| ") and "---" not in line}
    mean_ours = (f1(results_dir, OURS) + f1(results_dir, "20261112-ours-syn-hand20-s2")) / 2  # both seeds, not the first
    assert "first is higher" in rows["ours vs worse"] and "| 2 and 1 |" in rows["ours vs worse"]
    assert f"| {mean_ours - f1(results_dir, WORSE):+.3f} |" in rows["ours vs worse"]
    assert "second is higher" in rows["worse vs ours"]
    assert "no measurable effect" in rows["worse vs itself"] and "+0.000" in rows["worse vs itself"]
    assert "not run yet" in rows["not run"]


def test_compare_table_resamples_respondents_when_the_gold_file_names_them(results_dir, tmp_path, monkeypatch):
    grouped, seen = tmp_path / "gold.jsonl", {}
    write_jsonl(grouped, [{**r, "meta": {"respondent": f"R{i // 5}"}} for i, r in enumerate(read_jsonl(HAND20))])
    monkeypatch.setattr(make_tables, "gold_path", lambda data: grouped)
    monkeypatch.setattr(make_tables, "paired_bootstrap",
                        lambda a, b, n_items, n=1000, groups=None: seen.update(groups=groups) or {"diff": 0.0, "ci95": [0.0, 0.0]})
    compare_table(load(results_dir), results_dir, [("x", "ours-syn-hand20", "worse-syn-hand20")], n=10)
    assert seen["groups"] == [f"R{i // 5}" for i in range(20)]


def test_normaliser_table_shows_the_pin_checks_and_the_fills(results_dir):
    table = normaliser_table(load(results_dir))
    assert "| 20261111-normalizer-hand20-s0 | handlabelled20 | 20 | 15 | 4 | 1 | 0 | 2 of 3 | – |" in table


def test_main_writes_the_five_tables(results_dir, tmp_path):
    main(results_dir, tmp_path, n=20)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["compare.md", "configs.md", "normaliser.md", "runs.md", "shared.md"]
    assert all("scripts/make_tables.py" in p.read_text() for p in tmp_path.iterdir())  # each says where it came from
