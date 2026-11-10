import json

import pytest

from src.data import label_substrings, make_record
from src.eval import (
    bootstrap_ci, evaluate, exact_match, field_scores, fields_from, normalise_value,
    slice_scores, span_scores, write_results,
)
from tests.test_data import ROUND_TRIP


@pytest.fixture
def records():
    return [make_record(f"hand-{i:04d}", t, label_substrings(t, p), "synthetic", "dev",
                        slices=["landmark"] if any(l == "LANDMARK" for _, l in p) else [])
            for i, (t, p) in enumerate(ROUND_TRIP)]


def test_gold_against_gold_scores_one(records):
    gold = [r["tags"] for r in records]
    s = span_scores(gold, gold)
    assert s["micro"]["f1"] == 1.0 and s["macro"]["f1"] == 1.0
    assert all(v["f1"] == 1.0 for v in s["per_label"].values())
    assert s["per_label"]["LANDMARK"]["support"] == 5
    assert exact_match(gold, gold) == 1.0


def test_all_o_scores_zero(records):
    gold = [r["tags"] for r in records]
    pred = [["O"] * len(t) for t in gold]
    s = span_scores(gold, pred)
    assert s["micro"]["f1"] == 0.0 and s["macro"]["f1"] == 0.0
    assert exact_match(gold, pred) == 0.0


def test_partial_prediction_is_strict_iob2(records):
    gold = [records[0]["tags"]]
    pred = [list(gold[0])]
    pred[0][0] = "O"  # drop the B- of HOUSE, leaving a dangling I-HOUSE: strict mode must not count it
    s = span_scores(gold, pred)
    assert s["per_label"]["HOUSE"]["f1"] == 0.0
    assert s["per_label"]["PIN"]["f1"] == 1.0
    assert 0 < s["micro"]["f1"] < 1


def test_normalise_value():
    assert normalise_value("  Opp. Ganesh  Temple, ") == "opp ganesh temple"
    assert normalise_value("C.G.") == "cg"
    assert normalise_value("हनुमान मंदिर") == "हनुमान मंदिर"


def test_fields_from_tokens_and_tags(records):
    r = records[0]
    f = fields_from(r["text"], r["tags"])
    assert f["LANDMARK"] == {"behind sbi atm", "nr hanuman mandir"}
    assert f["PIN"] == {"490023"} and f["CITY"] == {"bhilai"}
    assert "DISTRICT" not in f


def test_field_scores_sets_and_shared_fields():
    gold = [{"CITY": {"bhilai"}, "PIN": {"490023"}, "STATE": {"chhattisgarh"}}]
    pred = [{"CITY": {"bhilai", "durg"}, "PIN": set(), "STATE": {"chhattisgarh"}}]
    s = field_scores(gold, pred, fields=("CITY", "PIN", "STATE"))
    assert s["per_field"]["CITY"] == {"p": 0.5, "r": 1.0, "f1": pytest.approx(2 / 3), "support": 1}
    assert s["per_field"]["PIN"]["f1"] == 0.0
    assert s["micro"]["p"] == pytest.approx(2 / 3) and s["micro"]["r"] == pytest.approx(2 / 3)
    only_state = field_scores(gold, pred, fields=("STATE",))
    assert only_state["micro"]["f1"] == 1.0


def test_bootstrap_ci_brackets_point_estimate(records):
    gold = [r["tags"] for r in records]
    pred = [list(t) for t in gold]
    pred[0] = ["O"] * len(pred[0])
    point = span_scores(gold, pred)["micro"]["f1"]
    lo, hi = bootstrap_ci(gold, pred, n=50, seed=0)
    assert lo <= point <= hi and lo < hi
    assert bootstrap_ci(gold, pred, n=50, seed=0) == (lo, hi)  # seeded


def test_slice_scores(records):
    gold = [r["tags"] for r in records]
    s = slice_scores(records, gold)
    assert s["landmark"]["n"] == 4 and s["landmark"]["f1"] == 1.0


def test_evaluate_writes_results_file(records, tmp_path):
    gold = [r["tags"] for r in records]
    res = evaluate(records, gold, run_id="20261110-rules-hand10-s0", model="rules", data="hand10", seed=0,
                   config={"k": 1}, bootstrap_n=20)
    path = write_results(res, results_dir=tmp_path)
    saved = json.loads(path.read_text())
    assert path.name == "20261110-rules-hand10-s0.json"
    assert saved["span"]["micro"]["f1"] == 1.0 and saved["exact_match"] == 1.0
    assert saved["n"] == 10 and saved["config"] == {"k": 1} and saved["label_support"]["PIN"] == 7
    assert saved["field"]["micro"]["f1"] == 1.0 and saved["slices"]["landmark"]["n"] == 4
    assert len(saved["ci95"]) == 2


def test_evaluate_rejects_length_mismatch(records):
    with pytest.raises(ValueError):
        evaluate(records, [["O"]], run_id="x", model="m", data="d", seed=0)
