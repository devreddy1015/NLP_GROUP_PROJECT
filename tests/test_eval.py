import json

import pytest

from src.data import label_substrings, make_record, read_jsonl
from src.eval import (
    bootstrap_ci, evaluate, exact_match, field_scorer, field_scores, fields_from, interval, normalise_value,
    paired_bootstrap, read_predictions, slice_scores, span_scorer, span_scores, write_predictions, write_results,
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


def test_bootstrap_on_single_addresses_gives_the_numbers_already_logged():
    """Every interval in results/ so far came from this resampling; a refactor must not move them."""
    recs = read_jsonl("tests/fixtures/handlabelled20.jsonl")
    gold = [r["tags"] for r in recs]
    pred = [[t if "PIN" not in t and "STREET" not in t else "O" for t in r["tags"]] for r in recs]
    assert bootstrap_ci(gold, pred, n=200, seed=3) == pytest.approx((0.7848101265822784, 0.8727272727272727))


def test_span_scorer_gives_exactly_what_seqeval_gives():
    """The bootstraps count spans once per address instead of calling seqeval per resample; the two must agree to the bit."""
    import random

    from seqeval.metrics import f1_score
    from seqeval.scheme import IOB2

    from src.data import TAGS
    rng = random.Random(7)
    recs = read_jsonl("tests/fixtures/handlabelled20.jsonl")
    gold = [r["tags"] for r in recs] + [["O", "O"]]
    # a third of the tags replaced at random: wrong labels, dangling I- tags, spans cut in two
    pred = [[t if rng.random() > 0.33 else rng.choice(TAGS) for t in tags] for tags in gold]
    score = span_scorer(gold, pred)
    samples = [[rng.randrange(len(gold)) for _ in gold] for _ in range(60)] + [list(range(len(gold))), [20], [20, 20], [3] * 9]
    for idx in samples:
        want = f1_score([gold[i] for i in idx], [pred[i] for i in idx], average="micro", mode="strict", scheme=IOB2, zero_division=0)
        assert score(idx) == want, idx


def test_interval_resamples_whole_groups():
    seen = []
    interval(lambda idx: seen.append(idx) or 0.0, 4, n=30, seed=1, groups=["r1", "r1", "r2", "r2"])
    assert len(seen) == 30 and all(len(idx) == 4 for idx in seen)
    assert all(idx.count(0) == idx.count(1) and idx.count(2) == idx.count(3) for idx in seen)  # a respondent stays whole
    assert {tuple(sorted(set(idx))) for idx in seen} == {(0, 1), (2, 3), (0, 1, 2, 3)}


def test_a_span_of_punctuation_alone_is_not_a_field():
    assert fields_from("Pune , 411001", ["B-CITY", "B-CITY", "B-PIN"]) == {"CITY": {"pune"}, "PIN": {"411001"}}


def test_paired_bootstrap_resamples_groups_when_given_them():
    gold = [["B-CITY", "O", "B-PIN"]] * 40
    worse = [["O", "O", "O"]] * 10 + gold[10:]  # wrong on the whole of the first respondent, right on the rest
    groups = [f"R{i // 10}" for i in range(40)]
    scores = span_scorer(gold, gold), span_scorer(gold, worse)
    plain = paired_bootstrap(*scores, 40, n=300, seed=0)["ci95"]
    grouped = paired_bootstrap(*scores, 40, n=300, seed=0, groups=groups)["ci95"]
    assert grouped[1] - grouped[0] > plain[1] - plain[0]  # four respondents are far less certain than forty addresses
    assert grouped[0] == 0.0  # some resamples leave the first respondent out altogether


def test_paired_bootstrap_sees_a_real_difference_and_no_fake_one():
    gold = [["B-CITY", "O", "B-PIN"]] * 40
    worse = [["B-CITY", "O", "O"]] * 20 + [["O", "O", "O"]] * 20
    found = paired_bootstrap(span_scorer(gold, gold), span_scorer(gold, worse), 40, n=200, seed=0)
    assert found["diff"] > 0.5 and found["ci95"][0] > 0
    same = paired_bootstrap(span_scorer(gold, worse), span_scorer(gold, worse), 40, n=200, seed=0)
    assert same == {"diff": 0.0, "ci95": [0.0, 0.0]}


def test_field_scorer_agrees_with_field_scores():
    gold = [{"CITY": {"pune"}, "PIN": {"411001"}}, {"CITY": {"durg"}}]
    pred = [{"CITY": {"pune"}, "PIN": {"411002"}}, {}]
    score = field_scorer(gold, pred, ("CITY", "PIN"))
    assert score([0, 1]) == pytest.approx(field_scores(gold, pred, ("CITY", "PIN"))["micro"]["f1"])
    assert score([0, 0]) == pytest.approx(0.5) and score([1, 1]) == 0.0


def test_macro_over_labels_with_enough_spans():
    tags = [["B-CITY"]] * 30 + [["B-PIN"]]
    assert span_scores(tags, tags)["macro_min30"] == {"f1": 1.0, "labels": ["CITY"]}
    assert span_scores(tags[:5], tags[:5])["macro_min30"] == {"f1": None, "labels": []}


def test_evaluate_resamples_respondents_when_records_carry_them(records):
    pred = [r["tags"] for r in records]
    grouped = [{**r, "meta": {**r["meta"], "respondent": f"R{i // 2}"}} for i, r in enumerate(records)]
    res = evaluate(grouped, pred, run_id="x", model="m", data="d", seed=0, bootstrap_n=20)
    assert res["ci_unit"] == "respondent" and res["field_ci95"] == [1.0, 1.0]
    assert evaluate(records, pred, run_id="x", model="m", data="d", seed=0, bootstrap_n=20)["ci_unit"] == "address"
    # with mistakes in it, the interval must be the one that resamples respondents, for spans and for fields
    wrong = [["O"] * len(p) if i in (0, 1, 6) else p for i, p in enumerate(pred)]
    groups, gold = [f"R{i // 2}" for i in range(len(records))], [r["tags"] for r in records]
    res = evaluate(grouped, wrong, run_id="x", model="m", data="d", seed=0, bootstrap_n=60)
    assert res["ci95"] == list(bootstrap_ci(gold, wrong, n=60, seed=0, groups=groups))
    assert res["ci95"] != list(bootstrap_ci(gold, wrong, n=60, seed=0))
    fields = [[fields_from(r["text"], tags) for r, tags in zip(records, side)] for side in (gold, wrong)]
    assert res["field_ci95"] == list(interval(field_scorer(*fields), len(records), 60, 0, groups))
    assert res["field_ci95"] != list(interval(field_scorer(*fields), len(records), 60, 0))


def test_predictions_round_trip(records, tmp_path):
    pred = [r["tags"] for r in records]
    write_predictions(records, pred, "20261111-x-s0", tmp_path)
    assert read_predictions(tmp_path / "20261111-x-s0.pred.jsonl", records) == pred
    with pytest.raises(ValueError, match="different addresses"):
        read_predictions(tmp_path / "20261111-x-s0.pred.jsonl", records[:-1])
    with pytest.raises(ValueError, match="different addresses"):  # the same addresses in another order
        read_predictions(tmp_path / "20261111-x-s0.pred.jsonl", records[::-1])
