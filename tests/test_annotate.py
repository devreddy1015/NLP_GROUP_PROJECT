import json
from pathlib import Path

import pytest

from src.annotate import agreement, form_to_tasks, labelstudio_to_jsonl, snap_span
from src.data import label_substrings, make_record, read_jsonl, tokenize, write_jsonl

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")

CSV = """respondent,place,address,pin,city,state
r1,Shree Sweets,"shop 4, nehru nagar, bhilai 490020",490020,Bhilai,Chhattisgarh
r1,IIT Bhilai,"kutelabhata, durg",491001,Durg,Chhattisgarh
r1,SHREE  SWEETS,"duplicate place, should be dropped",490020,Bhilai,Chhattisgarh
r2,Big Bazaar,"surya mall, junwani road, bhilai",490020,Bhilai,CG
r2,Ram Mandir,"vaishali nagar bhilai",490023,Bhilai,CG
r3,Apollo Pharmacy,"station road raipur",492001,Raipur,CG
r3,Jail Road Clinic,,492001,Raipur,CG
r4,Meenakshi Temple,"east chitrai st, madurai",625001,Madurai,TN
r5,Petrol Pump,"NH 53, kumhari",490042,Durg,CG
r6,,"no place given, bhilai",490020,Bhilai,CG
r6,,"no place given either, durg",491001,Durg,CG
r7,Sai Mandir,"sai mandir raipur",492001,Raipur,CG
r7,Sai Mandir,"sai mandir bhilai",490023,Bhilai,CG
"""


def test_form_to_tasks_splits_by_respondent_and_dedups(tmp_path):
    csv = tmp_path / "form.csv"
    csv.write_text(CSV)
    counts = form_to_tasks(csv, tmp_path, n_dev=2, n_pool=2, seed=0)
    assert sum(counts.values()) == 11  # 13 rows minus one empty address minus one duplicate place; blank places and same-name-other-city stay
    tasks = {split: json.loads((tmp_path / f"tasks_{split}.json").read_text()) for split in counts}
    by_resp = {}
    for split, items in tasks.items():
        for t in items:
            assert set(t["data"]) >= {"item_id", "text", "place", "pin", "city", "state", "respondent", "split"}
            assert t["data"]["split"] == split
            by_resp.setdefault(t["data"]["respondent"], set()).add(split)
    assert all(len(s) == 1 for s in by_resp.values())  # a respondent is never in two splits
    assert counts["dev"] >= 2 and counts["train"] >= 2 and counts["test"] >= 1
    ids = [t["data"]["item_id"] for items in tasks.values() for t in items]
    assert len(set(ids)) == 11 and all(i.startswith("real-") for i in ids)


def test_form_to_tasks_requires_columns(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("place,address\nx,y\n")
    with pytest.raises(ValueError):
        form_to_tasks(csv, tmp_path)


def test_snap_span_expands_to_token_boundaries():
    text = "flat 3b, shanti apts"
    toks = tokenize(text)
    assert snap_span(text, toks, 1, 6, "HOUSE") == (0, 7, "HOUSE")      # partial tokens -> whole tokens
    assert snap_span(text, toks, 9, 20, "BUILDING") == (9, 20, "BUILDING")
    assert snap_span(text, toks, 8, 16, "BUILDING") == (9, 15, "BUILDING")  # surrounding spaces trimmed -> exactly "shanti"
    with pytest.raises(ValueError):
        snap_span(text, toks, 3, 3, "HOUSE")


def _ls_export(items):
    return [{"id": i, "data": d, "annotations": [] if res is None else [{"id": 100 + i, "completed_by": 1, "result": res}]}
            for i, (d, res) in enumerate(items, 1)]


def test_labelstudio_export_to_jsonl(tmp_path):
    text = "flat 3b, shanti apts, near SBI atm, supela bhilai 490023"
    res = [
        {"type": "labels", "value": {"start": 0, "end": 7, "text": "flat 3b", "labels": ["HOUSE"]}},
        {"type": "labels", "value": {"start": 9, "end": 21, "text": "shanti apts,", "labels": ["BUILDING"]}},  # sloppy end on the comma
        {"type": "labels", "value": {"start": 22, "end": 34, "text": "near SBI atm", "labels": ["LANDMARK"]}},
        {"type": "labels", "value": {"start": 50, "end": 56, "text": "490023", "labels": ["PIN"]}},
        {"type": "choices", "value": {"choices": ["landmark"]}},
    ]
    data = {"item_id": "real-0007", "text": text, "place": "X", "pin": "490023", "city": "Bhilai", "state": "CG", "respondent": "r1", "split": "dev"}
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_ls_export([(data, res), ({**data, "item_id": "real-0008", "text": "unannotated"}, None),
                                             ({**data, "item_id": "real-0009", "text": "all O item"}, [])])))
    out = tmp_path / "dev.jsonl"
    recs = labelstudio_to_jsonl(export, out)
    assert [r["id"] for r in recs] == ["real-0007", "real-0009"]  # unannotated skipped; submitted with no spans kept as all-O
    assert recs[1]["tags"] == ["O", "O", "O"]
    r = read_jsonl(out)[0]
    assert r["source"] == "real" and r["split"] == "dev" and r["slices"] == ["landmark"] and r["meta"]["pin"] == "490023"
    assert r["tags"][:6] == ["B-HOUSE", "I-HOUSE", "O", "B-BUILDING", "I-BUILDING", "O"]


def test_labelstudio_reports_every_bad_item(tmp_path):
    text = "supela bhilai"
    overlap = [{"type": "labels", "value": {"start": 0, "end": 6, "labels": ["LOCALITY"]}},
               {"type": "labels", "value": {"start": 0, "end": 13, "labels": ["CITY"]}}]
    split_token = [{"type": "labels", "value": {"start": 0, "end": 1, "labels": ["HOUSE"]}},
                   {"type": "labels", "value": {"start": 1, "end": 2, "labels": ["BUILDING"]}}]  # "3" and "b" of one token
    export = tmp_path / "e.json"
    export.write_text(json.dumps(_ls_export([({"item_id": "real-0001", "text": text, "split": "dev"}, overlap),
                                             ({"item_id": "real-0002", "text": "3b flat", "split": "dev"}, split_token)])))
    with pytest.raises(ValueError, match=r"real-0001.*\n.*real-0002"):
        labelstudio_to_jsonl(export, tmp_path / "o.jsonl")
    assert not (tmp_path / "o.jsonl").exists()


def test_labelstudio_picks_annotator(tmp_path):
    text = "supela bhilai"
    task = {"id": 1, "data": {"item_id": "real-0001", "text": text, "split": "dev"}, "annotations": [
        {"id": 1, "completed_by": 1, "result": [{"type": "labels", "value": {"start": 0, "end": 6, "labels": ["LOCALITY"]}}]},
        {"id": 2, "completed_by": 2, "result": [{"type": "labels", "value": {"start": 0, "end": 6, "labels": ["CITY"]}}]}]}
    export = tmp_path / "e.json"
    export.write_text(json.dumps([task]))
    assert labelstudio_to_jsonl(export, tmp_path / "a.jsonl", annotator=2)[0]["tags"][0] == "B-CITY"
    assert labelstudio_to_jsonl(export, tmp_path / "b.jsonl")[0]["tags"][0] == "B-LOCALITY"


def test_agreement_identical_and_disagreeing(tmp_path):
    recs = read_jsonl(HAND20)
    a = tmp_path / "a.jsonl"
    write_jsonl(a, recs)
    same = agreement(a, a)
    assert same["n"] == 20 and same["span_f1_micro"] == 1.0 and same["token_kappa"] == pytest.approx(1.0)
    b = tmp_path / "b.jsonl"
    write_jsonl(b, [{**r, "tags": ["O"] * len(r["tags"])} for r in recs])
    diff = agreement(a, b)
    assert diff["span_f1_micro"] == 0.0 and diff["token_kappa"] <= 0.0
    c = tmp_path / "c.jsonl"
    write_jsonl(c, [{**recs[0], "id": "other-0001"}])
    with pytest.raises(ValueError):
        agreement(a, c)
