import json
from pathlib import Path

import pytest

from src.data import read_jsonl, tags_to_spans, tokenize, validate
from src.models.rules import predict, run
from src.normalize.gazetteer import Gazetteer, build

FIXTURE = Path("tests/fixtures/pincode_sample.csv")
HAND20 = Path("tests/fixtures/handlabelled20.jsonl")


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    out = tmp_path_factory.mktemp("gaz") / "p.csv"
    build(raw=FIXTURE, out=out)
    return Gazetteer.load(out)


def labelled(text, tags):
    return [(text[s:e], label) for s, e, label in tags_to_spans(tokenize(text), tags)]


@pytest.mark.parametrize("text,expected", [
    ("Shop No. 4, Gupta Complex, MG Road, near Hanuman Mandir, Supela Bhilai, Durg, District Durg, Chhattisgarh 490023",
     [("Shop No. 4", "HOUSE"), ("Gupta Complex", "BUILDING"), ("MG Road", "STREET"), ("near Hanuman Mandir", "LANDMARK"),
      ("Supela Bhilai", "LOCALITY"), ("Durg", "CITY"), ("Durg", "DISTRICT"), ("Chhattisgarh", "STATE"), ("490023", "PIN")]),
    ("behind SBI atm nr hanuman mandir, bhilai 490 023",
     [("behind SBI atm", "LANDMARK"), ("nr hanuman mandir", "LANDMARK"), ("490 023", "PIN")]),
    ("Civic Centre Library, Sector 6, Bhilai, Distt. Durg, C.G.",
     [("Sector 6", "LOCALITY"), ("Durg", "DISTRICT"), ("C.G.", "STATE")]),
    ("sai mobile shop, main road, kohka, c.g",
     [("main road", "STREET"), ("c.g", "STATE")]),
    ("Plot 12, Phase 2, Bhilai 490026",
     [("Plot 12", "HOUSE"), ("Phase 2", "LOCALITY"), ("490026", "PIN")]),
    ("Tower C, Sunshine Enclave, Andheri West, Mumbai, MH 400058",
     [("Tower C", "BUILDING"), ("Sunshine Enclave", "BUILDING"), ("MH", "STATE"), ("400058", "PIN")]),
    ("hanuman mandir ke paas, gali no 3, shanti nagar",
     [("hanuman mandir ke paas", "LANDMARK"), ("gali no 3", "STREET"), ("shanti nagar", "LOCALITY")]),
    ("shop near temple or bus stand, raipur", [("near temple or bus stand", "LANDMARK")]),  # lowercase "or" is not Odisha
    ("Kohka Bhilai, 490023", [("Kohka Bhilai", "LOCALITY"), ("490023", "PIN")]),  # locality via the PIN's offices
    ("", []),
])
def test_rules_predict(gaz, text, expected):
    tags = predict(text, gaz)
    assert labelled(text, tags) == expected


def test_punctuated_office_name_matches():
    import pandas as pd
    g = Gazetteer(pd.DataFrame({"pin": ["494001"], "office": ["Bade Kilepal-2"], "office_type": ["BO"], "delivery": ["Delivery"],
                                "city": ["Jagdalpur"], "district": ["Bastar"], "state": ["Chhattisgarh"], "lat": [""], "lon": [""]}))
    text = "Bade Kilepal-2, 494001"
    assert labelled(text, predict(text, g)) == [("Bade Kilepal-2", "LOCALITY"), ("494001", "PIN")]


def test_predictions_are_valid_iob2_on_hand20(gaz):
    for rec in read_jsonl(HAND20):
        pred = predict(rec["text"], gaz)
        validate({**rec, "tags": pred})


def test_run_writes_results_file(gaz, tmp_path):
    res = run(HAND20, run_id="20261110-rules-hand20-s0", results_dir=tmp_path, gaz=gaz, bootstrap_n=20)
    saved = json.loads((tmp_path / "20261110-rules-hand20-s0.json").read_text())
    assert saved["n"] == 20 and saved["model"] == "rules" and saved["data"] == "handlabelled20"
    assert 0 < saved["span"]["micro"]["f1"] < 1 and saved["span"]["per_label"]["PIN"]["f1"] > 0.5
    assert res["run_id"] == saved["run_id"]
    assert len((tmp_path / "20261110-rules-hand20-s0.pred.jsonl").read_text().splitlines()) == 20  # tags kept per address
