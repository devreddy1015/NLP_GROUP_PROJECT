import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import label_substrings, make_record, read_jsonl
from src.normalize.gazetteer import Gazetteer, build
from src.normalize.normalize import expand, normalise, parse_landmark, score

FIXTURE = Path("tests/fixtures/pincode_sample.csv")
HAND20 = Path("tests/fixtures/handlabelled20.jsonl")


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    out = tmp_path_factory.mktemp("gaz") / "p.csv"
    build(raw=FIXTURE, out=out)
    return Gazetteer.load(out)


def rec(text, pairs):
    return make_record("x", text, label_substrings(text, pairs), "synthetic", "dev")


@pytest.mark.parametrize("raw,clean", [
    ("Shanti Apts", "Shanti Apartments"), ("MG Rd.", "MG Road"), ("sec 10", "Sector 10"), ("Govt. Hosp", "Government Hospital"),
    ("G.E. Road", "G.E. Road"), ("Shop No. 4", "Shop No. 4"), ("SUPELA CHOWK", "Supela Chowk"), ("supela", "Supela"),
    ("Hyd", "Hyderabad"), ("b'lore", "Bangalore"), ("Off MG Road", "Off MG Road"), ("Col. Sharma Marg", "Col. Sharma Marg"),
    ("Cal Road", "Cal Road"), ("Nehru Nagar Ext", "Nehru Nagar Extension"), ("Ph 2", "Phase 2"), ("Stn Rd", "Station Road"),
])
def test_expand(raw, clean):
    assert expand(raw) == clean


@pytest.mark.parametrize("span,relation,anchor", [
    ("near Hanuman Mandir", "near", "Hanuman Mandir"), ("nr hanuman mandir", "near", "Hanuman Mandir"),
    ("Opp. petrol pump", "opposite", "Petrol Pump"), ("in front of Clock Tower", "opposite", "Clock Tower"),
    ("behind old bus stand", "behind", "Old Bus Stand"), ("next to HDFC Bank", "beside", "HDFC Bank"),
    ("hanuman mandir ke paas", "near", "Hanuman Mandir"), ("SBI ATM ke samne", "opposite", "SBI ATM"),
    ("Police Stn ke peeche", "behind", "Police Station"), ("Hanuman Mandir", None, "Hanuman Mandir"),
    ("near hanuman mandir ke paas", "near", "Hanuman Mandir"), ("Opp.Park", "opposite", "Park"),
])
def test_parse_landmark(span, relation, anchor):
    assert parse_landmark(span) == {"relation": relation, "anchor": anchor}


def test_normalise_fills_district_and_state_from_pin(gaz):
    r = rec("flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490 023",
            [("flat 3b", "HOUSE"), ("shanti apts", "BUILDING"), ("behind SBI atm", "LANDMARK"), ("nr hanuman mandir", "LANDMARK"),
             ("supela", "LOCALITY"), ("bhilai", "CITY"), ("490 023", "PIN")])
    out = normalise(r["text"], r["tags"], gaz)
    assert out["house"] == "Flat 3b" and out["building"] == "Shanti Apartments"
    assert out["landmarks"] == [{"relation": "behind", "anchor": "SBI atm"}, {"relation": "near", "anchor": "Hanuman Mandir"}]
    assert out["locality"] == "Supela" and out["city"] == "Bhilai"
    assert out["pin"] == "490023" and out["pin_check"] == "valid"
    assert out["district"] == "Durg" and out["state"] == "Chhattisgarh" and out["filled"] == ["district", "state"]
    assert out["locality_known"] is True


@pytest.mark.parametrize("text,pairs,check,state", [
    ("Civic Centre, Bhilai, C.G. 490006", [("Bhilai", "CITY"), ("C.G.", "STATE"), ("490006", "PIN")], "valid", "Chhattisgarh"),
    ("Civic Centre, Bhilai, MH 490006", [("Bhilai", "CITY"), ("MH", "STATE"), ("490006", "PIN")], "mismatch", "Maharashtra"),
    ("Civic Centre, Distt Raipur, 490006", [("Raipur", "DISTRICT"), ("490006", "PIN")], "mismatch", None),  # no filling on a mismatch
    ("Civic Centre, Bhilai, 000000", [("Bhilai", "CITY"), ("000000", "PIN")], "not_found", None),
    ("hosur road, b'lore-68", [("hosur road", "STREET"), ("b'lore", "CITY"), ("68", "PIN")], "not_found", None),
    ("Civic Centre, Bhilai, Chhattisgarh", [("Bhilai", "CITY"), ("Chhattisgarh", "STATE")], "missing", "Chhattisgarh"),
    # a misspelt name is not a contradiction; a different state still is
    ("Civic Centre, Bhilai, Chhatisgarh 490006", [("Bhilai", "CITY"), ("Chhatisgarh", "STATE"), ("490006", "PIN")], "valid", "Chhattisgarh"),
    ("Civic Centre, Bhilai, Madhya Pradewsh 490006", [("Bhilai", "CITY"), ("Madhya Pradewsh", "STATE"), ("490006", "PIN")], "mismatch", "Madhya Pradesh"),
    ("Shop 1, Distt Patiyala, 147111", [("Patiyala", "DISTRICT"), ("147111", "PIN")], "valid", "Punjab"),
    ("भिलाई छत्तीसगढ़ ४९००२३", [("छत्तीसगढ़", "STATE"), ("४९००२३", "PIN")], "valid", "छत्तीसगढ़"),  # not Latin script: left unchecked
])
def test_pin_check(gaz, text, pairs, check, state):
    r = rec(text, pairs)
    out = normalise(r["text"], r["tags"], gaz)
    assert out["pin_check"] == check and out["state"] == state


@pytest.mark.parametrize("district,pin", [("Durg District", "490023"), ("Distt. Durg", "490023"), ("Bangalore", "560001"), ("Bengaluru", "560001")])
def test_district_variants_are_not_mismatches(gaz, district, pin):
    r = rec(f"Shop 1, {district}, {pin}", [(district.replace("Distt. ", ""), "DISTRICT"), (pin, "PIN")])
    assert normalise(r["text"], r["tags"], gaz)["pin_check"] == "valid"


def test_pin_spanning_two_districts_accepts_either_and_fills_neither():
    # a made-up PIN whose offices sit in two districts, as 1,168 of the 19,300 real PINs do
    two = Gazetteer(pd.DataFrame({"pin": ["799999"] * 3, "office": ["Thanlon", "Parbung", "Singngat"], "office_type": ["BO"] * 3,
                                  "delivery": ["Delivery"] * 3, "city": ["Manipur"] * 3, "state": ["Manipur"] * 3,
                                  "district": ["Pherzawl", "Pherzawl", "Churachandpur"], "lat": [""] * 3, "lon": [""] * 3}))
    r = rec("Shop 1, Distt Churachandpur, 799999", [("Churachandpur", "DISTRICT"), ("799999", "PIN")])
    assert normalise(r["text"], r["tags"], two)["pin_check"] == "valid"  # the less common district is still right
    r = rec("Shop 1, 799999", [("799999", "PIN")])
    out = normalise(r["text"], r["tags"], two)
    assert out["pin_check"] == "valid" and out["district"] is None and out["filled"] == ["state"]


def test_score_accepts_a_district_written_another_way(gaz):
    r = {**rec("MG Road, 560001", [("MG Road", "STREET"), ("560001", "PIN")]), "meta": {"district": "Bangalore"}}
    res = score([r], [r["tags"]], gaz, run_id="x", data="y")
    assert res["filled_accuracy"]["district"] == {"n": 1, "correct": 1, "accuracy": 1.0}  # filled "Bengaluru Urban"


def test_prefers_a_pin_the_directory_knows_and_reads_devanagari_digits(gaz):
    r = rec("Plot 123456, Bhilai 490023", [("123456", "PIN"), ("490023", "PIN")])
    assert normalise(r["text"], r["tags"], gaz)["pin"] == "490023"
    r = rec("भिलाई ४९००२३", [("४९००२३", "PIN")])
    out = normalise(r["text"], r["tags"], gaz)
    assert out["pin"] == "490023" and out["pin_check"] == "valid"


def test_generic_locality_words_do_not_count_as_known(gaz):
    r = rec("Shanti Nagar, 490023", [("Shanti Nagar", "LOCALITY"), ("490023", "PIN")])
    assert normalise(r["text"], r["tags"], gaz)["locality_known"] is False


def test_unknown_locality_is_flagged_not_mismatched(gaz):
    r = rec("Xyz Colony, 490023", [("Xyz Colony", "LOCALITY"), ("490023", "PIN")])
    out = normalise(r["text"], r["tags"], gaz)
    assert out["pin_check"] == "valid" and out["locality_known"] is False


def test_score_on_hand20(gaz, tmp_path):
    recs = read_jsonl(HAND20)
    res = score(recs, [r["tags"] for r in recs], gaz, run_id="20261110-normalizer-hand20-s0", data="hand20")
    assert res["n"] == 20 and sum(res["pin_check"].values()) == 20 and res["pin_check"]["valid"] >= 1  # fixture gazetteer knows few PINs
    assert set(res["filled_accuracy"]) == {"district", "state"}
    assert res["filled_accuracy"]["state"]["n"] >= 0
