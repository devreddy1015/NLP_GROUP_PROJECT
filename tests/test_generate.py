import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import LABELS, read_jsonl, tags_to_spans, tokenize, validate
from src.generate import CONFIGS, Generator, generate, split_pins
from src.generate import lexicon as L
from src.normalize.gazetteer import build

FIXTURE = Path("tests/fixtures/pincode_sample.csv")


@pytest.fixture(scope="module")
def gaz_df(tmp_path_factory):
    return build(raw=FIXTURE, out=tmp_path_factory.mktemp("gaz") / "p.csv")


def labels_of(rec):
    return [label for _, _, label in tags_to_spans(tokenize(rec["text"]), rec["tags"])]


def test_clean_record_has_all_nine_fields_in_fixed_order(gaz_df):
    rec = Generator(gaz_df, CONFIGS["clean"], seed=1).record("train", 0)
    validate(rec)
    assert labels_of(rec) == list(LABELS)
    assert rec["source"] == "synthetic" and rec["split"] == "train" and rec["noise"] == []
    assert rec["id"] == "syn-clean-train-000000"
    assert set(rec["meta"]) >= {"pin", "city", "district", "state"}


def test_clean_record_prefixes_place_name_tagged_o_and_uses_cue_words(gaz_df):
    rec = Generator(gaz_df, CONFIGS["clean"], seed=3).record("dev", 7)
    first_comma = rec["tokens"].index(",")
    assert first_comma > 0 and all(t == "O" for t in rec["tags"][:first_comma])
    spans = {label: rec["text"][s:e] for s, e, label in tags_to_spans(tokenize(rec["text"]), rec["tags"])}
    assert spans["LANDMARK"].startswith("near ")
    assert f"District {spans['DISTRICT']}" in rec["text"]
    assert rec["text"].endswith(f"{spans['STATE']} {spans['PIN']}")
    assert spans["PIN"] == rec["meta"]["pin"]


def test_locality_drops_the_city_it_ends_with():
    gaz = pd.DataFrame({"pin": ["490023", "490006", "490001"], "office": ["Supela Bhilai", "Civic Centre Bhilai", "Kohka Bhilai"],
                        "office_type": "SO", "delivery": "Delivery", "city": "Bhilai", "district": "Durg",
                        "state": "Chhattisgarh", "lat": "", "lon": ""})
    rec = Generator(gaz, CONFIGS["clean"], seed=0).record("train", 0)
    spans = {label: rec["text"][s:e] for s, e, label in tags_to_spans(tokenize(rec["text"]), rec["tags"])}
    assert spans["CITY"] == "Bhilai" and spans["LOCALITY"] in {"Supela", "Civic Centre", "Kohka"}  # written "Supela, Bhilai"


def test_named_streets_draw_on_real_place_names(gaz_df):
    gen = Generator(gaz_df, CONFIGS["clean"], seed=5)
    stems = set()
    for i in range(300):
        rec = gen.record("train", i)
        stems |= {rec["text"][s:e].rsplit(" ", 1)[0] for s, e, label in tags_to_spans(tokenize(rec["text"]), rec["tags"])
                  if label == "STREET"}
    office_words = {office.split()[0] for office in gaz_df.office}
    assert stems & (office_words - set(L.STREET_NAMES))  # "Kohka Road": not only the 30 fixed names a model could memorise


def test_same_seed_same_output_different_seed_differs(gaz_df):
    a = [Generator(gaz_df, CONFIGS["clean"], seed=42).record("train", i) for i in range(50)]
    b = [Generator(gaz_df, CONFIGS["clean"], seed=42).record("train", i) for i in range(50)]
    c = [Generator(gaz_df, CONFIGS["clean"], seed=43).record("train", i) for i in range(50)]
    assert a == b and a != c


def test_split_pins_is_disjoint_complete_and_fixed(gaz_df):
    pins = sorted(gaz_df.pin.unique())
    s = split_pins(pins)
    assert set(s) == {"train", "dev", "test"}
    assert sorted(s["train"] + s["dev"] + s["test"]) == pins
    assert all(s["dev"]) and all(s["test"])
    assert split_pins(list(reversed(pins))) == s  # order of input does not matter


def test_generate_writes_splits_with_held_out_pins(gaz_df, tmp_path):
    gaz_path = tmp_path / "gaz.csv"
    gaz_df.to_csv(gaz_path, index=False)
    out = tmp_path / "clean"
    counts = generate("clean", seed=7, n_train=30, n_dev=5, n_test=5, out_dir=out, gaz_path=gaz_path)
    assert counts == {"train": 30, "dev": 5, "test": 5}
    train, test = read_jsonl(out / "train.jsonl"), read_jsonl(out / "test.jsonl")
    assert len(train) == 30 and len(test) == 5
    assert not {r["meta"]["pin"] for r in train} & {r["meta"]["pin"] for r in test}
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["config"] == "clean" and cfg["seed"] == 7 and cfg["counts"] == counts
    assert len(cfg["gazetteer_sha256"]) == 64


def test_unknown_config_rejected(gaz_df):
    with pytest.raises(KeyError):
        generate("nope", seed=0, n_train=1, n_dev=1, n_test=1, out_dir="/nonexistent", gaz_path=FIXTURE)


def test_generator_skips_unusable_office_names():
    df = pd.DataFrame({"pin": ["900056", "490023", "492001", "625001"], "office": ["1CBPO", "Supela Bhilai", "Raipur", "Madurai"],
                       "office_type": ["PO"] * 4, "delivery": ["Delivery"] * 4, "city": ["Kushi Nagar", "Durg", "Raipur", "Madurai"],
                       "district": ["Kushi Nagar", "Durg", "Raipur", "Madurai"],
                       "state": ["Uttar Pradesh", "Chhattisgarh", "Chhattisgarh", "Tamil Nadu"], "lat": [""] * 4, "lon": [""] * 4})
    gen = Generator(df, CONFIGS["clean"], seed=0)
    recs = [gen.record(split, i) for split in ("train", "dev", "test") for i in range(10)]
    assert all(r["meta"]["pin"] != "900056" and "1CBPO" not in r["text"] for r in recs)
