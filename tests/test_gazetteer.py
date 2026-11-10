from pathlib import Path

import pandas as pd
import pytest

from src.normalize.gazetteer import Gazetteer, build, canonical_state, clean_city, clean_office_name, STATES

FIXTURE = Path("tests/fixtures/pincode_sample.csv")


@pytest.mark.parametrize("raw,clean", [
    ("Supela Bhilai SO", "Supela Bhilai"),
    ("Kohka Bhilai SO", "Kohka Bhilai"),
    ("Bhilai B.O", "Bhilai"),
    ("Peddakotla B.O.", "Peddakotla"),
    ("Abhiya bazarBO", "Abhiya bazar"),
    ("Jogia  B.O (Alauli S.O)", "Jogia"),
    ("Mumbai GPO", "Mumbai"),
    ("Kanpur H.O", "Kanpur"),
    ("SWAI SINGH WALA B.O", "Swai Singh Wala"),
    ("Uravakonda (Bus Stand) B.O", "Uravakonda"),
    ("ST. MARY'S COLONY B.O", "St. Mary's Colony"),
    ("1ST BLOCK BO", "1st Block"),
])
def test_clean_office_name(raw, clean):
    assert clean_office_name(raw) == clean


@pytest.mark.parametrize("division,district,city", [
    ("Durg Division", "DURG", "Durg"),
    ("Kanpur HO Division", "KANPUR NAGAR", "Kanpur"),
    ("BANGALORE GPO DIVISION", "BENGALURU", "Bangalore"),
    ("Divisional Office Katihar", "KATIHAR", "Katihar"),
    ("NA", "WEST GODAVARI", "West Godavari"),
])
def test_clean_city(division, district, city):
    assert clean_city(division, district) == city


@pytest.mark.parametrize("raw,canon", [
    ("CHHATTISGARH", "Chhattisgarh"), ("Chattisgarh", "Chhattisgarh"), ("C.G.", "Chhattisgarh"), ("cg", "Chhattisgarh"),
    ("MH", "Maharashtra"), ("Tamilnadu", "Tamil Nadu"), ("Orissa", "Odisha"), ("J&K", "Jammu and Kashmir"),
    ("THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU", "Dadra and Nagar Haveli and Daman and Diu"),
    ("New Delhi", "Delhi"), ("Pondicherry", "Puducherry"), ("Bengaluru", None), ("", None),
])
def test_canonical_state(raw, canon):
    assert canonical_state(raw) == canon


def test_state_table_has_36_rows_and_unique_aliases():
    assert len(STATES) == 36
    aliases = [a.lower() for v in STATES.values() for a in v]
    assert len(aliases) == len(set(aliases))


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    out = tmp_path_factory.mktemp("gaz") / "pincodes.csv"
    build(raw=FIXTURE, out=out)
    return Gazetteer.load(out)


def test_build_writes_clean_columns(tmp_path):
    df = build(raw=FIXTURE, out=tmp_path / "p.csv")
    assert list(df.columns) == ["pin", "office", "office_type", "delivery", "city", "district", "state", "lat", "lon"]
    assert (tmp_path / "p.csv").exists()
    assert set(df.state) <= set(STATES)
    assert df[df.pin == "490023"].office.tolist() == ["Kohka Bhilai", "Vaishali Nagar Bhilai", "Supela Bhilai"]


def test_lookup_490023_is_supela_durg_chhattisgarh(gaz):
    hit = gaz.lookup("490023")
    assert "Supela Bhilai" in hit["offices"]
    assert hit == {"pin": "490023", "offices": ["Kohka Bhilai", "Vaishali Nagar Bhilai", "Supela Bhilai"],
                   "city": "Durg", "district": "Durg", "districts": ["Durg"], "state": "Chhattisgarh"}


def test_lookup_unknown_pin_is_none(gaz):
    assert gaz.lookup("000000") is None
    assert gaz.lookup("49002") is None


def test_name_sets_are_normalised(gaz):
    assert "chhattisgarh" in gaz.states and "durg" in gaz.districts and "durg" in gaz.cities
    assert "supela bhilai" in gaz.offices and "bhilai" in gaz.offices
    g = Gazetteer(pd.DataFrame({"pin": ["494001"], "office": ["Bade Kilepal-2"], "office_type": ["BO"], "delivery": ["Delivery"],
                                "city": ["Jagdalpur"], "district": ["Bastar"], "state": ["Chhattisgarh"], "lat": [""], "lon": [""]}))
    assert "bade kilepal2" in g.offices  # same key rules.py builds from the text


@pytest.mark.skipif(not Path("data/gazetteer/pincodes.csv").exists(), reason="full gazetteer not built")
def test_full_gazetteer_lookup():
    hit = Gazetteer.load().lookup("490023")
    assert hit["district"] == "Durg" and hit["state"] == "Chhattisgarh" and "Supela Bhilai" in hit["offices"]
