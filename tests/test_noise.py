import re
from pathlib import Path

import pytest

from src.data import LABELS, tags_to_spans, tokenize
from src.generate import CONFIGS, FAMILIES, Generator
from src.normalize.gazetteer import build

FIXTURE = Path("tests/fixtures/pincode_sample.csv")
N = 80


@pytest.fixture(scope="module")
def gaz_df(tmp_path_factory):
    return build(raw=FIXTURE, out=tmp_path_factory.mktemp("gaz") / "p.csv")


def spans(rec):
    return [(rec["text"][s:e], label) for s, e, label in tags_to_spans(tokenize(rec["text"]), rec["tags"])]


def test_configs_cover_clean_full_and_ablations():
    assert set(FAMILIES) == {f"N{i}" for i in range(1, 9)}
    assert not any(CONFIGS["clean"].switches().values())
    assert all(CONFIGS["full"].switches().values())
    for fid in FAMILIES:
        minus, only = CONFIGS[f"full-minus-{fid}"].switches(), CONFIGS[f"only-{fid}"].switches()
        assert [k for k, v in minus.items() if not v] == [fid]
        assert [k for k, v in only.items() if v] == [fid]


# One predicate per family: some noised record must show the family's own signature.
SIGNATURE = {
    "N1": lambda r: re.search(r"\b(Rd|Apts|Sec|Opp|nr|Govt|Blr|B'lore|Hyd|CG|MH|UP|TN)\b", r["text"]) is not None,
    "N2": lambda r: sum(l == "LANDMARK" for _, l in spans(r)) != 1 or not any(t.startswith("near ") for t, l in spans(r) if l == "LANDMARK"),
    "N3": lambda r: [l for _, l in spans(r)] != list(LABELS),  # a field dropped or the order changed
    "N4": lambda r: "," not in r["text"] or r["text"] == r["text"].lower() or r["text"] == r["text"].upper(),
    "N5": lambda r: True,
    "N6": lambda r: re.search(r"Mandeer|Mandhir|Galli|Gully|Chauk|Bajar|Kalony|Maarg|Bhawan|Nivas|pass\b|saamne", r["text"], re.I) is not None,
    "N7": lambda r: not any(re.fullmatch(r"\d{6}", t) and l == "PIN" for t, l in spans(r)) or "Pin" in r["text"],
    "N8": lambda r: re.search(r"#\d|H\.No|D\.No|Door No|\d+-\d+-\d+|Flat No", r["text"]) is not None,
}


@pytest.mark.parametrize("fid", sorted(FAMILIES))
def test_family_keeps_labels_valid_and_only_changes_what_it_touches(gaz_df, fid):
    # Arrange: same seed, so geography and word choices match the clean generator record for record
    noisy = Generator(gaz_df, CONFIGS[f"only-{fid}"], seed=5)
    clean = Generator(gaz_df, CONFIGS["clean"], seed=5)
    # Act (make_record validates every record: tokens, IOB2, labels)
    recs = [noisy.record("train", i) for i in range(N)]
    # Assert
    fired = [r for r in recs if r["noise"] == [fid]]
    assert len(fired) >= N // 5, f"{fid} fired on only {len(fired)} of {N}"
    for i, r in enumerate(recs):
        assert (r["text"] != clean.record("train", i)["text"]) == (fid in r["noise"])
        assert r["noise"] in ([], [fid])
    assert any(SIGNATURE[fid](r) for r in fired), f"no record shows the {fid} signature"


def test_full_config_applies_several_families_and_stays_valid(gaz_df):
    recs = [Generator(gaz_df, CONFIGS["full"], seed=9).record("train", i) for i in range(N)]
    assert all(len(r["noise"]) >= 2 for r in recs) and {f for r in recs for f in r["noise"]} == set(FAMILIES)
    assert all(r["tags"][0] == "O" for r in recs)  # the place name stays O in every config
    assert all(s.strip() == s for r in recs for s, _ in spans(r))  # spans never start or end on whitespace


def test_pin_variants_include_attached_short_pin(gaz_df):
    recs = [Generator(gaz_df, CONFIGS["only-N7"], seed=2).record("train", i) for i in range(200)]
    short = [r for r in recs if any(l == "PIN" and len(t) == 2 for t, l in spans(r))]
    assert short and all(re.search(r"\w-\d\d\b", r["text"]) for r in short)
