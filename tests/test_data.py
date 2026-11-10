import json

import pytest

from src.data import (
    LABELS, TAGS, label_substrings, make_record, read_jsonl, spans_to_tags,
    tags_to_spans, tokenize, validate, write_jsonl,
)

# Hand-written public-place style addresses (no names / phones). (text, [(substring, label), ...])
ROUND_TRIP = [
    ("flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490023",
     [("flat 3b", "HOUSE"), ("shanti apts", "BUILDING"), ("behind SBI atm", "LANDMARK"),
      ("nr hanuman mandir", "LANDMARK"), ("supela", "LOCALITY"), ("bhilai", "CITY"), ("490023", "PIN")]),
    ("opp. ganesh temple, 2nd cross, hosur road, madivala, b'lore-68",
     [("opp. ganesh temple", "LANDMARK"), ("2nd cross", "STREET"), ("hosur road", "STREET"),
      ("madivala", "LOCALITY"), ("b'lore", "CITY"), ("68", "PIN")]),
    ("Civic Centre Library, Sector 6, Bhilai, Durg, Chhattisgarh 490 006",
     [("Sector 6", "LOCALITY"), ("Bhilai", "CITY"), ("Durg", "DISTRICT"), ("Chhattisgarh", "STATE"), ("490 006", "PIN")]),
    ("हनुमान मंदिर के पास, सुपेला, भिलाई 490023",
     [("हनुमान मंदिर के पास", "LANDMARK"), ("सुपेला", "LOCALITY"), ("भिलाई", "CITY"), ("490023", "PIN")]),
    ("Shop No. 4, Gupta Complex, MG Road, Raipur, C.G. - 492001",
     [("Shop No. 4", "HOUSE"), ("Gupta Complex", "BUILDING"), ("MG Road", "STREET"),
      ("Raipur", "CITY"), ("C.G.", "STATE"), ("492001", "PIN")]),
    ("D.No 5-6-7/A, Himayat Nagar, Hyd",
     [("D.No 5-6-7/A", "HOUSE"), ("Himayat Nagar", "LOCALITY"), ("Hyd", "CITY")]),
    ("#12, 3rd main, Koramangala 4th Block, Bengaluru, Karnataka",
     [("#12", "HOUSE"), ("3rd main", "STREET"), ("Koramangala 4th Block", "LOCALITY"),
      ("Bengaluru", "CITY"), ("Karnataka", "STATE")]),
    ("GOVT HIGH SCHOOL NEAR BUS STAND DISTT DURG CG",
     [("NEAR BUS STAND", "LANDMARK"), ("DURG", "DISTRICT"), ("CG", "STATE")]),
    ("Sai Medical, 2nd floor, Tower C, Sunshine Enclave, Andheri West, Mumbai, MH 400058",
     [("2nd floor", "HOUSE"), ("Tower C", "BUILDING"), ("Sunshine Enclave", "BUILDING"),
      ("Andheri West", "LOCALITY"), ("Mumbai", "CITY"), ("MH", "STATE"), ("400058", "PIN")]),
    ("Post office, Gali No. 4, Gandhi Nagar, Pin: 110031, Delhi",
     [("Gali No. 4", "STREET"), ("Gandhi Nagar", "LOCALITY"), ("110031", "PIN"), ("Delhi", "CITY")]),
]


def test_label_set_and_tags():
    assert LABELS == ("HOUSE", "BUILDING", "STREET", "LANDMARK", "LOCALITY", "CITY", "DISTRICT", "STATE", "PIN")
    assert "O" in TAGS and "B-PIN" in TAGS and "I-LANDMARK" in TAGS and len(TAGS) == 19


@pytest.mark.parametrize("text,expected", [
    ("flat 3b, shanti apts", ["flat", "3b", ",", "shanti", "apts"]),
    ("b'lore-68", ["b", "'", "lore", "-", "68"]),
    ("490 023", ["490", "023"]),
    ("H.No. 12/3", ["H", ".", "No", ".", "12", "/", "3"]),
    ("#12,  3rd\nmain", ["#", "12", ",", "3rd", "main"]),
    ("हनुमान मंदिर के पास।", ["हनुमान", "मंदिर", "के", "पास", "।"]),
    ("", []),
])
def test_tokenize_splits_whitespace_and_punctuation(text, expected):
    toks = tokenize(text)
    assert [t.text for t in toks] == expected
    assert all(text[t.start:t.end] == t.text for t in toks)


@pytest.mark.parametrize("text,pairs", ROUND_TRIP)
def test_spans_round_trip_through_bio(text, pairs):
    # Arrange
    spans = label_substrings(text, pairs)
    tokens = tokenize(text)
    # Act
    tags = spans_to_tags(tokens, spans)
    back = tags_to_spans(tokens, tags)
    # Assert
    assert len(tags) == len(tokens)
    assert back == spans
    assert [text[s:e] for s, e, _ in back] == [p for p, _ in pairs]


def test_back_to_back_landmarks_get_two_b_tags():
    text = "behind SBI atm nr hanuman mandir"
    tags = spans_to_tags(tokenize(text), label_substrings(text, [("behind SBI atm", "LANDMARK"), ("nr hanuman mandir", "LANDMARK")]))
    assert tags == ["B-LANDMARK", "I-LANDMARK", "I-LANDMARK", "B-LANDMARK", "I-LANDMARK", "I-LANDMARK"]


def test_span_must_align_to_token_boundaries():
    tokens = tokenize("supela bhilai")
    with pytest.raises(ValueError):
        spans_to_tags(tokens, [(0, 3, "LOCALITY")])


def test_overlapping_spans_rejected():
    tokens = tokenize("supela bhilai")
    with pytest.raises(ValueError):
        spans_to_tags(tokens, [(0, 6, "LOCALITY"), (0, 13, "CITY")])


def test_label_substrings_handles_repeats_and_unknown():
    spans = label_substrings("Durg, Dist Durg", [("Durg", "CITY"), ("Durg", "DISTRICT")])
    assert spans == [(0, 4, "CITY"), (11, 15, "DISTRICT")]
    with pytest.raises(ValueError):
        label_substrings("abc", [("zzz", "CITY")])


def test_make_record_and_validate():
    text, pairs = ROUND_TRIP[0]
    rec = make_record("hand-0001", text, label_substrings(text, pairs), source="synthetic", split="dev",
                      slices=["landmark"], meta={"pin": "490023"})
    validate(rec)
    assert rec["tokens"][0] == "flat" and rec["tags"][0] == "B-HOUSE"
    assert rec["noise"] == [] and rec["slices"] == ["landmark"]


@pytest.mark.parametrize("patch", [
    {"tags": ["O"]},                                   # length mismatch
    {"tokens": ["x"]},                                 # tokens do not match text
    {"source": "human"},
    {"split": "eval"},
    {"id": 7},
    {"text": ""},
])
def test_validate_rejects_bad_records(patch):
    text, pairs = ROUND_TRIP[0]
    rec = {**make_record("hand-0001", text, label_substrings(text, pairs), "synthetic", "dev"), **patch}
    with pytest.raises(ValueError):
        validate(rec)


def test_validate_rejects_broken_iob2_and_unknown_tags():
    rec = make_record("x", "supela bhilai", [(0, 6, "LOCALITY")], "synthetic", "dev")
    with pytest.raises(ValueError):
        validate({**rec, "tags": ["I-LOCALITY", "O"]})
    with pytest.raises(ValueError):
        validate({**rec, "tags": ["B-LOCALITY", "I-CITY"]})
    with pytest.raises(ValueError):
        validate({**rec, "tags": ["B-TOWN", "O"]})


def test_jsonl_round_trip(tmp_path):
    recs = [make_record(f"hand-{i:04d}", t, label_substrings(t, p), "synthetic", "dev") for i, (t, p) in enumerate(ROUND_TRIP)]
    path = tmp_path / "x.jsonl"
    write_jsonl(path, recs)
    assert read_jsonl(path) == recs
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[3])["text"].startswith("हनुमान")
    with pytest.raises(ValueError):
        write_jsonl(path, [{**recs[0], "split": "nope"}])


def test_jsonl_survives_unicode_line_separators(tmp_path):
    text = "shop 4\u2028near temple\u0085bhilai"
    rec = make_record("x", text, label_substrings(text, [("near temple", "LANDMARK")]), "synthetic", "dev")
    path = tmp_path / "u.jsonl"
    write_jsonl(path, [rec])
    assert read_jsonl(path) == [rec]


def test_tags_to_spans_restarts_after_o_and_dangling_i():
    toks = tokenize("a b c d")
    assert tags_to_spans(toks, ["B-CITY", "I-CITY", "O", "I-CITY"]) == [(0, 3, "CITY"), (6, 7, "CITY")]
    assert tags_to_spans(toks, ["I-PIN", "I-PIN", "B-PIN", "I-CITY"]) == [(0, 3, "PIN"), (4, 5, "PIN"), (6, 7, "CITY")]
