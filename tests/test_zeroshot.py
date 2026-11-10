import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from src.data import LABELS, read_jsonl
from src.models import zeroshot
from src.models.zeroshot import SYSTEMS, System, only, run, tag, to_ours, trim

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")
THEIRS = ["O", "B-city", "I-city", "B-pincode", "I-pincode", "B-country", "I-country"]


class PositionNet(torch.nn.Module):
    """Labels each subword "L<position + row in the batch>", so a test can see which subword and which row tag() reads."""
    device = torch.device("cpu")
    config = SimpleNamespace(id2label={i: f"L{i}" for i in range(20)})

    def forward(self, input_ids, **_):
        rows, length = input_ids.shape
        ids = (torch.arange(length).expand(rows, length) + torch.arange(rows).unsqueeze(1)) % 20
        return SimpleNamespace(logits=torch.nn.functional.one_hot(ids, 20).float())


@pytest.fixture(scope="module")
def tokenizer():
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "zz", "##qx", ","]
    return BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A random one-layer tagger with someone else's label names, saved to disk and registered as the system "fake"."""
    words = sorted({t.lower() for r in read_jsonl(HAND20) for t in r["tokens"]})
    vocab = dict.fromkeys(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words])
    BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}).save_pretrained(tmp_path / "m")
    torch.manual_seed(0)
    BertForTokenClassification(BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        id2label=dict(enumerate(THEIRS)), label2id={t: i for i, t in enumerate(THEIRS)})).save_pretrained(tmp_path / "m")
    spec = System(str(tmp_path / "m"), None, 64, {"city": "CITY", "pincode": "PIN"})
    monkeypatch.setitem(zeroshot.SYSTEMS, "fake", spec)
    return spec


def test_to_ours_maps_labels_and_starts_every_span_with_b():
    mapping = SYSTEMS["shiprocket-tinybert"].mapping
    theirs = ["B-house_details", "I-house_details", "B-floor", "O", "I-city", "I-city", "I-state", "B-country"]
    assert to_ours(theirs, mapping) == ["B-HOUSE", "I-HOUSE", "B-HOUSE", "O", "B-CITY", "I-CITY", "B-STATE", "O"]
    assert to_ours(["B-locality", "I-sub_locality"], mapping) == ["B-LOCALITY", "I-LOCALITY"]  # two labels of theirs, one of ours
    assert to_ours(["B-city", "O", "I-city"], mapping) == ["B-CITY", "O", "B-CITY"]  # an O ends the span


def test_only_blanks_the_labels_a_system_does_not_have():
    assert only(["B-HOUSE", "I-HOUSE", "O", "B-CITY", "B-DISTRICT"], {"CITY", "PIN"}) == ["O", "O", "O", "B-CITY", "O"]


def test_trim_drops_stray_commas_and_brackets_from_span_ends():
    text = "Shree Sweets, Bhilai, (C.G.)"  # tokens: Shree Sweets , Bhilai , ( C . G . )
    tags = ["B-BUILDING", "I-BUILDING", "I-BUILDING", "B-CITY", "B-LANDMARK",
            "B-STATE", "I-STATE", "I-STATE", "I-STATE", "I-STATE", "I-STATE"]
    assert trim(text, tags) == ["B-BUILDING", "I-BUILDING", "O", "B-CITY", "O",  # a span that is only a comma goes
                                "O", "B-STATE", "I-STATE", "I-STATE", "I-STATE", "O"]  # dots stay: "C.G."


def test_tag_reads_the_first_subword_that_overlaps_each_token(tokenizer):
    # "Zzqx, zz" -> [CLS] zz ##qx , zz [SEP]: our tokens are Zzqx (two subwords), the comma and zz
    records = [{"text": "Zzqx, zz"}, {"text": "zz \N{ZERO WIDTH SPACE} zz"}, {"text": "zz zz"}]
    first, dropped, next_batch = tag(PositionNet(), tokenizer, records, max_len=64, batch_size=2)
    assert first == ["L1", "L3", "L4"]
    assert dropped == ["L2", "O", "L3"]  # second row of its batch (+1); the tokenizer drops the zero-width space
    assert next_batch == ["L1", "L2"]  # first row of the second batch
    cut = tag(PositionNet(), tokenizer, [{"text": "zz " * 30}], max_len=8)[0]  # 8 = [CLS] + 6 subwords + [SEP]
    assert cut == [f"L{i}" for i in range(1, 7)] + ["O"] * 24


def test_mappings_are_the_ones_in_research_md():
    """research.md §2. A change here changes E1, so it needs a row in the decision log first (build-plan B14)."""
    shiprocket = {"house_details": "HOUSE", "floor": "HOUSE", "building_name": "BUILDING", "road": "STREET",
                  "landmarks": "LANDMARK", "locality": "LOCALITY", "sub_locality": "LOCALITY", "city": "CITY",
                  "state": "STATE", "pincode": "PIN"}
    assert SYSTEMS["shiprocket-indicbert"].mapping == SYSTEMS["shiprocket-tinybert"].mapping == shiprocket
    assert SYSTEMS["mukuls-muril"].mapping == {f"ADDRESS_{label}": label for label in LABELS}
    assert SYSTEMS["gagan-tinybert"].mapping == {
        "houseNumber": "HOUSE", "houseName": "HOUSE", "poi": "LANDMARK", "street": "STREET", "subsubLocality": "LOCALITY",
        "subLocality": "LOCALITY", "locality": "LOCALITY", "village": "LOCALITY", "district": "DISTRICT",
        "subDistrict": "DISTRICT", "city": "CITY", "state": "STATE", "pincode": "PIN"}
    assert {name: spec.max_len for name, spec in SYSTEMS.items()} == {
        "shiprocket-indicbert": 128, "shiprocket-tinybert": 128, "mukuls-muril": 192, "gagan-tinybert": 160}
    assert all(len(spec.revision) == 40 for spec in SYSTEMS.values())  # a full commit hash, not a branch


def test_run_maps_trims_and_scores_what_the_model_tagged(fake, tmp_path, monkeypatch):
    """The model's answer is canned: gold CITY and PIN under their names, each span dragging the next comma along."""
    names, seen, loads = {"CITY": "city", "PIN": "pincode"}, [], []

    def canned(net, tok, records, max_len):
        seen.append(max_len)
        out = []
        for r in records:
            theirs = [t[:2] + names[t[2:]] if t[2:] in names else "O" for t in r["tags"]]
            for i in range(1, len(theirs)):
                if r["tokens"][i] == "," and theirs[i - 1] != "O" and theirs[i] == "O":
                    theirs[i] = "I-" + theirs[i - 1][2:]
            out.append(theirs)
        return out

    real = zeroshot.AutoModelForTokenClassification.from_pretrained
    monkeypatch.setattr(zeroshot.AutoModelForTokenClassification, "from_pretrained",
                        lambda *a, **k: loads.append(k) or real(*a, **k))
    monkeypatch.setattr(zeroshot, "tag", canned)
    res = run("fake", HAND20, "20261111-fake-hand20-s0", results_dir=tmp_path, bootstrap_n=10, device="cpu")
    assert seen == [64] and loads == [{"revision": None, "use_safetensors": True}]
    assert res["span"]["micro"]["f1"] == 1.0 and res["field"]["micro"]["f1"] == 1.0  # mapped, and the commas trimmed
    recs = read_jsonl(HAND20)
    kept = [json.loads(line) for line in (tmp_path / "20261111-fake-hand20-s0.pred.jsonl").read_text().splitlines()]
    assert [k["id"] for k in kept] == [r["id"] for r in recs] and any("I-city" in k["theirs"] for k in kept)
    assert all(len(k["tags"]) == len(k["theirs"]) == len(r["tags"]) for k, r in zip(kept, recs))


def test_run_scores_a_system_on_the_labels_it_shares_with_us(fake, tmp_path):
    res = run("fake", HAND20, "20261111-fake-hand20-s0", results_dir=tmp_path, bootstrap_n=10, device="cpu")
    saved = json.loads((tmp_path / "20261111-fake-hand20-s0.json").read_text())
    assert saved["model"] == "fake" and saved["n"] == 20 and saved["data"] == "fixtures/handlabelled20"
    assert saved["config"]["shared_labels"] == ["CITY", "PIN"] and saved["config"]["max_len"] == 64
    assert saved["config"]["transformers"] and saved["config"]["torch"]
    assert set(saved["span"]["per_label"]) <= {"CITY", "PIN"}
    assert saved["label_support"]["CITY"] > 0 and saved["label_support"]["HOUSE"] == 0
    assert res["span"] == saved["span"]


def test_run_rejects_a_mapping_the_model_cannot_produce(fake, tmp_path, monkeypatch):
    monkeypatch.setitem(zeroshot.SYSTEMS, "typo", System(fake.repo, None, 64, {"citty": "CITY"}))
    with pytest.raises(ValueError, match="citty"):
        run("typo", HAND20, "20261111-typo-hand20-s0", results_dir=tmp_path, device="cpu")
