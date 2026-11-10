from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

from app.demo import EXAMPLE, crf_tagger, parse, rules_tagger, transformer_tagger
from src.data import TAGS, tokenize, write_jsonl
from src.generate import CONFIGS, Generator
from src.normalize.gazetteer import Gazetteer, build

FIXTURE = Path("tests/fixtures/pincode_sample.csv")


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    out = tmp_path_factory.mktemp("gaz") / "p.csv"
    build(raw=FIXTURE, out=out)
    return Gazetteer.load(out)


def test_parse_shows_every_character_once_and_the_cleaned_fields(gaz):
    out = parse(EXAMPLE, rules_tagger(gaz), gaz)
    assert "".join(piece for piece, _ in out["spans"]) == EXAMPLE
    assert ("490023", "PIN") in out["spans"] and ("behind SBI atm", "LANDMARK") in out["spans"]
    assert out["json"]["pin"] == "490023" and out["json"]["pin_check"] == "valid"
    assert out["json"]["landmarks"][0] == {"relation": "behind", "anchor": "SBI atm"}
    assert any("filled in from the PIN" in note and "Durg" in note for note in out["notes"])


def test_parse_says_when_the_pin_is_unknown_or_missing(gaz):
    unknown = parse("Shop 4, MG Road, Pune 999999", rules_tagger(gaz), gaz)
    assert unknown["json"]["pin_check"] == "not_found" and any("not in the India Post directory" in n for n in unknown["notes"])
    missing = parse("Shop 4, MG Road, Pune", rules_tagger(gaz), gaz)
    assert missing["json"]["pin_check"] == "missing" and any("No PIN" in n for n in missing["notes"])


def test_parse_of_nothing_asks_for_an_address(gaz):
    assert parse("   ", rules_tagger(gaz), gaz) == {"spans": [], "json": {}, "notes": ["Type an address first."]}


def test_the_crf_tagger_trains_and_gives_one_tag_per_token(gaz, tmp_path):
    gen = Generator(gaz.df, CONFIGS["full"], seed=3)
    write_jsonl(tmp_path / "train.jsonl", [gen.record("train", i) for i in range(200)])
    tags = crf_tagger(gaz, tmp_path / "train.jsonl", max_iter=20)(EXAMPLE)
    assert len(tags) == len(tokenize(EXAMPLE)) and "B-PIN" in tags


def test_a_saved_transformer_gives_one_tag_per_token(tmp_path):
    """A random tiny tagger with our tag list, saved the way `src.models.transformer --save-dir` saves one."""
    vocab = dict.fromkeys(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", *sorted({t.text.lower() for t in tokenize(EXAMPLE)})])
    BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}).save_pretrained(tmp_path)
    torch.manual_seed(0)
    BertForTokenClassification(BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        id2label=dict(enumerate(TAGS)), label2id={t: i for i, t in enumerate(TAGS)})).save_pretrained(tmp_path)
    tags = transformer_tagger(tmp_path, device="cpu")(EXAMPLE)
    assert len(tags) == len(tokenize(EXAMPLE)) and set(tags) <= set(TAGS)
