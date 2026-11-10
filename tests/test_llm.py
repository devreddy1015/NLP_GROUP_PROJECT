import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast, GPT2Config, GPT2LMHeadModel

from src.data import read_jsonl
from src.models import llm
from src.models.llm import SYSTEMS, System, fold, generate, gold_fields, messages, parse_json, run, to_fields

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")
ABR = SYSTEMS["abr-parser"]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random one-layer causal LM with a plain chat template, saved to disk, so no test downloads a model."""
    d = tmp_path_factory.mktemp("lm")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "zz", "yy", "{", "}"]
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    tok.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    tok.save_pretrained(d)
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64, n_embd=32, n_layer=1, n_head=2)).save_pretrained(d)
    return str(d)


def test_parse_json_finds_the_object_or_gives_none():
    assert parse_json('Here: {"flat": "3B", "city": "Pune"} done') == {"flat": "3B", "city": "Pune"}
    assert parse_json("no json here") is None and parse_json('{"flat": ') is None and parse_json("[1, 2]") is None
    assert parse_json('{"flat": }') is None


def test_the_parser_mapping_is_the_one_in_research_md():
    """research.md §2. A change here changes E1, so it needs a row in the decision log first (build-plan B14)."""
    assert ABR.mapping == {"flat": "HOUSE", "building": "BUILDING", "street": "STREET", "landmark": "LANDMARK",
                           "area": "LOCALITY", "city": "CITY", "state": "STATE", "pincode": "PIN"}
    assert ABR.either == ("CITY", "DISTRICT") and len(ABR.revision) == 40 and not ABR.shots
    assert ABR.prompt.startswith("You are an Indian address parser.") and ABR.prompt.endswith("Output only JSON.")


def test_to_fields_maps_keys_drops_blanks_and_writes_the_state_in_full():
    obj = {"flat": "14/2", "building": "", "street": "Gandhi Road", "landmark": None, "area": "Pollachi",
           "city": "Coimbatore", "state": "TN", "pincode": 642001, "phone": "000"}
    assert to_fields(obj, ABR.mapping) == {"HOUSE": {"142"}, "STREET": {"gandhi road"}, "LOCALITY": {"pollachi"},
                                           "CITY": {"coimbatore"}, "STATE": {"tamil nadu"}, "PIN": {"642001"}}
    assert to_fields(None, ABR.mapping) == {}


def test_gold_fields_write_the_state_in_full_too():
    rec = next(r for r in read_jsonl(HAND20) if "(C.G.)" in r["text"])
    assert gold_fields(rec)["STATE"] == {"chhattisgarh"} and gold_fields(rec)["PIN"] == {"492001"}


def test_fold_counts_their_city_as_right_if_it_is_our_city_or_our_district():
    gold = {"CITY": {"bhilai"}, "DISTRICT": {"durg"}, "PIN": {"490020"}}
    assert fold(gold, {"CITY": {"durg"}}, ("CITY", "DISTRICT")) == {"CITY": {"durg"}, "PIN": {"490020"}}
    missed = fold(gold, {"CITY": {"raipur"}}, ("CITY", "DISTRICT"))
    assert len(missed["CITY"]) == 1 and "DISTRICT" not in missed  # one miss, not two
    assert fold({"PIN": {"490020"}}, {"CITY": {"durg"}}, ("CITY", "DISTRICT")) == {"PIN": {"490020"}}
    assert fold(gold, {}, ()) == gold


def test_messages_put_the_examples_between_the_system_prompt_and_the_address():
    got = messages("Parse it.", "Shop 2, Durg", [("Flat 1, Pune", '{"flat": "1"}')])
    assert [m["role"] for m in got] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in got] == ["Parse it.", "Flat 1, Pune", '{"flat": "1"}', "Shop 2, Durg"]


def test_generate_returns_only_the_new_text_one_answer_per_address(tiny):
    tok, net = AutoTokenizer.from_pretrained(tiny), AutoModelForCausalLM.from_pretrained(tiny)
    out = generate(net, tok, "zz zz zz zz zz zz", ["yy", "yy yy yy"], batch_size=2, max_new_tokens=3)
    assert len(out) == 2 and all(len(tok.tokenize(o)) <= 3 for o in out)  # the six-word prompt is not echoed back


def test_run_scores_fields_and_keeps_the_raw_answers(tiny, tmp_path, monkeypatch):
    recs = read_jsonl(HAND20)
    pins = [next(iter(gold_fields(r).get("PIN", [])), None) for r in recs]
    answers = [json.dumps({"city": "Nowhere", **({"pincode": pin} if pin else {})}) for pin in pins[:-1]] + ["not json"]
    monkeypatch.setitem(llm.SYSTEMS, "fake", System(tiny, None, "Parse it.", {"pincode": "PIN", "city": "CITY"}))
    monkeypatch.setattr(llm, "generate", lambda *a, **k: answers)
    res = run("fake", HAND20, "20261111-fake-hand20-s0", results_dir=tmp_path, device="cpu")
    saved = json.loads((tmp_path / "20261111-fake-hand20-s0.json").read_text())
    assert saved["model"] == "fake" and saved["n"] == 20 and saved["json_valid"] == 0.95 and "span" not in saved
    assert saved["config"]["shared_labels"] == ["CITY", "PIN"] and saved["config"]["n_shots"] == 0
    assert saved["config"]["transformers"] and saved["config"]["torch"]
    assert saved["field"]["per_field"]["PIN"]["p"] == 1.0 and saved["field"]["per_field"]["CITY"]["f1"] == 0
    assert res["field"] == saved["field"]
    raw = [json.loads(line) for line in (tmp_path / "20261111-fake-hand20-s0.pred.jsonl").read_text().splitlines()]
    assert [x["id"] for x in raw] == [r["id"] for r in recs] and raw[-1]["output"] == "not json"
