import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig, BertModel, BertTokenizerFast

from src.data import TAGS, read_jsonl, write_jsonl
from src.eval import span_scores
from src.models import transformer
from src.models.transformer import encode, predict, run, train

HAND20 = Path("tests/fixtures/handlabelled20.jsonl")
# "Zzqx" is two subwords (zz ##qx); the tokenizer drops the zero-width space altogether
ODD = {"tokens": ["Zzqx", ",", "\N{ZERO WIDTH SPACE}", "zz"], "tags": ["B-LOCALITY", "O", "O", "B-CITY"]}
LONG = {"tokens": ["zz"] * 30, "tags": ["O"] * 30}


class PositionNet(torch.nn.Module):
    """Predicts tag number = subword position, so a test can see which subword predict() reads."""
    device = torch.device("cpu")

    def forward(self, input_ids, **_):
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        return SimpleNamespace(logits=torch.nn.functional.one_hot(positions, len(TAGS)).float())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A random two-layer BERT saved to disk, so no test downloads a model. Its vocab holds every hand-labelled token whole."""
    d = tmp_path_factory.mktemp("tiny")
    words = sorted({t.lower() for r in read_jsonl(HAND20) for t in r["tokens"]})
    vocab = dict.fromkeys(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "zz", "##qx", ",", *words])
    BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}).save_pretrained(d)
    torch.manual_seed(0)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=128)).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def train_file(tmp_path_factory):
    """The hand-labelled addresses under other ids: run() refuses to train on an address that is also in an eval file."""
    path = tmp_path_factory.mktemp("train") / "train.jsonl"
    write_jsonl(path, [{**r, "id": "t-" + r["id"], "split": "train"} for r in read_jsonl(HAND20)])
    return path


@pytest.fixture(scope="module")
def trained(tiny):
    return train(read_jsonl(HAND20), tiny, seed=0, epochs=40, lr=1e-3, batch_size=8, device="cpu")


@pytest.fixture
def no_training(monkeypatch):
    monkeypatch.setattr(transformer, "train", lambda *a, **k: pytest.fail("trained before checking its arguments"))


def test_encode_labels_only_the_first_subword_of_each_token(tiny):
    tok = AutoTokenizer.from_pretrained(tiny)
    row = encode([ODD], tok)[0]
    assert tok.convert_ids_to_tokens(row["input_ids"]) == ["[CLS]", "zz", "##qx", ",", "zz", "[SEP]"]
    assert row["labels"] == [-100, TAGS.index("B-LOCALITY"), -100, TAGS.index("O"), TAGS.index("B-CITY"), -100]
    assert len(encode([LONG], tok, max_len=8)[0]["labels"]) == 8


def test_predict_reads_the_first_subword_of_each_token(tiny):
    tok = AutoTokenizer.from_pretrained(tiny)
    # [CLS] zz ##qx , zz [SEP]: "Zzqx" is read at position 1, not 2; the dropped token stays O
    assert predict(PositionNet(), tok, [ODD, {"tokens": ["zz"] * 10}])[0] == [TAGS[1], TAGS[3], "O", TAGS[4]]
    cut = predict(PositionNet(), tok, [LONG], max_len=8)[0]  # 8 = [CLS] + 6 subwords + [SEP]
    assert cut == list(TAGS[1:7]) + ["O"] * 24


def test_training_overfits_twenty_addresses(trained):
    model, tok, losses = trained
    recs = read_jsonl(HAND20)
    model.train()
    pred = predict(model, tok, recs)
    assert not model.training  # predict switches dropout off itself
    assert len(losses) == 40 and losses[-1] < losses[0] / 5
    assert span_scores([r["tags"] for r in recs], pred)["micro"]["f1"] > 0.9
    short, long = min(recs, key=lambda r: len(r["tokens"])), max(recs, key=lambda r: len(r["tokens"]))
    assert predict(model, tok, [short]) == predict(model, tok, [short, long])[:1]  # padding changes nothing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed precision only runs on a GPU")
def test_training_on_the_gpu_with_mixed_precision_still_learns(tiny):
    recs = read_jsonl(HAND20)
    model, tok, losses = train(recs, tiny, seed=0, epochs=40, lr=1e-3, batch_size=8, device="cuda")
    assert transformer.AMP and losses[-1] < losses[0] / 5
    assert span_scores([r["tags"] for r in recs], predict(model, tok, recs))["micro"]["f1"] > 0.9


def test_run_trains_once_scores_every_eval_file_and_repeats_with_the_same_seed(tiny, train_file, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transformer, "train", lambda *a, **k: calls.append(a) or train(*a, **k))
    ids = ["20261111-tiny-hand20a-s1", "20261111-tiny-hand20b-s1"]
    kw = dict(model=tiny, seed=1, epochs=15, lr=1e-3, batch_size=8, bootstrap_n=10, device="cpu")
    a, b = run(train_file, [HAND20, HAND20], ids, results_dir=tmp_path / "x", save_dir=tmp_path / "model", **kw)
    assert len(calls) == 1
    saved = json.loads((tmp_path / "x" / f"{ids[0]}.json").read_text())
    assert saved["n"] == 20 and saved["seed"] == 1 and saved["data"] == "fixtures/handlabelled20"
    assert saved["config"]["n_train"] == 20 and saved["config"]["epochs"] == 15 and len(saved["config"]["epoch_loss"]) == 15
    assert saved["config"]["amp"] is False  # mixed precision is a GPU setting; a CPU run must not claim it
    assert a["span"] == b["span"] and a["span"]["micro"]["f1"] > 0
    recs = read_jsonl(HAND20)
    preds = [json.loads(line) for line in (tmp_path / "x" / f"{ids[1]}.pred.jsonl").read_text().splitlines()]
    assert [p["id"] for p in preds] == [r["id"] for r in recs]
    assert all(len(p["tags"]) == len(r["tags"]) for p, r in zip(preds, recs))
    saved_net = AutoModelForTokenClassification.from_pretrained(tmp_path / "model")
    assert predict(saved_net, AutoTokenizer.from_pretrained(tmp_path / "model"), recs) == [p["tags"] for p in preds]  # the model that was scored
    run(train_file, [HAND20], ids[:1], results_dir=tmp_path / "y", **kw)
    assert (tmp_path / "y" / f"{ids[0]}.pred.jsonl").read_text() == (tmp_path / "x" / f"{ids[0]}.pred.jsonl").read_text()


@pytest.mark.parametrize("n_eval, ids", [(1, ["a-s42", "b-s42"]), (1, ["a-s7"]), (2, ["a-s42", "a-s42"])])
def test_run_wants_one_distinct_run_id_per_eval_file_ending_in_the_seed(n_eval, ids, tiny, tmp_path, no_training):
    with pytest.raises(ValueError, match="run id"):
        run(HAND20, [HAND20] * n_eval, ids, model=tiny, seed=42, results_dir=tmp_path)


def test_run_reads_every_eval_file_before_training(tiny, tmp_path, no_training):
    (tmp_path / "empty.jsonl").write_text("")
    with pytest.raises(FileNotFoundError):
        run(HAND20, [HAND20, tmp_path / "missing.jsonl"], ["a-s42", "b-s42"], model=tiny, results_dir=tmp_path)
    with pytest.raises(ValueError, match="empty"):
        run(HAND20, [tmp_path / "empty.jsonl"], ["a-s42"], model=tiny, results_dir=tmp_path)


def test_run_refuses_to_train_on_an_address_it_is_scored_on(tiny, train_file, tmp_path, no_training):
    with pytest.raises(ValueError, match="also in an eval file"):  # the overlap is in the second eval file
        run(train_file, [HAND20, train_file], ["a-s42", "b-s42"], model=tiny, results_dir=tmp_path)
    with pytest.raises(ValueError, match="train split"):  # the fixture rows are marked dev
        run(HAND20, [train_file], ["a-s42"], model=tiny, results_dir=tmp_path)
