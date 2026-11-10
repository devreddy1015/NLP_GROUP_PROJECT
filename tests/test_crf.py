import json
from pathlib import Path

import pytest

from src.data import read_jsonl, validate, write_jsonl
from src.generate import CONFIGS, Generator
from src.models.crf import Lexicon, predict, run, token_features, train
from src.normalize.gazetteer import Gazetteer, build

FIXTURE = Path("tests/fixtures/pincode_sample.csv")
HAND20 = Path("tests/fixtures/handlabelled20.jsonl")


@pytest.fixture(scope="module")
def gaz(tmp_path_factory):
    out = tmp_path_factory.mktemp("gaz") / "p.csv"
    build(raw=FIXTURE, out=out)
    return Gazetteer.load(out)


@pytest.fixture(scope="module")
def synthetic(gaz, tmp_path_factory):
    gen = Generator(gaz.df, CONFIGS["full"], seed=3)
    d = tmp_path_factory.mktemp("syn")
    write_jsonl(d / "train.jsonl", [gen.record("train", i) for i in range(300)])
    write_jsonl(d / "dev.jsonl", [gen.record("dev", i) for i in range(60)])
    return d


def test_token_features_cues_and_gazetteer(gaz):
    lex = Lexicon(gaz)
    toks = ["near", "Hanuman", "Mandir", ",", "Durg", "490023"]
    f0, f4, f5 = token_features(toks, 0, lex), token_features(toks, 4, lex), token_features(toks, 5, lex)
    assert f0["cue_rel"] and f0["BOS"] and f0["+1:w"] == "hanuman"
    assert f4["gaz_district"] and f4["gaz_city"] and f4["shape"] == "Xx" and f4["-1:w"] == ","
    assert f5["pin_like"] and f5["EOS"] and f5["shape"] == "d"
    assert token_features(["MG", "Road"], 1, lex)["cue_street"]
    assert token_features(["Chhattisgarh"], 0, lex)["state_alias"]


def test_crf_learns_synthetic_addresses(gaz, synthetic):
    lex = Lexicon(gaz)
    tr, dev = read_jsonl(synthetic / "train.jsonl"), read_jsonl(synthetic / "dev.jsonl")
    crf = train(tr, lex, max_iter=60)
    pred = predict(crf, dev, lex)
    assert len(pred) == len(dev) and all(len(p) == len(r["tags"]) for p, r in zip(pred, dev))
    from src.eval import span_scores
    assert span_scores([r["tags"] for r in dev], pred)["micro"]["f1"] > 0.85


def test_run_writes_results_and_subsamples(gaz, synthetic, tmp_path):
    res = run(synthetic / "train.jsonl", HAND20, run_id="20261110-crf-test-s1", seed=1, n_train=100,
              results_dir=tmp_path, gaz=gaz, max_iter=30, bootstrap_n=10)
    saved = json.loads((tmp_path / "20261110-crf-test-s1.json").read_text())
    assert saved["model"] == "crf" and saved["n"] == 20 and saved["config"]["n_train"] == 100 and saved["seed"] == 1
    assert len((tmp_path / "20261110-crf-test-s1.pred.jsonl").read_text().splitlines()) == 20  # tags kept per address
    big = run(synthetic / "train.jsonl", HAND20, run_id="20261110-crf-test-s2", n_train=10**6,
              results_dir=tmp_path, gaz=gaz, max_iter=5, bootstrap_n=10)
    assert big["config"]["n_train"] == big["config"]["n_train_total"]  # asking for more than there is uses the whole set
    assert res["config"]["train"].endswith("train.jsonl") and 0 <= saved["span"]["micro"]["f1"] <= 1
    with pytest.raises(ValueError, match="train split"):  # the fixture rows are marked dev
        run(HAND20, synthetic / "dev.jsonl", run_id="20261110-crf-test-s3", results_dir=tmp_path, gaz=gaz)
