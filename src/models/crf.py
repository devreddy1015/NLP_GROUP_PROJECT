"""B1: CRF baseline (sklearn-crfsuite) with shape, affix, cue-word and gazetteer features.

Run: uv run python -m src.models.crf --train data/synthetic/full/train.jsonl --eval data/synthetic/full/dev.jsonl \
         --run-id YYYYMMDD-crf-synfull-s0 [--n-train 5000] [--max-iter 100]
"""
import argparse
import random
import re
from pathlib import Path

import sklearn_crfsuite

from src.data import read_jsonl, require_train
from src.eval import evaluate, write_predictions, write_results
from src.models.rules import BUILDING_CUES, DISTRICT_CUES, HOUSE_CUES, LOCALITY_CUES, POSTPOSITIONS, RELATION, STREET_CUES
from src.normalize.gazetteer import Gazetteer, canonical_state


class Lexicon:
    """Word sets from the gazetteer for membership features."""

    def __init__(self, gaz: Gazetteer):
        self.office, self.city = _words(gaz.offices), _words(gaz.cities)
        self.district, self.state = _words(gaz.districts), _words(gaz.states)


def _words(names) -> set[str]:
    return {w for n in names for w in n.split()}


def shape(w: str) -> str:
    s = re.sub(r"\d", "d", re.sub(r"[A-Z]", "X", re.sub(r"[a-z]", "x", w)))  # letters first, or the "d" gets rewritten
    return re.sub(r"(.)\1+", r"\1", s)


def token_features(tokens: list[str], i: int, lex: Lexicon) -> dict:
    w = tokens[i]
    lw = w.lower()
    key = lw.rstrip(".")
    f = {"bias": 1.0, "w": lw, "suf3": lw[-3:], "suf2": lw[-2:], "pre3": lw[:3], "shape": shape(w), "digit": w.isdigit(),
         "pin_like": w.isdigit() and len(w) == 6, "upper": w.isupper(), "title": w.istitle(),
         "punct": not any(c.isalnum() for c in w),
         "cue_rel": key in RELATION or key in POSTPOSITIONS, "cue_dist": key in DISTRICT_CUES, "cue_house": key in HOUSE_CUES,
         "cue_street": key in STREET_CUES, "cue_bld": key in BUILDING_CUES, "cue_loc": key in LOCALITY_CUES,
         "gaz_office": key in lex.office, "gaz_city": key in lex.city, "gaz_district": key in lex.district,
         "gaz_state": key in lex.state, "state_alias": canonical_state(w) is not None and (len(w) > 3 or w.isupper()),
         "BOS": i == 0, "EOS": i == len(tokens) - 1}
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            nw = tokens[j]
            nk = nw.lower().rstrip(".")
            f[f"{off:+d}:w"] = nw.lower()
            f[f"{off:+d}:shape"] = shape(nw)
            f[f"{off:+d}:cue_rel"] = nk in RELATION
            f[f"{off:+d}:cue_street"] = nk in STREET_CUES
            f[f"{off:+d}:cue_loc"] = nk in LOCALITY_CUES
            f[f"{off:+d}:cue_dist"] = nk in DISTRICT_CUES
    return f


def featurize(records, lex: Lexicon) -> list[list[dict]]:
    return [[token_features(r["tokens"], i, lex) for i in range(len(r["tokens"]))] for r in records]


def train(records, lex: Lexicon, c1: float = 0.1, c2: float = 0.1, max_iter: int = 100) -> sklearn_crfsuite.CRF:
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter, all_possible_transitions=True)
    crf.fit(featurize(records, lex), [r["tags"] for r in records])
    return crf


def predict(crf: sklearn_crfsuite.CRF, records, lex: Lexicon) -> list[list[str]]:
    return [list(p) for p in crf.predict(featurize(records, lex))]


def run(train_path, eval_path, run_id: str, seed: int = 0, n_train: int | None = None, results_dir="results",
        gaz: Gazetteer | None = None, max_iter: int = 100, c1: float = 0.1, c2: float = 0.1, bootstrap_n: int = 1000) -> dict:
    lex = Lexicon(gaz or Gazetteer.load())
    tr = read_jsonl(train_path)
    require_train(tr, str(train_path))
    n_total = len(tr)
    if n_train:  # the seed only matters here: lbfgs on the full set is deterministic, so seeds agree unless n_train is set
        tr = random.Random(seed).sample(tr, min(n_train, n_total))
    crf = train(tr, lex, c1, c2, max_iter)
    ev = read_jsonl(eval_path)
    pred = predict(crf, ev, lex)
    data = f"{Path(eval_path).parent.name}/{Path(eval_path).stem}"
    res = evaluate(ev, pred, run_id=run_id, model="crf", data=data, seed=seed, bootstrap_n=bootstrap_n,
                   config={"train": str(train_path), "n_train": len(tr), "n_train_total": n_total, "c1": c1, "c2": c2,
                           "max_iter": max_iter, "features": "v1"})
    write_results(res, results_dir)
    write_predictions(ev, pred, run_id, results_dir)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    r = run(a.train, a.eval, a.run_id, a.seed, a.n_train, a.results_dir, max_iter=a.max_iter)
    print(f"{a.run_id}: n={r['n']} span micro F1={r['span']['micro']['f1']:.3f} macro F1={r['span']['macro']['f1']:.3f} "
          f"exact={r['exact_match']:.3f} CI95={r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}")
