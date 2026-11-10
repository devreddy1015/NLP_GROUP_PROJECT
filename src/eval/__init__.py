"""Evaluation: strict IOB2 span F1 (seqeval), exact match, field-level set P/R/F1, bootstrap CI, slices.

`evaluate()` assembles everything into one dict; `write_results()` saves it as results/<run-id>.json.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

from src.data import LABELS, normalise_value, tags_to_spans, tokenize

Tags = Sequence[Sequence[str]]
_STRICT = {"mode": "strict", "scheme": IOB2, "zero_division": 0}


def span_scores(gold: Tags, pred: Tags) -> dict:
    # macro averages over the labels present in gold or pred; compare runs together with `label_support`
    rep = classification_report(list(map(list, gold)), list(map(list, pred)), output_dict=True, **_STRICT)
    def prf(d): return {"p": d["precision"], "r": d["recall"], "f1": d["f1-score"], "support": int(d["support"])}
    per_label = {k: prf(v) for k, v in rep.items() if not k.endswith(" avg")}
    return {"micro": prf(rep["micro avg"]), "macro": prf(rep["macro avg"]), "per_label": per_label}


def exact_match(gold: Tags, pred: Tags) -> float:
    return sum(list(g) == list(p) for g, p in zip(gold, pred)) / len(gold)


def fields_from(text: str, tags: Sequence[str]) -> dict[str, set[str]]:
    """Field -> set of normalised span strings, from the address text and its IOB2 tags."""
    fields: dict[str, set[str]] = {}
    for s, e, label in tags_to_spans(tokenize(text), list(tags)):
        fields.setdefault(label, set()).add(normalise_value(text[s:e]))
    return fields


def field_scores(gold: Sequence[dict], pred: Sequence[dict], fields: Iterable[str] = LABELS) -> dict:
    """Per-field and micro P/R/F1 over sets of normalised strings, restricted to `fields` (shared fields only)."""
    counts = {f: [0, 0, 0] for f in fields}  # tp, fp, fn
    for g, p in zip(gold, pred):
        for f in counts:
            gs, ps = set(g.get(f, ())), set(p.get(f, ()))
            counts[f][0] += len(gs & ps); counts[f][1] += len(ps - gs); counts[f][2] += len(gs - ps)
    per_field = {f: {**_prf(*c), "support": c[0] + c[2]} for f, c in counts.items()}
    micro = _prf(*[sum(c[i] for c in counts.values()) for i in range(3)])
    return {"per_field": per_field, "micro": micro}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def bootstrap_ci(gold: Tags, pred: Tags, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI of micro span F1, resampling addresses."""
    rng = random.Random(seed)
    gold, pred = list(map(list, gold)), list(map(list, pred))
    scores = []
    for _ in range(n):
        idx = [rng.randrange(len(gold)) for _ in gold]
        scores.append(f1_score([gold[i] for i in idx], [pred[i] for i in idx], average="micro", **_STRICT))
    scores.sort()
    return scores[int(0.025 * n)], scores[min(int(0.975 * n), n - 1)]


def slice_scores(records: Sequence[dict], pred: Tags) -> dict:
    names = sorted({s for r in records for s in r.get("slices", [])})
    out = {}
    for name in names:
        idx = [i for i, r in enumerate(records) if name in r.get("slices", [])]
        g, p = [records[i]["tags"] for i in idx], [pred[i] for i in idx]
        out[name] = {"n": len(idx), "f1": f1_score(list(map(list, g)), list(map(list, p)), average="micro", **_STRICT)}
    return out


def evaluate(records: Sequence[dict], pred: Tags, run_id: str, model: str, data: str, seed: int,
             config: dict | None = None, bootstrap_n: int = 1000) -> dict:
    if len(records) != len(pred) or any(len(r["tags"]) != len(p) for r, p in zip(records, pred)):
        raise ValueError("predictions do not line up with records")
    gold = [r["tags"] for r in records]
    span = span_scores(gold, pred)
    gold_fields = [fields_from(r["text"], r["tags"]) for r in records]
    pred_fields = [fields_from(r["text"], p) for r, p in zip(records, pred)]
    return {
        "run_id": run_id, "model": model, "data": data, "seed": seed, "config": config or {},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(records),
        "span": span, "exact_match": exact_match(gold, pred),
        "field": field_scores(gold_fields, pred_fields),
        "ci95": list(bootstrap_ci(gold, pred, n=bootstrap_n, seed=seed)),
        "slices": slice_scores(records, pred),
        "label_support": {label: span["per_label"].get(label, {}).get("support", 0) for label in LABELS},
    }


def write_results(result: dict, results_dir: str | Path = "results") -> Path:
    path = Path(results_dir) / f"{result['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
