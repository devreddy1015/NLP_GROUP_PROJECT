"""Real-data annotation plumbing: form CSV -> Label Studio tasks, Label Studio export -> our JSONL, agreement.

    uv run python -m src.annotate tasks --csv form.csv --out data/real            # split by respondent, dedupe by place
    uv run python -m src.annotate export --export ls_export.json --out data/real/dev.jsonl [--annotator 2]
    uv run python -m src.annotate agreement a.jsonl b.jsonl                        # span F1 + token kappa on shared ids

The CSV needs the columns respondent, place, address, pin, city, state (one row per address).
Label Studio config: data/real/labelstudio_config.xml. Offsets are code points; emoji would shift them.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.data import Span, Token, make_record, normalise_value, read_jsonl, tokenize, write_jsonl
from src.eval import span_scores

COLUMNS = ("respondent", "place", "address", "pin", "city", "state")
_TRIM = {",", ";", "(", ")", "[", "]"}  # annotators often drag over a comma; dots and dashes stay (C.G., 5-6-7/A)
# initials.md §5.3: the minimum workable set first (50 dev, 200 test, 50 pool), then up to the full target (250 test, 200 pool)
FILL = (("dev", 50), ("test", 200), ("train", 50), ("test", 250), ("train", 200))


def form_to_tasks(csv_path: str | Path, out_dir: str | Path, fill=FILL, seed: int = 0) -> dict[str, int]:
    """Write Label Studio task files per split. Whole respondents go to one split (build-plan B3, B4).

    The respondent -> split map is kept in <out_dir>/splits.json, so re-running on a longer CSV never moves anyone.
    """
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}; expected {COLUMNS}")
    df = df[df.address.str.strip() != ""]
    if (df.respondent.str.strip() == "").any():
        raise ValueError("every row needs a respondent code (one code per form response, not a name)")
    place, city = df.place.map(normalise_value), df.city.map(normalise_value)
    df = df[~((place + "|" + city).duplicated() & (place != ""))]  # same place in the same city once; blank places all stay
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_file = out / "splits.json"
    split_of = json.loads(split_file.read_text()) if split_file.exists() else {}
    sizes = df.respondent.value_counts()
    filled = {"dev": 0, "test": 0, "train": 0}
    for r, split in split_of.items():
        filled[split] += int(sizes.get(r, 0))
    new = sorted(set(df.respondent) - set(split_of))
    random.Random(seed).shuffle(new)
    for r in new:
        split_of[r] = next((s for s, target in fill if filled[s] < target), "test")
        filled[split_of[r]] += int(sizes[r])
    split_file.write_text(json.dumps(split_of, indent=1, sort_keys=True))
    tasks: dict[str, list] = {}
    for i, row in enumerate(df.itertuples(index=False), 1):
        split = split_of[row.respondent]
        data = {"item_id": f"real-{i:04d}", "text": row.address, "place": row.place, "pin": row.pin, "city": row.city,
                "state": row.state, "respondent": row.respondent, "split": split}
        tasks.setdefault(split, []).append({"data": data})
    for split, items in tasks.items():
        (out / f"tasks_{split}.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    return {split: len(items) for split, items in tasks.items()}


def snap_span(text: str, tokens: list[Token], start: int, end: int, label: str) -> Span:
    """Snap a character span to whole tokens: trim whitespace and stray brackets/commas, then widen to token edges."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    inside = [t for t in tokens if t.end > start and t.start < end] if end > start else []
    while inside and inside[0].text in _TRIM:
        inside = inside[1:]
    while inside and inside[-1].text in _TRIM:
        inside = inside[:-1]
    if not inside:
        raise ValueError(f"empty span {label} at {start}:{end} in {text!r}")
    return inside[0].start, inside[-1].end, label


def labelstudio_to_jsonl(export_path: str | Path, out_path: str | Path, annotator: int | None = None) -> list[dict]:
    """Convert a Label Studio JSON export to our JSONL. Uses the first (or the given annotator's) annotation per task."""
    records, errors = [], []
    for task in json.loads(Path(export_path).read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]  # an all-O item has result == []
        if annotator is not None:
            anns = [a for a in anns if a.get("completed_by") == annotator]
        if not anns:
            continue
        d = task["data"]
        text = d["text"]
        tokens = tokenize(text)
        spans, slices = [], []
        try:
            for r in anns[0].get("result") or []:
                if r["type"] == "labels":
                    spans.append(snap_span(text, tokens, r["value"]["start"], r["value"]["end"], r["value"]["labels"][0]))
                elif r["type"] == "choices":
                    slices += r["value"]["choices"]
            meta = {k: d.get(k, "") for k in ("place", "pin", "city", "state", "respondent")}
            records.append(make_record(d["item_id"], text, sorted(spans), "real", d["split"], slices=slices, meta=meta))
        except ValueError as e:
            errors.append(f"{d['item_id']}: {e}")
    if errors:  # report every bad item at once so the annotator fixes them in one pass
        raise ValueError("\n".join(errors))
    write_jsonl(out_path, records)
    return records


def agreement(a_path: str | Path, b_path: str | Path) -> dict:
    """Inter-annotator agreement on the items both files share: strict span F1 (A as reference) and token-level Cohen's kappa."""
    a = {r["id"]: r for r in read_jsonl(a_path)}
    b = {r["id"]: r for r in read_jsonl(b_path)}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two files share no item ids")
    tags_a, tags_b = [a[i]["tags"] for i in ids], [b[i]["tags"] for i in ids]
    span = span_scores(tags_a, tags_b)
    kappa = cohen_kappa_score([t for ts in tags_a for t in ts], [t for ts in tags_b for t in ts])
    return {"n": len(ids), "span_f1_micro": span["micro"]["f1"], "per_label_f1": {k: v["f1"] for k, v in span["per_label"].items()},
            "token_kappa": float(kappa)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("--csv", required=True); t.add_argument("--out", default="data/real")
    e = sub.add_parser("export"); e.add_argument("--export", required=True); e.add_argument("--out", required=True)
    e.add_argument("--annotator", type=int, default=None)
    g = sub.add_parser("agreement"); g.add_argument("a"); g.add_argument("b")
    args = p.parse_args()
    if args.cmd == "tasks":
        print(form_to_tasks(args.csv, args.out))
    elif args.cmd == "export":
        print(f"wrote {len(labelstudio_to_jsonl(args.export, args.out, args.annotator))} records to {args.out}")
    else:
        print(json.dumps(agreement(args.a, args.b), indent=1))
