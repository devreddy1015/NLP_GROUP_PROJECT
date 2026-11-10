"""Report tables built from results/ and nothing else, so that every number traces to a file (CLAUDE.md rule 3).

    uv run python -m scripts.make_tables [--results results] [--out report/tables]

runs.md        one row per results file of a tagger or a JSON system
configs.md     the seeds of one configuration averaged (mean ± standard deviation)
shared.md      E1: each existing system on the labels it shares with us, and our systems rescored on those same labels
compare.md     E2-E4: one configuration minus another, with a paired bootstrap over the same addresses; an interval
               that includes zero is "no measurable effect" (initials.md §8)
normaliser.md  E5: PIN checks and fills

shortcut: tables only, no figures (matplotlib is not installed); add them when the report needs a plot.
"""
import argparse
import json
import re
import statistics
from datetime import date
from pathlib import Path

from src.data import read_jsonl
from src.eval import field_scores, fields_from, paired_bootstrap, read_predictions, span_scorer
from src.models.llm import fold, gold_fields
from src.models.zeroshot import only

_EVALS = ("realdev", "realtest")
PAIRS = [pair for ev in _EVALS for pair in (
    [("E2: full noise minus clean", f"muril-synfull-{ev}", f"muril-synclean-{ev}")]
    + [(f"E3: full noise minus (full without N{n})", f"muril-synfull-{ev}", f"muril-synfullminusN{n}-{ev}") for n in range(1, 9)]
    + [(f"E4: k = {k} minus k = 0", f"muril-synfullk{k}-{ev}", f"muril-synfull-{ev}") for k in (25, 50, 100, 200)]
    + [("E4 (H3): k = 200 minus 40k synthetic", f"muril-synfullk200-{ev}", f"muril-synfull40k-{ev}"),
       ("XLM-R minus MuRIL, full noise", f"xlmr-synfull-{ev}", f"muril-synfull-{ev}")])]


def load(results_dir) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(results_dir).glob("*.json"))]


def config_key(run_id: str) -> str:
    """The run id without its date and seed: the runs of one configuration share it."""
    return re.sub(r"^\d{8}-|-s\d+$", "", run_id)


def gold_path(data: str) -> Path:
    """The gold file behind a results file's `data` name ("real/dev", "full/dev", "fixtures/handlabelled20")."""
    options = [Path(f"data/{data}.jsonl"), Path(f"data/synthetic/{data}.jsonl"), Path(f"tests/{data}.jsonl"),
               Path(f"tests/fixtures/{data}.jsonl")]
    found = [p for p in options if p.exists()]
    if not found:
        raise FileNotFoundError(f"no gold file for {data!r}: looked in {[str(p) for p in options]}")
    return found[0]


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]) + "\n"


def _mean_std(values: list[float]) -> str:
    if not values:
        return "–"
    return f"{statistics.mean(values):.3f} ± " + (f"{statistics.stdev(values):.3f}" if len(values) > 1 else "–")


def runs_table(results: list[dict]) -> str:
    rows = []
    for r in (r for r in results if "field" in r):
        span = r.get("span")  # a JSON system has no spans
        rows.append([r["run_id"], r["model"], r["data"], r["n"],
                     f"{span['micro']['f1']:.3f} ({r['ci95'][0]:.3f}–{r['ci95'][1]:.3f})" if span else "–",
                     f"{span['macro']['f1']:.3f}" if span else "–", f"{r['field']['micro']['f1']:.3f}",
                     f"{r['exact_match']:.3f}" if span else "–"])
    return _table(["Run ID", "Model", "Scored on", "n", "Span micro F1 (95% interval)", "Span macro F1", "Field micro F1",
                   "Exact match"], rows)


def configs_table(results: list[dict]) -> str:
    groups: dict[str, list[dict]] = {}
    for r in (r for r in results if "field" in r):
        groups.setdefault(config_key(r["run_id"]), []).append(r)
    rows = [[key, runs[0]["model"], runs[0]["data"], ", ".join(str(s) for s in sorted(r["seed"] for r in runs)),
             _mean_std([r["span"]["micro"]["f1"] for r in runs if "span" in r]),
             _mean_std([r["field"]["micro"]["f1"] for r in runs])] for key, runs in sorted(groups.items())]
    return _table(["Configuration", "Model", "Scored on", "Seeds", "Span micro F1 (mean ± std)", "Field micro F1 (mean ± std)"],
                  rows)


def _same_labels_f1(theirs: dict, records: list[dict], pred: list[list[str]]) -> float:
    """Field micro F1 of one of our systems, scored the way `theirs` was: on its shared labels, by its rules."""
    shared, either = theirs["config"]["shared_labels"], tuple(theirs["config"].get("either", ()))
    if "span" in theirs:  # a tagger: blank every other label on both sides, as zeroshot.run does
        gold_f = [fields_from(r["text"], only(r["tags"], set(shared))) for r in records]
        pred_f = [fields_from(r["text"], only(p, set(shared))) for r, p in zip(records, pred)]
    else:  # a JSON system: states in full, and its one key may stand for several of our labels, as in llm.run
        pred_f = [gold_fields({"text": r["text"], "tags": p}) for r, p in zip(records, pred)]
        if either:  # our tags name the district DISTRICT; put it under the same key their answer uses
            pred_f = [{**{k: v for k, v in f.items() if k not in either},
                       either[0]: set().union(*(f.get(label, set()) for label in either))} for f in pred_f]
        gold_f = [fold(gold_fields(r), f, either) for r, f in zip(records, pred_f)]
    return field_scores(gold_f, pred_f, shared)["micro"]["f1"]


def shared_table(results: list[dict], results_dir) -> str:
    rows = []
    ours = [r for r in results if "span" in r and "shared_labels" not in r.get("config", {})]
    for theirs in (r for r in results if "shared_labels" in r.get("config", {})):
        records = read_jsonl(gold_path(theirs["data"]))
        for r in ours:
            pred_file = Path(results_dir) / f"{r['run_id']}.pred.jsonl"
            if Path(r["data"]).name == Path(theirs["data"]).name and pred_file.exists():
                rows.append([theirs["run_id"], ", ".join(theirs["config"]["shared_labels"]),
                             f"{theirs['field']['micro']['f1']:.3f}", r["run_id"],
                             f"{_same_labels_f1(theirs, records, read_predictions(pred_file, records)):.3f}", theirs["data"]])
    return _table(["Existing system", "Labels it shares with us", "Its field micro F1", "Our system",
                   "Ours on the same labels", "Scored on"], rows)


def compare_table(results: list[dict], results_dir, pairs=PAIRS, n: int = 1000) -> str:
    by_key: dict[str, list[dict]] = {}
    for r in (r for r in results if "span" in r and (Path(results_dir) / f"{r['run_id']}.pred.jsonl").exists()):
        by_key.setdefault(config_key(r["run_id"]), []).append(r)
    rows = []
    for title, key_a, key_b in pairs:
        a, b = by_key.get(key_a, []), by_key.get(key_b, [])
        if not a or not b or a[0]["data"] != b[0]["data"]:
            rows.append([title, "–", "–", "–", "–", "not run yet"])
            continue
        records = read_jsonl(gold_path(a[0]["data"]))
        gold = [r["tags"] for r in records]

        def mean_score(runs):  # the mean over seeds of micro span F1, on whichever addresses a resample holds
            scorers = [span_scorer(gold, read_predictions(Path(results_dir) / f"{r['run_id']}.pred.jsonl", records)) for r in runs]
            return lambda idx: statistics.mean(score(idx) for score in scorers)

        by_respondent = all(r.get("meta", {}).get("respondent") for r in records)
        groups = [r["meta"]["respondent"] for r in records] if by_respondent else None
        d = paired_bootstrap(mean_score(a), mean_score(b), len(records), n=n, groups=groups)
        lo, hi = d["ci95"]
        verdict = "first is higher" if lo > 0 else "second is higher" if hi < 0 else "no measurable effect"
        rows.append([title, a[0]["data"], f"{len(a)} and {len(b)}", f"{d['diff']:+.3f}", f"{lo:+.3f} to {hi:+.3f}", verdict])
    return _table(["Comparison (first minus second)", "Scored on", "Seeds", "Difference in span micro F1", "95% interval",
                   "Verdict"], rows)


def normaliser_table(results: list[dict]) -> str:
    def filled(d: dict) -> str:
        return f"{d['correct']} of {d['n']}" if d["n"] else "–"
    rows = [[r["run_id"], r["data"], r["n"], *(r["pin_check"].get(k, 0) for k in ("valid", "missing", "not_found", "mismatch")),
             filled(r["filled_accuracy"]["district"]), filled(r["filled_accuracy"]["state"])]
            for r in results if "pin_check" in r]
    return _table(["Run ID", "Scored on", "n", "PIN valid", "PIN missing", "PIN not found", "PIN mismatch",
                   "District filled right", "State filled right"], rows)


def main(results_dir="results", out_dir="report/tables", n: int = 1000) -> None:
    results = load(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    note = (f"Generated by `scripts/make_tables.py` from `{results_dir}/` on {date.today().isoformat()}. "
            "Do not edit by hand: run the script again.\n\n")
    tables = {"runs": runs_table(results), "configs": configs_table(results), "shared": shared_table(results, results_dir),
              "compare": compare_table(results, results_dir, n=n), "normaliser": normaliser_table(results)}
    for name, table in tables.items():
        (out / f"{name}.md").write_text(note + table, encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report/tables")
    a = p.parse_args()
    main(a.results, a.out)
    print(f"tables written to {a.out}/")
