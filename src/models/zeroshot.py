"""Z: existing open address taggers, run as they are (no training), their labels mapped onto ours (E1).

Each model reads the raw address text, as its model card does, and a token of ours takes the label of the first
subword that overlaps it. A system is scored only on the labels it shares with us: gold spans of any other label are
blanked. Stray commas and brackets at the ends of a predicted span are dropped, by the rule human annotations get:
these models tag "Bhilai," with its comma, which is a tokenising habit and not a parsing error.
The mappings come from research.md §2 and stay provisional until they are checked on real dev (build-plan B14).

Run: uv run python -m src.models.zeroshot --system mukuls-muril --data data/real/dev.jsonl \
         --run-id YYYYMMDD-mukuls-muril-realdev-s0
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.annotate import snap_span
from src.data import LABELS, read_jsonl, spans_to_tags, tags_to_spans, tokenize
from src.eval import evaluate, write_results


@dataclass(frozen=True)
class System:
    repo: str  # Hugging Face name, or a local folder
    revision: str | None  # hub commit, pinned so a run can be repeated (checked 11 Nov 2026)
    max_len: int  # the context length its model card uses
    mapping: dict[str, str]  # their label -> ours; a label of theirs that is left out becomes O


_SHIPROCKET = {"house_details": "HOUSE", "floor": "HOUSE", "building_name": "BUILDING", "road": "STREET",
               "landmarks": "LANDMARK", "locality": "LOCALITY", "sub_locality": "LOCALITY", "city": "CITY",
               "state": "STATE", "pincode": "PIN"}  # their "country" has no label of ours
SYSTEMS = {
    "shiprocket-indicbert": System("shiprocket-ai/open-indicbert-indian-address-ner",
                                   "4e5d716c92c9de41007fad1fe12504702364ff09", 128, _SHIPROCKET),
    "shiprocket-tinybert": System("shiprocket-ai/open-tinybert-indian-address-ner",
                                  "fc11437846d50eab01a65df9ec7f105495976ff8", 128, _SHIPROCKET),
    "mukuls-muril": System("mukuls9971/muril-indian-address-ner-v1", "6cdb861744e4a30cbbc0bf231deb8e737923d51d", 192,
                           {f"ADDRESS_{label}": label for label in LABELS}),
    "gagan-tinybert": System("gagan1985/tinybert-4l-312d-indian-address-parser",
                             "20d532b406d85edf7d7c92562719c62cbc463205", 160,
                             {"houseNumber": "HOUSE", "houseName": "HOUSE", "poi": "LANDMARK", "street": "STREET",
                              "subsubLocality": "LOCALITY", "subLocality": "LOCALITY", "locality": "LOCALITY",
                              "village": "LOCALITY", "district": "DISTRICT", "subDistrict": "DISTRICT", "city": "CITY",
                              "state": "STATE", "pincode": "PIN"}),
}


def to_ours(theirs: list[str], mapping: dict[str, str]) -> list[str]:
    """Their tag per token -> our IOB2 tags. A span starts at their B-, or wherever our label changes."""
    out, prev = [], None
    for their_tag in theirs:
        prefix, _, name = their_tag.partition("-")  # "B-city" -> "B", "city"; "O" has no name
        label = mapping.get(name)
        out.append("O" if label is None else ("B-" if prefix == "B" or label != prev else "I-") + label)
        prev = label
    return out


def trim(text: str, tags: list[str]) -> list[str]:
    """Drop stray commas and brackets from the ends of each span, exactly as annotate.snap_span does for annotators."""
    tokens = tokenize(text)
    spans = []
    for start, end, label in tags_to_spans(tokens, tags):
        try:
            spans.append(snap_span(text, tokens, start, end, label))
        except ValueError:  # the span held nothing but commas or brackets, so nothing is left of it
            continue
    return spans_to_tags(tokens, spans)


def only(tags: list[str], labels: set[str]) -> list[str]:
    """Tags of any other label become O, so that a system is scored on the shared labels alone."""
    return [t if t[2:] in labels else "O" for t in tags]


@torch.no_grad()
def tag(net, tok, records: list[dict], max_len: int, batch_size: int = 32) -> list[list[str]]:
    """The model's own label for each token of ours: that of the first subword overlapping it in the raw text."""
    net.eval()
    out = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len, padding=True,
                  return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        best = net(**{k: v.to(net.device) for k, v in enc.items()}).logits.argmax(-1).tolist()
        for j, r in enumerate(chunk):
            tags = []
            for t in tokenize(r["text"]):
                # special and padding subwords span (0, 0) and never overlap; a token past max_len, or one the tokenizer drops, stays O
                pos = next((p for p, (s, e) in enumerate(offsets[j]) if s < t.end and e > t.start), None)
                tags.append("O" if pos is None else net.config.id2label[best[j][pos]])
            out.append(tags)
    return out


def run(system: str, data_path, run_id: str, results_dir="results", bootstrap_n: int = 1000,
        device: str | None = None) -> dict:
    spec = SYSTEMS[system]
    records = read_jsonl(data_path)
    if not records:
        raise ValueError(f"{data_path} is empty")
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    # safetensors only: these are other people's weights, and a pickle file can run code when it is loaded
    net = AutoModelForTokenClassification.from_pretrained(spec.repo, revision=spec.revision, use_safetensors=True)
    net = net.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names = {name.partition("-")[2] for name in net.config.id2label.values()}
    if unknown := sorted(set(spec.mapping) - names):
        raise ValueError(f"{system}: the model has no label {unknown}")
    shared = set(spec.mapping.values())
    theirs = tag(net, tok, records, spec.max_len)
    pred = [trim(r["text"], to_ours(tags, spec.mapping)) for r, tags in zip(records, theirs)]
    gold = [{**r, "tags": only(r["tags"], shared)} for r in records]
    data = f"{Path(data_path).parent.name}/{Path(data_path).stem}"
    res = evaluate(gold, pred, run_id=run_id, model=system, data=data, seed=0, bootstrap_n=bootstrap_n,
                   config={"repo": spec.repo, "revision": spec.revision, "max_len": spec.max_len, "mapping": spec.mapping,
                           "shared_labels": [label for label in LABELS if label in shared],
                           "torch": torch.__version__, "transformers": transformers.__version__})
    write_results(res, results_dir)
    # the model's own tags are kept beside ours: the mapping still has to be audited on real dev (build-plan B14)
    (Path(results_dir) / f"{run_id}.pred.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "theirs": t, "tags": p}) + "\n" for r, t, p in zip(records, theirs, pred)),
        encoding="utf-8")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    p.add_argument("--data", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    r = run(a.system, a.data, a.run_id, a.results_dir)
    print(f"{a.run_id}: n={r['n']} shared={len(r['config']['shared_labels'])} span micro F1={r['span']['micro']['f1']:.3f} "
          f"macro F1={r['span']['macro']['f1']:.3f} field micro F1={r['field']['micro']['f1']:.3f} "
          f"exact={r['exact_match']:.3f} CI95={r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}")
