"""Systems that answer in JSON, not tags: the generative address parser (Z, E1) and the few-shot LLM (L1, E6).

A chat prompt goes in and a JSON object comes out. Its keys are mapped onto our labels and scored at field level only:
there are no spans, so there is no span F1. State names are written in full on both sides, because the parser turns
MH into Maharashtra. The mappings come from research.md §2 and stay provisional until checked on real dev (build-plan B14).

Run: uv run python -m src.models.llm --system abr-parser --data data/real/dev.jsonl \
         --run-id YYYYMMDD-abr-parser-realdev-s0
"""
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import LABELS, normalise_value, read_jsonl, tags_to_spans, tokenize
from src.eval import field_scores, write_results
from src.normalize.gazetteer import canonical_state


@dataclass(frozen=True)
class System:
    repo: str  # Hugging Face name, or a local folder
    revision: str | None  # hub commit, pinned so a run can be repeated (checked 11 Nov 2026)
    prompt: str  # the system prompt
    mapping: dict[str, str]  # their JSON key -> our label; a key that is left out is ignored
    either: tuple[str, ...] = ()  # labels of ours that one key of theirs may hold; that key maps to the first of them
    shots: tuple[tuple[str, str], ...] = ()  # few-shot examples: (address, the JSON answer)


SYSTEMS = {
    # the prompt is word for word from the model card, which says the model was trained with it
    "abr-parser": System(
        "abr-6024/indian-address-parser-1.5B", "483951f3d7855ae47e0659f34ba1d2145055dcc7",
        "You are an Indian address parser. Convert the address into JSON with exactly these keys: "
        "flat, building, street, landmark, area, city, state, pincode, phone. "
        "Use an empty string for anything missing. Output only JSON.",
        {"flat": "HOUSE", "building": "BUILDING", "street": "STREET", "landmark": "LANDMARK", "area": "LOCALITY",
         "city": "CITY", "state": "STATE", "pincode": "PIN"},
        either=("CITY", "DISTRICT")),  # its `city` holds the district, or the metro for a big city
}


def parse_json(text: str) -> dict | None:
    """The {...} object in the model's answer; None if there is none or it does not parse."""
    start, end = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[start:end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _value(label: str, raw: str) -> str:
    return normalise_value(canonical_state(raw) or raw) if label == "STATE" else normalise_value(raw)


def to_fields(obj: dict | None, mapping: dict[str, str]) -> dict[str, set[str]]:
    """Our label -> the set of normalised strings the system gave for it. Blank values are dropped."""
    fields: dict[str, set[str]] = {}
    for key, label in mapping.items():
        value = _value(label, str((obj or {}).get(key) or ""))
        if value:
            fields.setdefault(label, set()).add(value)
    return fields


def gold_fields(record: dict) -> dict[str, set[str]]:
    fields: dict[str, set[str]] = {}
    for start, end, label in tags_to_spans(tokenize(record["text"]), record["tags"]):
        fields.setdefault(label, set()).add(_value(label, record["text"][start:end]))
    return fields


def fold(gold: dict[str, set[str]], pred: dict[str, set[str]], either: tuple[str, ...]) -> dict[str, set[str]]:
    """One key of theirs may hold any of several labels of ours (the parser's `city` is our CITY or our DISTRICT).

    The gold then counts as found if the prediction matches any of them, and as a single miss otherwise.
    """
    union = set().union(*(gold.get(label, set()) for label in either))
    rest = {label: values for label, values in gold.items() if label not in either}
    if not union:
        return rest
    return {**rest, either[0]: (union & pred.get(either[0], set())) or {min(union)}}


def messages(prompt: str, address: str, shots=()) -> list[dict]:
    chat = [{"role": "system", "content": prompt}]
    for shot_address, answer in shots:
        chat += [{"role": "user", "content": shot_address}, {"role": "assistant", "content": answer}]
    return chat + [{"role": "user", "content": address}]


@torch.no_grad()
def generate(net, tok, prompt: str, addresses: list[str], shots=(), batch_size: int = 8,
             max_new_tokens: int = 200) -> list[str]:
    """One answer per address: greedy decoding, with the prompt cut off the front."""
    tok.padding_side = "left"  # a causal model continues from the right-hand end of each row
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    net.eval()
    out = []
    for i in range(0, len(addresses), batch_size):
        texts = [tok.apply_chat_template(messages(prompt, a, shots), tokenize=False, add_generation_prompt=True)
                 for a in addresses[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(net.device)
        gen = net.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


def run(system: str, data_path, run_id: str, results_dir="results", device: str | None = None) -> dict:
    spec = SYSTEMS[system]
    records = read_jsonl(data_path)
    if not records:
        raise ValueError(f"{data_path} is empty")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
    # safetensors only: these are other people's weights, and a pickle file can run code when it is loaded
    net = AutoModelForCausalLM.from_pretrained(spec.repo, revision=spec.revision, use_safetensors=True,
                                               dtype=torch.float16 if device.type == "cuda" else torch.float32).to(device)
    answers = generate(net, tok, spec.prompt, [r["text"] for r in records], spec.shots)
    objects = [parse_json(a) for a in answers]
    pred = [to_fields(obj, spec.mapping) for obj in objects]
    gold = [fold(gold_fields(r), p, spec.either) for r, p in zip(records, pred)]
    shared = [label for label in LABELS if label in set(spec.mapping.values())]
    res = {"run_id": run_id, "model": system, "data": f"{Path(data_path).parent.name}/{Path(data_path).stem}", "seed": 0,
           "config": {"repo": spec.repo, "revision": spec.revision, "prompt": spec.prompt, "mapping": spec.mapping,
                      "either": list(spec.either), "shared_labels": shared, "n_shots": len(spec.shots),
                      "torch": torch.__version__, "transformers": transformers.__version__},
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(records),
           "json_valid": sum(obj is not None for obj in objects) / len(records),
           "field": field_scores(gold, pred, shared)}
    write_results(res, results_dir)
    # the raw answers, kept for error analysis and for the paired bootstrap
    (Path(results_dir) / f"{run_id}.pred.jsonl").write_text(
        "".join(json.dumps({"id": r["id"], "output": a}, ensure_ascii=False) + "\n" for r, a in zip(records, answers)),
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
    print(f"{a.run_id}: n={r['n']} valid JSON={r['json_valid']:.2f} shared={len(r['config']['shared_labels'])} "
          f"field micro P={r['field']['micro']['p']:.3f} R={r['field']['micro']['r']:.3f} F1={r['field']['micro']['f1']:.3f}")
