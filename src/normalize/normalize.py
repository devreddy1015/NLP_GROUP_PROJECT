"""Step 13: tagged spans -> clean JSON with PIN-directory checks (initials.md §2, §8).

Expands abbreviations, splits each landmark into relation + anchor, canonicalises the state, cleans the PIN,
checks it against the directory and fills district / state when the address leaves them out.
Run: uv run python -m src.normalize.normalize --data <jsonl> --run-id YYYYMMDD-normalizer-<data>-s0   (uses the gold tags)
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path

from src.data import normalise_value, read_jsonl, tags_to_spans, tokenize
from src.eval import write_results
from src.generate.noise import ABBREV  # the normaliser undoes exactly what N1 does
from src.models.rules import DISTRICT_CUES, LOCALITY_CUES
from src.normalize.gazetteer import STATES, Gazetteer, canonical_state

# shortcut: untuned similarity cut-off for a misspelt state or district; tune it on E5's injected PIN errors
_TYPO = 0.8
_AMBIGUOUS = {"off", "col", "cal", "ind", "mum", "fl", "gr"}  # ordinary words too ("Off MG Road", "Col. Sharma Marg")
_EXPAND = {a.lower().rstrip("."): word.title() for word, aliases in ABBREV.items() if word != "number"
           for a in aliases if len(a.rstrip(".")) > 1 and a.lower().rstrip(".") not in _AMBIGUOUS}
_RENAMED = {"bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "gurgaon": "gurugram",
            "poona": "pune", "baroda": "vadodara", "allahabad": "prayagraj", "cochin": "kochi", "trivandrum": "thiruvananthapuram",
            "mysore": "mysuru", "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "vizag": "visakhapatnam"}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_PREFIX = {"near": "near", "nr": "near", "close to": "near", "nearby": "near", "opp": "opposite", "opposite": "opposite",
           "in front of": "opposite", "infront of": "opposite", "front of": "opposite", "facing": "opposite", "behind": "behind",
           "back side of": "behind", "backside of": "behind", "backside": "behind", "bh": "behind", "bhd": "behind",
           "beside": "beside", "besides": "beside", "next to": "beside", "adjacent to": "beside", "adj to": "beside",
           "adjacent": "beside", "above": "above", "below": "below", "inside": "inside"}
_SUFFIX = {"ke paas": "near", "ke pass": "near", "ke pas": "near", "ke samne": "opposite", "ke saamne": "opposite",
           "ke samney": "opposite", "ke peeche": "behind", "ke piche": "behind", "ke pichhe": "behind", "ke bagal me": "beside",
           "ke bagal mein": "beside", "ke bagal": "beside", "ke upar": "above", "ke niche": "below", "ke andar": "inside"}


def expand(value: str) -> str:
    """Fix case (all-lower / all-caps input) and expand abbreviations word by word."""
    letters = [c for c in value if c.isalpha()]
    words = value.split()
    if letters and all(c.islower() for c in letters):
        words = [w.capitalize() for w in words]
    elif letters and all(c.isupper() for c in letters):
        words = [w if len(w) <= 4 else w.capitalize() for w in words]  # short all-caps words are acronyms (SBI, ATM, NH)
    out = []
    for i, w in enumerate(words):
        key = w.lower().rstrip(".")
        nxt = words[i + 1] if i + 1 < len(words) else ""
        # short forms only at the end, before a number or before another short form ("MG Rd", "Sec 10", "Stn Rd")
        if len(key) <= 3 and nxt and not (nxt[:1].isdigit() or nxt.lower().rstrip(".") in _EXPAND):
            out.append(w)
        else:
            out.append(_EXPAND.get(key, w))
    return " ".join(out)


def parse_landmark(span: str) -> dict:
    """Split 'behind SBI atm' / 'hanuman mandir ke paas' into a canonical relation and the anchor."""
    relation, low = None, span.lower()
    for p in sorted(_PREFIX, key=len, reverse=True):
        m = re.match(rf"{re.escape(p)}(?:\.\s*|\s+)", low)
        if m:
            relation, span, low = _PREFIX[p], span[m.end():].strip(), low[m.end():].strip()
            break
    for s in sorted(_SUFFIX, key=len, reverse=True):
        if low.endswith(" " + s):
            relation, span = relation or _SUFFIX[s], span[:-len(s)].strip()
            break
    return {"relation": relation, "anchor": expand(span)}


def normalise(text: str, tags: list[str], gaz: Gazetteer) -> dict:
    by = defaultdict(list)
    for s, e, label in tags_to_spans(tokenize(text), tags):
        by[label].append(text[s:e])
    join = lambda vals: ", ".join(expand(v) for v in vals) or None
    state = by["STATE"][0] if by["STATE"] else None
    pins = [re.sub(r"[^0-9]", "", p.translate(_DEVANAGARI_DIGITS)) for p in by["PIN"]]
    known = [p for p in pins if len(p) == 6 and gaz.lookup(p)]
    pin = known[0] if known else next((p for p in pins if len(p) == 6), pins[0] if pins else None)
    out = {"house": join(by["HOUSE"]), "building": join(by["BUILDING"]), "street": join(by["STREET"]),
           "landmarks": [parse_landmark(v) for v in by["LANDMARK"]], "locality": join(by["LOCALITY"]), "city": join(by["CITY"]),
           "district": join(by["DISTRICT"]), "state": _state(state) if state else None,
           "pin": pin, "pin_check": "missing", "filled": [], "locality_known": None}
    hit = gaz.lookup(pin) if pin and len(pin) == 6 else None
    if pin and not hit:
        out["pin_check"] = "not_found"
    elif hit:
        state_ok = _unchecked(out["state"]) or normalise_value(out["state"]) == normalise_value(hit["state"])
        district_ok = _unchecked(out["district"]) or any(_same_district(out["district"], d) for d in hit["districts"])
        out["pin_check"] = "valid" if state_ok and district_ok else "mismatch"
        if out["pin_check"] == "valid":
            only_district = hit["district"] if len(hit["districts"]) == 1 else None  # a PIN spanning districts cannot say which
            for field, value in (("district", only_district), ("state", canonical_state(hit["state"]) or hit["state"])):
                if out[field] is None and value:
                    out[field] = value
                    out["filled"].append(field)
        if out["locality"]:
            words = {w for w in normalise_value(out["locality"]).split() if len(w) >= 3 and w not in LOCALITY_CUES}
            out["locality_known"] = any(words & set(normalise_value(o).split()) for o in hit["offices"])
    return out


def _district_words(name: str) -> set[str]:
    return {_RENAMED.get(w, w) for w in normalise_value(name).split() if w not in DISTRICT_CUES}


def _same_district(written: str, directory: str) -> bool:
    """'Durg District' == 'Durg'; 'Bangalore' == 'Bengaluru Urban'; 'Patiyala' == 'Patiala': cue words dropped, old names
    mapped, subset either way, or a near-identical spelling."""
    a, b = _district_words(written), _district_words(directory)
    if not a:
        return False
    return a <= b or b <= a or SequenceMatcher(None, " ".join(sorted(a)), " ".join(sorted(b))).ratio() >= _TYPO


def _state(written: str) -> str:
    """Canonical state name, also for a misspelt one ('Madhya Pradewsh'); otherwise the text as written."""
    close = get_close_matches(written.title(), STATES, n=1, cutoff=_TYPO)
    return canonical_state(written) or (close[0] if close else expand(written))


def _unchecked(value: str | None) -> bool:
    """Nothing to hold against the directory: the field is absent or not in Latin script."""
    return value is None or not value.isascii()


def score(records, tags, gaz: Gazetteer, run_id: str, data: str) -> dict:
    """E5: pin_check counts and the accuracy of filled district / state against `meta` where it is known."""
    outs = [normalise(r["text"], t, gaz) for r, t in zip(records, tags)]
    filled = {f: {"n": 0, "correct": 0} for f in ("district", "state")}
    for r, o in zip(records, outs):
        truth = r.get("meta", {})
        for f in filled:
            if f in o["filled"] and truth.get(f):
                want = canonical_state(truth[f]) or truth[f] if f == "state" else truth[f]
                filled[f]["n"] += 1
                filled[f]["correct"] += _same_district(o[f], want) if f == "district" else normalise_value(o[f]) == normalise_value(want)
    return {"run_id": run_id, "model": "normalizer", "data": data, "seed": 0,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(records),
            "pin_check": dict(Counter(o["pin_check"] for o in outs)),
            "filled_accuracy": {f: {**v, "accuracy": v["correct"] / v["n"] if v["n"] else None} for f, v in filled.items()},
            "mismatch_ids": [r["id"] for r, o in zip(records, outs) if o["pin_check"] == "mismatch"][:20]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    recs = read_jsonl(a.data)
    res = score(recs, [r["tags"] for r in recs], Gazetteer.load(), a.run_id, f"{Path(a.data).parent.name}/{Path(a.data).stem}")
    write_results(res, a.results_dir)
    print(json.dumps({k: res[k] for k in ("run_id", "n", "pin_check", "filled_accuracy")}, indent=1))
