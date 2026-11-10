"""B0: rules + gazetteer baseline. Splits the address at commas and labels each chunk from cue words,
PIN regexes and gazetteer lookups. Deliberately simple: the floor the learned models must beat.

Run: uv run python -m src.models.rules --data tests/fixtures/handlabelled20.jsonl --run-id <id>
"""
import argparse
from pathlib import Path

from src.data import Span, Token, read_jsonl, spans_to_tags, tokenize
from src.data import normalise_value
from src.eval import evaluate, write_predictions, write_results
from src.normalize.gazetteer import Gazetteer, canonical_state

RELATION = {"near", "nr", "opp", "opposite", "behind", "beside", "bh", "adjacent", "backside", "infront", "next", "above", "below"}
RELATION_BIGRAM_FIRST = {"in": "front"}  # "in front of"
POSTPOSITIONS = {"paas", "pass", "pas", "samne", "saamne", "peeche", "piche", "pichhe", "bagal", "upar", "niche", "andar", "bahar"}
DISTRICT_CUES = {"dist", "distt", "district", "zila", "zilla", "dt"}
VILLAGE_CUES = {"vill", "village"}
HOUSE_CUES = {"flat", "house", "h", "hno", "plot", "shop", "door", "d", "dno", "room", "office", "unit", "gala", "khasra", "#", "no", "ground", "basement"}
STREET_CUES = {"road", "rd", "street", "marg", "lane", "gali", "galli", "cross", "main", "highway", "nh", "sh", "path", "bypass"}
BUILDING_CUES = {"apartment", "apartments", "apts", "apt", "society", "soc", "complex", "enclave", "residency", "towers", "tower",
                 "heights", "plaza", "mansion", "bhavan", "bhawan", "niwas", "nivas", "villa", "chs", "chsl", "chambers", "arcade",
                 "mall", "building", "bldg", "wing"}
LOCALITY_CUES = {"nagar", "nagr", "colony", "sector", "sec", "layout", "vihar", "puram", "extension", "extn", "ext", "phase", "block",
                 "pada", "wadi", "gaon", "mohalla", "basti", "bazar", "bazaar", "market", "para", "tola", "town", "area", "camp",
                 "chowk", "chawk", "peth", "palya", "halli", "garden", "gardens", "estate"}


def predict(text: str, gaz: Gazetteer) -> list[str]:
    tokens = tokenize(text)
    spans: list[Span] = []
    pin_hit = None
    chunks = _split(tokens, lambda t: t.text == ",")
    for chunk in chunks:
        pins = _pins(chunk)
        spans += pins
        if pins and pin_hit is None:
            pin_hit = gaz.lookup(text[pins[0][0]:pins[0][1]].replace(" ", ""))
    offices = {normalise_value(o) for o in pin_hit["offices"]} if pin_hit else set()
    for chunk in chunks:
        used = {s for s, _, _ in spans}
        rest = [t for t in chunk if not any(s <= t.start < e for s, e, _ in spans)]
        landmarks = _landmarks(rest)
        spans += landmarks
        for run in _split(rest, lambda t: any(s <= t.start < e for s, e, _ in landmarks)):
            spans += _classify(run, text, gaz, offices, pin_hit)
    return spans_to_tags(tokens, sorted(spans))


def _split(tokens: list[Token], is_sep) -> list[list[Token]]:
    runs, cur = [], []
    for t in tokens:
        if is_sep(t):
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(t)
    return runs + ([cur] if cur else [])


def _pins(chunk: list[Token]) -> list[Span]:
    out = []
    i = 0
    while i < len(chunk):
        t = chunk[i]
        if t.text.isdigit() and len(t.text) == 6:
            out.append((t.start, t.end, "PIN")); i += 1
        elif t.text.isdigit() and len(t.text) == 3 and i + 1 < len(chunk) and chunk[i + 1].text.isdigit() and len(chunk[i + 1].text) == 3:
            out.append((t.start, chunk[i + 1].end, "PIN")); i += 2
        else:
            i += 1
    return out


def _landmarks(run: list[Token]) -> list[Span]:
    words = [t.text.lower() for t in run]
    starts = [i for i, w in enumerate(words)
              if w in RELATION or (RELATION_BIGRAM_FIRST.get(w) and i + 1 < len(words) and words[i + 1] == RELATION_BIGRAM_FIRST[w])]
    if starts:
        bounds = starts + [len(run)]
        return [(run[a].start, run[b - 1].end, "LANDMARK") for a, b in zip(bounds, bounds[1:])]
    post = [i for i, w in enumerate(words) if w in POSTPOSITIONS and i > 0 and words[i - 1] == "ke"]
    return [(run[0].start, run[post[-1]].end, "LANDMARK")] if post else []


def _classify(run: list[Token], text: str, gaz: Gazetteer, offices: set[str], pin_hit: dict | None) -> list[Span]:
    run = _strip_punct(run)
    if not run:
        return []
    words = [t.text.lower() for t in run]
    if words[0] in DISTRICT_CUES:
        return _span(_strip_punct(run[1:]), "DISTRICT")
    if words[-1] in DISTRICT_CUES:
        return _span(_strip_punct(run[:-1]), "DISTRICT")
    if words[0] in VILLAGE_CUES:
        return _span(_strip_punct(run[1:]), "LOCALITY")
    joined = text[run[0].start:run[-1].end]  # the original slice, so "Kilepal-2" keeps its hyphen
    if canonical_state(joined) and (len(normalise_value(joined)) > 3 or joined.isupper() or "." in joined):
        return _span(run, "STATE")
    has_digit = any(ch.isdigit() for ch in joined)
    if (words[0] in HOUSE_CUES and has_digit) or "floor" in words or (has_digit and all(len(w) <= 1 or w.isdigit() for w in words)):
        return _span(run, "HOUSE")
    if set(words) & STREET_CUES or words[-1] == "st":
        return _span(run, "STREET")
    if set(words) & BUILDING_CUES:
        return _span(run, "BUILDING")
    key = normalise_value(joined)
    if set(words) & LOCALITY_CUES or key in offices:
        return _span(run, "LOCALITY")
    if key in gaz.cities or key in gaz.districts or (pin_hit and key in map(normalise_value, (pin_hit["city"], pin_hit["district"]))):
        return _span(run, "CITY")
    return []


def _strip_punct(run: list[Token]) -> list[Token]:
    while run and not any(ch.isalnum() for ch in run[0].text):
        run = run[1:]
    while run and not any(ch.isalnum() for ch in run[-1].text):
        if run[-1].text == "." and len(run) > 1 and len(run[-2].text) == 1 and run[-2].text.isalpha():
            break  # the dot of a dotted abbreviation ("C.G.") stays in the span
        run = run[:-1]
    return run


def _span(run: list[Token], label: str) -> list[Span]:
    return [(run[0].start, run[-1].end, label)] if run else []


def run(data_path: str | Path, run_id: str, results_dir: str | Path = "results", gaz: Gazetteer | None = None,
        bootstrap_n: int = 1000) -> dict:
    gaz = gaz or Gazetteer.load()
    records = read_jsonl(data_path)
    pred = [predict(r["text"], gaz) for r in records]
    res = evaluate(records, pred, run_id=run_id, model="rules", data=Path(data_path).stem, seed=0,
                   config={"rules": "v1", "data_path": str(data_path)}, bootstrap_n=bootstrap_n)
    write_results(res, results_dir)
    write_predictions(records, pred, run_id, results_dir)
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    r = run(a.data, a.run_id, a.results_dir)
    print(f"{a.run_id}: n={r['n']} span micro F1={r['span']['micro']['f1']:.3f} macro F1={r['span']['macro']['f1']:.3f} "
          f"exact={r['exact_match']:.3f} CI95={r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}")
