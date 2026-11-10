"""Shared data definitions: labels, the one tokenizer, spans <-> BIO, JSONL I/O with validation.

Every module (generator, annotation export, CRF, transformer alignment, eval) must go through
`tokenize()` so that token boundaries agree everywhere (build-plan A3).
"""
import json
import unicodedata
from pathlib import Path
from typing import Iterable, NamedTuple

LABELS = ("HOUSE", "BUILDING", "STREET", "LANDMARK", "LOCALITY", "CITY", "DISTRICT", "STATE", "PIN")
TAGS = ("O",) + tuple(f"{p}-{label}" for label in LABELS for p in ("B", "I"))
SOURCES = ("synthetic", "real")
SPLITS = ("train", "dev", "test")

Span = tuple[int, int, str]  # character offsets [start, end) and label


class Token(NamedTuple):
    text: str
    start: int
    end: int


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def tokenize(text: str) -> list[Token]:
    """Split on whitespace; every punctuation or symbol character is its own token.

    Combining marks (Devanagari matras) stay attached to their word.
    """
    tokens, start = [], None
    for i, ch in enumerate(text):
        if ch.isspace() or _is_punct(ch):
            if start is not None:
                tokens.append(Token(text[start:i], start, i))
                start = None
            if _is_punct(ch):
                tokens.append(Token(ch, i, i + 1))
        elif start is None:
            start = i
    if start is not None:
        tokens.append(Token(text[start:], start, len(text)))
    return tokens


def spans_to_tags(tokens: list[Token], spans: Iterable[Span]) -> list[str]:
    """IOB2 tags for `tokens`. Spans must start and end on token boundaries and not overlap."""
    tags = ["O"] * len(tokens)
    starts = {t.start: i for i, t in enumerate(tokens)}
    ends = {t.end: i for i, t in enumerate(tokens)}
    for start, end, label in spans:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        if start not in starts or end not in ends or end <= start:
            raise ValueError(f"span {(start, end, label)} is not on token boundaries")
        first, last = starts[start], ends[end]
        if any(tags[i] != "O" for i in range(first, last + 1)):
            raise ValueError(f"span {(start, end, label)} overlaps another span")
        tags[first] = f"B-{label}"
        for i in range(first + 1, last + 1):
            tags[i] = f"I-{label}"
    return tags


def tags_to_spans(tokens: list[Token], tags: list[str]) -> list[Span]:
    """Character spans from IOB2 tags. A dangling I- (no B-) opens a new span, like seqeval's lenient mode."""
    spans: list[Span] = []
    prev_label = None
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            prev_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "I" and label == prev_label:
            spans[-1] = (spans[-1][0], tok.end, label)
        else:
            spans.append((tok.start, tok.end, label))
        prev_label = label
    return spans


def normalise_value(s: str) -> str:
    """Lower-case, drop punctuation and symbols, collapse whitespace (build-plan B9). Used for every string match."""
    cleaned = "".join(ch for ch in s.lower() if unicodedata.category(ch)[0] not in "PS")
    return " ".join(cleaned.split())


def label_substrings(text: str, pairs: Iterable[tuple[str, str]]) -> list[Span]:
    """Spans from (substring, label) pairs, matched left to right. For hand-labelled examples."""
    spans, cursor = [], 0
    for sub, label in pairs:
        start = text.find(sub, cursor)
        if start < 0:
            raise ValueError(f"{sub!r} not found after offset {cursor} in {text!r}")
        spans.append((start, start + len(sub), label))
        cursor = start + len(sub)
    return spans


def make_record(id: str, text: str, spans: Iterable[Span], source: str, split: str, noise: Iterable[str] = (),
                slices: Iterable[str] = (), meta: dict | None = None) -> dict:
    tokens = tokenize(text)
    rec = {"id": id, "text": text, "tokens": [t.text for t in tokens], "tags": spans_to_tags(tokens, spans),
           "source": source, "split": split, "noise": list(noise), "slices": list(slices), "meta": meta or {}}
    validate(rec)
    return rec


def validate(rec: dict) -> None:
    """Raise ValueError unless `rec` follows the CLAUDE.md JSONL format."""
    for key, typ in (("id", str), ("text", str), ("tokens", list), ("tags", list), ("source", str), ("split", str)):
        if not isinstance(rec.get(key), typ):
            raise ValueError(f"{rec.get('id')}: field {key!r} missing or not {typ.__name__}")
    if not rec["text"].strip():
        raise ValueError(f"{rec['id']}: empty text")
    if rec["tokens"] != [t.text for t in tokenize(rec["text"])]:
        raise ValueError(f"{rec['id']}: tokens do not match tokenize(text)")
    if len(rec["tags"]) != len(rec["tokens"]):
        raise ValueError(f"{rec['id']}: {len(rec['tags'])} tags for {len(rec['tokens'])} tokens")
    prev = "O"
    for tag in rec["tags"]:
        if tag not in TAGS:
            raise ValueError(f"{rec['id']}: unknown tag {tag!r}")
        if tag.startswith("I-") and prev[2:] != tag[2:]:
            raise ValueError(f"{rec['id']}: {tag} does not follow B-/I-{tag[2:]}")
        prev = tag
    if rec["source"] not in SOURCES or rec["split"] not in SPLITS:
        raise ValueError(f"{rec['id']}: source {rec['source']!r} / split {rec['split']!r} not allowed")
    for key in ("noise", "slices"):
        if not all(isinstance(x, str) for x in rec.get(key, [])):
            raise ValueError(f"{rec['id']}: {key} must be a list of strings")
    if not isinstance(rec.get("meta", {}), dict):
        raise ValueError(f"{rec['id']}: meta must be an object")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:  # not splitlines(): it also splits on U+2028/U+0085, which json.dumps leaves unescaped
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        validate(rec)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    records = list(records)
    for rec in records:
        validate(rec)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
