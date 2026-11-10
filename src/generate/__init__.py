"""Synthetic address generator v1: fixed-order clean template over real geography (PIN directory).

`clean` = all nine fields, one fixed order, commas, full words, one plain "near X" landmark, prefixed by a
public-place name tagged O (build-plan A2, B2). Noise families N1-N8 arrive as config switches in step 8.
"""
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.data import Span, make_record, write_jsonl
from src.generate import lexicon as L
from src.normalize.gazetteer import CLEAN_PATH


@dataclass(frozen=True)
class Config:
    name: str
    # noise switches N1..N8 are added in step 8


CONFIGS = {"clean": Config("clean")}

_OFFICE_OK = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,40}$")
_SPLIT_SEED = 0  # the PIN split never changes with config or seed, so held-out test PINs stay held out


def split_pins(pins) -> dict[str, list[str]]:
    """90 / 5 / 5 split of PINs into train / dev / test, fixed across configs and seeds."""
    pins = sorted(set(pins))
    random.Random(_SPLIT_SEED).shuffle(pins)
    n_test = n_dev = max(1, round(0.05 * len(pins)))
    return {"test": sorted(pins[:n_test]), "dev": sorted(pins[n_test:n_test + n_dev]),
            "train": sorted(pins[n_test + n_dev:])}


def compose(parts: list[list[tuple[str, str | None]]], part_sep: str = ", ", chunk_sep: str = " ") -> tuple[str, list[Span]]:
    """Join labelled chunks into text and character spans. Parts join with `part_sep`, chunks with `chunk_sep`."""
    text, spans = "", []
    for pi, part in enumerate(parts):
        if pi:
            text += part_sep
        for ci, (chunk, label) in enumerate(part):
            if ci:
                text += chunk_sep
            start = len(text)
            text += chunk
            if label:
                spans.append((start, len(text), label))
    return text, spans


class Generator:
    def __init__(self, gaz: pd.DataFrame, config: Config, seed: int):
        self.config, self.seed = config, seed
        splits = split_pins(gaz.pin.unique())  # over every PIN, so the office filter below cannot move a PIN between splits
        usable = gaz[gaz.office.str.match(_OFFICE_OK) & ~gaz.office.str.startswith("NDC")]
        self.rows = {split: usable[usable.pin.isin(pins)].to_dict("records") for split, pins in splits.items()}
        empty = [split for split, rows in self.rows.items() if not rows]
        if empty:
            raise ValueError(f"no usable offices in split(s) {empty}; the gazetteer is too small")

    def record(self, split: str, index: int) -> dict:
        rng = random.Random(f"{self.seed}:{split}:{index}")  # order-independent: record i is fixed by (seed, split, i)
        row = rng.choice(self.rows[split])
        parts = [
            [(rng.choice(L.PLACE_NAMES), None)],
            [(L.house(rng), "HOUSE")],
            [(L.building(rng), "BUILDING")],
            [(L.street(rng), "STREET")],
            [(f"near {rng.choice(L.LANDMARKS)}", "LANDMARK")],
            [(row["office"], "LOCALITY")],
            [(row["city"], "CITY")],
            [("District", None), (row["district"], "DISTRICT")],
            [(row["state"], "STATE"), (row["pin"], "PIN")],
        ]
        text, spans = compose(parts)
        meta = {k: row[k] for k in ("pin", "office", "city", "district", "state")}
        return make_record(f"syn-{self.config.name}-{split}-{index:06d}", text, spans, "synthetic", split, meta=meta)


def generate(config: str, seed: int, n_train: int, n_dev: int, n_test: int, out_dir: str | Path,
             gaz_path: str | Path = CLEAN_PATH) -> dict[str, int]:
    cfg = CONFIGS[config]
    gaz = pd.read_csv(gaz_path, dtype=str, keep_default_na=False)
    gen = Generator(gaz, cfg, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        write_jsonl(out / f"{split}.jsonl", (gen.record(split, i) for i in range(n)))
        counts[split] = n
    (out / "config.json").write_text(json.dumps({"config": config, "seed": seed, "counts": counts,
                                                 "switches": asdict(cfg), "gazetteer_rows": len(gaz),
                                                 "gazetteer_sha256": hashlib.sha256(Path(gaz_path).read_bytes()).hexdigest()}, indent=1))
    return counts
