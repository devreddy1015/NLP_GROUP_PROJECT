"""Synthetic address generator: a fixed-order clean template over real geography (PIN directory), plus
noise families N1-N8 as config switches (`src/generate/noise.py`).

`clean` = all nine fields, one fixed order, commas, full words, one plain "near X" landmark, prefixed by a
public-place name tagged O (build-plan A2, B2). `full` = every family on; `full-minus-Nx` for the E3 ablation.
"""
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data import make_record, write_jsonl
from src.generate import lexicon as L
from src.generate import noise
from src.generate.noise import Address, Part, compose
from src.normalize.gazetteer import CLEAN_PATH

FAMILIES = {"N1": noise.n1_abbrev, "N2": noise.n2_landmark, "N3": noise.n3_order_drop, "N4": noise.n4_delim_case,
            "N5": noise.n5_typos, "N6": noise.n6_spelling, "N7": noise.n7_pin, "N8": noise.n8_numbers}
ORDER = ("N2", "N3", "N8", "N7", "N1", "N6", "N5", "N4")  # structure first, then wording, typos, and case/delimiters last


@dataclass(frozen=True)
class Config:
    name: str
    on: frozenset[str] = frozenset()  # family ids switched on

    def switches(self) -> dict[str, bool]:
        return {fid: fid in self.on for fid in FAMILIES}


_ALL = frozenset(FAMILIES)
CONFIGS = {"clean": Config("clean"), "full": Config("full", _ALL)}
CONFIGS |= {f"full-minus-{f}": Config(f"full-minus-{f}", _ALL - {f}) for f in FAMILIES}
CONFIGS |= {f"only-{f}": Config(f"only-{f}", frozenset({f})) for f in FAMILIES}

_OFFICE_OK = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,40}$")
_SPLIT_SEED = 0  # the PIN split never changes with config or seed, so held-out test PINs stay held out


def split_pins(pins) -> dict[str, list[str]]:
    """90 / 5 / 5 split of PINs into train / dev / test, fixed across configs and seeds."""
    pins = sorted(set(pins))
    random.Random(_SPLIT_SEED).shuffle(pins)
    n_test = n_dev = max(1, round(0.05 * len(pins)))
    return {"test": sorted(pins[:n_test]), "dev": sorted(pins[n_test:n_test + n_dev]),
            "train": sorted(pins[n_test + n_dev:])}


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
        street_stem = rng.choice(self.rows[split])["office"].split()[0]  # a real place name: "Junwani Road"
        locality = row["office"].removesuffix(" " + row["city"])  # "Supela Bhilai" in Bhilai is written "Supela, Bhilai"
        addr = Address((
            Part("place", ((rng.choice(L.PLACE_NAMES), None),)),
            Part("house", ((L.house(rng), "HOUSE"),)),
            Part("building", ((L.building(rng), "BUILDING"),)),
            Part("street", ((L.street(rng, street_stem), "STREET"),)),
            Part("landmark", ((f"near {rng.choice(L.LANDMARKS)}", "LANDMARK"),)),
            Part("locality", ((locality, "LOCALITY"),)),
            Part("city", ((row["city"], "CITY"),)),
            Part("district", (("District", None), (row["district"], "DISTRICT"))),
            Part("statepin", ((row["state"], "STATE"), (row["pin"], "PIN"))),
        ))
        applied = []
        for fid in ORDER:
            if fid in self.config.on:
                noised = FAMILIES[fid](addr, rng)
                if noised != addr:
                    applied.append(fid)
                    addr = noised
        text, spans = compose(addr)
        meta = {k: row[k] for k in ("pin", "office", "city", "district", "state")}
        return make_record(f"syn-{self.config.name}-{split}-{index:06d}", text, spans, "synthetic", split,
                           noise=sorted(applied), meta=meta)


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
                                                 "switches": cfg.switches(), "gazetteer_rows": len(gaz),
                                                 "gazetteer_sha256": hashlib.sha256(Path(gaz_path).read_bytes()).hexdigest()}, indent=1))
    return counts
