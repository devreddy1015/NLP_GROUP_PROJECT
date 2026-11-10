"""India Post PIN directory as a gazetteer: download, clean, and PIN -> offices / city / district / state.

Source: All India Pincode Directory (GODL-India) via the dropdevrahul/pincodes-india GitHub mirror.
Build or rebuild with `uv run python -m src.normalize.gazetteer`.
"""
import re
import string
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from src.data import normalise_value

RAW_URL = "https://raw.githubusercontent.com/dropdevrahul/pincodes-india/d817d3e652e8167a70507350d9a8e4acda9113ef/pincode.csv"  # pinned commit (main on 10 Nov 2026)
RAW_PATH = Path("data/raw/pincode.csv")
CLEAN_PATH = Path("data/gazetteer/pincodes.csv")

# Canonical name -> aliases (abbreviations, old names, India Post spellings). 36 states and UTs.
STATES: dict[str, tuple[str, ...]] = {
    "Andhra Pradesh": ("AP",), "Arunachal Pradesh": ("AR",), "Assam": ("AS",), "Bihar": ("BR",),
    "Chhattisgarh": ("CG", "Chattisgarh"), "Goa": ("GA",), "Gujarat": ("GJ", "Guj"), "Haryana": ("HR",),
    "Himachal Pradesh": ("HP",), "Jharkhand": ("JH",), "Karnataka": ("KA", "Ktk"), "Kerala": ("KL",),
    "Madhya Pradesh": ("MP",), "Maharashtra": ("MH", "Maha"), "Manipur": ("MN",), "Meghalaya": ("ML",),
    "Mizoram": ("MZ",), "Nagaland": ("NL",), "Odisha": ("OD", "OR", "Orissa"), "Punjab": ("PB",),
    "Rajasthan": ("RJ", "Raj"), "Sikkim": ("SK",), "Tamil Nadu": ("TN", "Tamilnadu"), "Telangana": ("TS", "TG"),
    "Tripura": ("TR",), "Uttar Pradesh": ("UP",), "Uttarakhand": ("UK", "UA", "Uttaranchal"),
    "West Bengal": ("WB",), "Andaman and Nicobar Islands": ("AN", "Andaman & Nicobar", "Andaman and Nicobar"),
    "Chandigarh": ("CH",),
    "Dadra and Nagar Haveli and Daman and Diu": ("DN", "DD", "The Dadra and Nagar Haveli and Daman and Diu",
                                                 "Dadra and Nagar Haveli", "Daman and Diu"),
    "Delhi": ("DL", "New Delhi", "NCT of Delhi"), "Jammu and Kashmir": ("JK", "J&K"), "Ladakh": ("LA",),
    "Lakshadweep": ("LD",), "Puducherry": ("PY", "Pondicherry"),
}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ]", "", s.lower().replace("&", "and"))


_STATE_INDEX = {_key(name): name for name in STATES} | {_key(a): name for name, al in STATES.items() for a in al}


def canonical_state(s: str) -> str | None:
    return _STATE_INDEX.get(_key(s)) if s else None


_OFFICE_TYPE = r"(?:G\.?P\.?O|B\.?O|S\.?O|H\.?O|P\.?O)\.?"


def clean_office_name(name: str) -> str:
    n = re.sub(r"\(.*?\)|\[.*?\]", " ", name)
    n = re.sub(rf"\s+{_OFFICE_TYPE}\s*$", "", n, flags=re.I)
    n = re.sub(r"(?<=[a-z])(?:BO|SO|HO)$", "", n)  # glued: "Abhiya bazarBO"
    n = re.sub(r"\s+", " ", n).strip(" ,.-")
    return _title(n)


def clean_city(division: str, district: str) -> str:
    # shortcut: the division name stands in for the city (build-plan A4); OSM names replace it for chosen cities later
    c = re.sub(r"\b(?:Divisional Office|Division|H\.?O|G\.?P\.?O)\b\.?", " ", division, flags=re.I)
    c = re.sub(r"\s+", " ", c).strip()
    c = re.sub(r"(?: (?:City|East|West|North|South|Central|Moffusil))+$", "", c, flags=re.I)  # "Mumbai City North West" -> "Mumbai"
    return _title(c if c and c != "NA" else district)


_TOWN_MIN_OFFICES, _TOWN_MAX_DISTRICTS = 3, 3


def _towns(clean: pd.DataFrame) -> pd.Series:
    """Rows whose office name ends in a town ("Supela Bhilai", "Kohka Bhilai") -> that town, to use as the city.

    shortcut: read off the directory's own names (build-plan A4); OSM place names replace it for the chosen cities.
    A last word is a town when several offices of one district end in it, it is not a suffix found all over the
    country ("Khurd", "Nagar"), and the district has an office named after it ("Bhilai East").
    """
    words = clean.office.str.split()
    last = words.str[-1].where((words.str.len() >= 2) & words.str[-1].str.fullmatch(r"[A-Za-z]{4,}", na=False))
    ends = clean.assign(last=last).dropna(subset="last")
    here = ends.groupby(["district", "last"]).office.transform("nunique")
    spread = ends.groupby("last").district.transform("nunique")
    heads = set(zip(clean.district, words.str[0]))
    named = pd.Series([key in heads for key in zip(ends.district, ends["last"])], index=ends.index)
    return ends["last"][(here >= _TOWN_MIN_OFFICES) & (spread <= _TOWN_MAX_DISTRICTS) & named]


def _title(s: str) -> str:
    return re.sub(r"\bAnd\b", "and", string.capwords(s)) if s.isupper() else s  # capwords keeps "Mary's" and "1st"


def download(path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RAW_URL} -> {path}", file=sys.stderr)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(RAW_URL, tmp)
    tmp.replace(path)  # never leave a half-written file behind
    return path


def build(raw: Path = RAW_PATH, out: Path = CLEAN_PATH) -> pd.DataFrame:
    if not Path(raw).exists():
        download(raw)
    df = pd.read_csv(raw, dtype=str, keep_default_na=False)
    states = df.StateName.map(canonical_state)
    if states.isna().any():
        raise ValueError(f"unknown states: {sorted(df.StateName[states.isna()].unique())}")
    clean = pd.DataFrame({
        "pin": df.Pincode, "office": df.OfficeName.map(clean_office_name), "office_type": df.OfficeType,
        "delivery": df.Delivery, "city": [clean_city(d, s) for d, s in zip(df.DivisionName, df.District)],
        "district": df.District.map(_title), "state": states, "lat": df.Latitude, "lon": df.Longitude,
    })
    clean = clean[clean.pin.str.fullmatch(r"\d{6}")].copy()
    towns = _towns(clean)
    clean.loc[towns.index, "city"] = towns
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    return clean


class Gazetteer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._by_pin = {pin: g for pin, g in df.groupby("pin", sort=False)}
        # keys use normalise_value so they match the same transform rules.py applies to the text
        self.states = {normalise_value(s) for s in df.state.unique()}
        self.districts = {normalise_value(d) for d in df.district.unique()}
        self.cities = {normalise_value(c) for c in df.city.unique()}
        self.offices = {normalise_value(o) for o in df.office.unique() if o}

    @classmethod
    def load(cls, path: Path = CLEAN_PATH) -> "Gazetteer":
        if not Path(path).exists():
            build(out=path)
        return cls(pd.read_csv(path, dtype=str, keep_default_na=False))

    def lookup(self, pin: str) -> dict | None:
        g = self._by_pin.get(pin)
        # civil PINs start with 1 to 8; the directory also holds a "Test Office" at 999999 and two army base post offices
        if g is None or pin[0] not in "12345678":
            return None
        # one PIN in 17 covers offices in more than one district: `district` is the commonest, `districts` is all of them
        return {"pin": pin, "offices": g.office.tolist(), "city": g.city.mode()[0], "district": g.district.mode()[0],
                "districts": sorted(g.district.unique()), "state": g.state.mode()[0]}


if __name__ == "__main__":
    df = build()
    print(f"wrote {CLEAN_PATH}: {len(df)} offices, {df.pin.nunique()} PINs")
