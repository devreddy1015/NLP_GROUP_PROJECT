"""Noise families N1-N8. One function per family: Address in, new Address out, nothing mutated.

Each transformation belongs to exactly one family (build-plan B1); the list lives in initials.md §5.2.
An address is a tuple of Parts (one per field, in order); a Part is labelled chunks joined by `sep`.
"""
import random
import string
from dataclasses import dataclass, replace

from src.generate import lexicon as L
from src.normalize.gazetteer import STATES

Chunk = tuple[str, str | None]


@dataclass(frozen=True)
class Part:
    role: str
    chunks: tuple[Chunk, ...]
    sep: str = " "


@dataclass(frozen=True)
class Address:
    parts: tuple[Part, ...]
    sep: str = ", "


def compose(a: Address) -> tuple[str, list[tuple[int, int, str]]]:
    text, spans = "", []
    for part in a.parts:
        if not part.chunks:
            continue
        if text:
            text += a.sep
        for ci, (chunk, label) in enumerate(part.chunks):
            if ci:
                text += part.sep
            start = len(text)
            text += chunk
            if label:
                spans.append((start, len(text), label))
    return text, spans


# N1: abbreviations of common words, cities and states
ABBREV = {
    "road": ["Rd", "Rd."], "street": ["St", "St."], "apartments": ["Apts", "Apt", "Apts."], "society": ["Soc", "Socy"],
    "complex": ["Cmplx"], "sector": ["Sec", "Sec."], "nagar": ["Ngr"], "colony": ["Col", "Cly"], "opposite": ["Opp", "Opp."],
    "near": ["nr", "nr.", "Nr"], "behind": ["bh", "bhd"], "number": ["No", "No."], "main": ["Mn"], "cross": ["Crs"],
    "floor": ["Flr", "Fl"], "ground": ["Gr", "G"], "building": ["Bldg"], "hospital": ["Hosp"], "station": ["Stn", "Stn."],
    "market": ["Mkt"], "government": ["Govt", "Govt."], "district": ["Dist", "Distt", "Dist.", "Distt."], "house": ["H", "Hse"],
    "office": ["Off", "Ofc"], "industrial": ["Indl", "Ind."], "extension": ["Extn", "Ext"], "layout": ["Lyt"],
    "phase": ["Ph", "Ph."], "block": ["Blk"], "bangalore": ["B'lore", "Blr"], "bengaluru": ["B'luru", "Blr"],
    "hyderabad": ["Hyd"], "secunderabad": ["Sec'bad"], "mumbai": ["Bom", "Mum"], "kolkata": ["Kol", "Cal"], "chennai": ["Chn"],
    "ahmedabad": ["Amd", "A'bad"], "thiruvananthapuram": ["Tvm"], "visakhapatnam": ["Vizag", "Vskp"],
}
# N6: transliteration spelling variants (Hinglish), never abbreviations
SPELLING = {
    "mandir": ["Mandeer", "Mandhir"], "masjid": ["Maszid", "Masjeed"], "gali": ["Galli", "Gully"], "chowk": ["Chauk", "Chouk"],
    "bazar": ["Bazaar", "Bajar"], "colony": ["Kalony", "Coloney"], "marg": ["Maarg"], "para": ["Paara"], "bhavan": ["Bhawan"],
    "niwas": ["Nivas"], "nagar": ["Nager"], "road": ["Rod", "Rode"], "station": ["Steshan"], "hospital": ["Haspatal"],
    "office": ["Offis"], "market": ["Markit"], "paas": ["pass", "pas"], "samne": ["saamne", "samney"], "peeche": ["piche", "pichhe"],
    "hanuman": ["Hanumaan"], "durga": ["Durgaa"], "shiv": ["Shiva"], "krishna": ["Krishn", "Krushna"], "ganga": ["Gangaa"],
}
RELATIONS = ["near", "opposite", "behind", "beside", "next to", "in front of", "adjacent to", "close to", "back side of", "above", "below"]
POSTPOSITIONS = ["ke paas", "ke samne", "ke peeche", "ke bagal me"]
DROPPABLE = ("house", "building", "street", "district", "city", "locality")  # PIN dropping belongs to N7; the place name stays
HOUSE_FORMATS = ["#{n}", "H.No {n}/{m}", "H.No. {n}", "H No {n}", "D.No {n}-{m}-{k}/{L}", "D.No. {n}-{m}", "{n}-{m}-{k}", "No. {n}",
                 "Door No {n}", "Flat No. {n}{L}", "{n}{L}", "Plot No {n}/{m}", "S No {n}", "Khasra No {n}", "{n}/{m}"]


def _idx(parts, role):
    return next((i for i, p in enumerate(parts) if p.role == role), None)


def _map_chunks(a: Address, fn) -> Address:
    return replace(a, parts=tuple(replace(p, chunks=tuple(fn(c) for c in p.chunks)) for p in a.parts))


def _sub_words(a: Address, rng: random.Random, table: dict, p: float) -> Address:
    def fix(chunk: Chunk) -> Chunk:
        text, label = chunk
        out = []
        for w in text.split(" "):
            key = w.rstrip(".").lower()
            if key in table and rng.random() < p:
                v = rng.choice(table[key])
                out.append(v.lower() if w.islower() else v.upper() if w.isupper() and len(w) > 1 else v)
            else:
                out.append(w)
        return " ".join(out), label
    return _map_chunks(a, fix)


def n1_abbrev(a: Address, rng: random.Random) -> Address:
    a = _sub_words(a, rng, ABBREV, p=0.6)

    def state(chunk: Chunk) -> Chunk:
        text, label = chunk
        if label != "STATE" or text not in STATES or rng.random() > 0.6:
            return chunk
        short = [al for al in STATES[text] if len(al) <= 3]
        if not short:
            return chunk
        alias = rng.choice(short)
        return (".".join(alias) + "." if len(alias) == 2 and rng.random() < 0.5 else alias), label
    return _map_chunks(a, state)


def n2_landmark(a: Address, rng: random.Random) -> Address:
    i = _idx(a.parts, "landmark")
    if i is None:
        return a
    r = rng.random()
    count = 0 if r < 0.15 else 2 if r < 0.4 else 1
    anchors = [a.parts[i].chunks[0][0].removeprefix("near "), rng.choice(L.LANDMARKS)][:count]

    def phrase(anchor):
        return f"{rng.choice(RELATIONS)} {anchor}" if rng.random() < 0.7 else f"{anchor} {rng.choice(POSTPOSITIONS)}"
    chunks = tuple((phrase(x), "LANDMARK") for x in anchors)
    parts = list(a.parts)
    if count == 2 and rng.random() < 0.3:
        parts[i] = replace(parts[i], chunks=chunks)  # "behind X near Y" in one breath: two spans, no comma
    else:
        parts[i:i + 1] = [Part("landmark", (c,)) for c in chunks]
    return replace(a, parts=tuple(parts))


def n3_order_drop(a: Address, rng: random.Random) -> Address:
    parts = [p for p in a.parts if not (p.role in DROPPABLE and rng.random() < 0.2)]
    if rng.random() < 0.2:
        parts = [replace(p, chunks=tuple(c for c in p.chunks if c[1] != "STATE")) if p.role == "statepin" else p for p in parts]
    r = rng.random()
    if r < 0.2:
        landmarks = [p for p in parts if p.role == "landmark"]
        parts = [parts[0]] + landmarks + [p for p in parts[1:] if p.role != "landmark"]
    elif r < 0.35:
        _swap(parts, "city", "locality")
    elif r < 0.5:
        _swap(parts, "street", "building")
    elif r < 0.6:
        i, j = _idx(parts, "district"), _idx(parts, "city")
        if i is not None and j is not None:
            parts.insert(j, parts.pop(i))
    elif r < 0.7:
        middle = parts[1:]
        rng.shuffle(middle)
        parts = parts[:1] + middle
    return replace(a, parts=tuple(parts))


def _swap(parts: list, r1: str, r2: str) -> None:
    i, j = _idx(parts, r1), _idx(parts, r2)
    if i is not None and j is not None:
        parts[i], parts[j] = parts[j], parts[i]


def n4_delim_case(a: Address, rng: random.Random) -> Address:
    sep = rng.choice([", ", ", ", " ", " ", " - ", ",", " / "])
    case = rng.choice([None, None, str.lower, str.upper, string.capwords])
    if case:
        a = _map_chunks(a, lambda c: (case(c[0]), c[1]))
    return replace(a, sep=sep)


def n5_typos(a: Address, rng: random.Random) -> Address:
    if rng.random() < 0.5:
        return a
    parts = list(a.parts)
    for _ in range(rng.choice([1, 1, 2])):
        spots = [(pi, ci, wi) for pi, p in enumerate(parts) for ci, (t, l) in enumerate(p.chunks) if l != "PIN"
                 for wi, w in enumerate(t.split(" ")) if len(w) >= 4 and w.isalpha()]
        if not spots:
            break
        pi, ci, wi = rng.choice(spots)
        words = parts[pi].chunks[ci][0].split(" ")
        words[wi] = _edit(words[wi], rng)
        chunks = list(parts[pi].chunks)
        chunks[ci] = (" ".join(words), chunks[ci][1])
        parts[pi] = replace(parts[pi], chunks=tuple(chunks))
    return replace(a, parts=tuple(parts))


def _edit(w: str, rng: random.Random) -> str:
    pos, letter = rng.randrange(len(w)), rng.choice(string.ascii_lowercase)
    op = rng.choice(["delete", "insert", "substitute", "swap"])
    if op == "delete":
        return w[:pos] + w[pos + 1:]
    if op == "insert":
        return w[:pos] + letter + w[pos:]
    if op == "substitute":
        return w[:pos] + letter + w[pos + 1:]
    pos = min(pos, len(w) - 2)
    return w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]


def n6_spelling(a: Address, rng: random.Random) -> Address:
    return _sub_words(a, rng, SPELLING, p=0.7)


def n7_pin(a: Address, rng: random.Random) -> Address:
    si = _idx(a.parts, "statepin")
    if si is None:
        return a
    part = a.parts[si]
    pins = [c for c in part.chunks if c[1] == "PIN"]
    if not pins:
        return a
    pin, others = pins[0][0], tuple(c for c in part.chunks if c[1] != "PIN")
    parts, r = list(a.parts), rng.random()
    ci = _idx(parts, "city")
    if r < 0.2:
        new = others
    elif r < 0.4:
        new = others + ((f"{pin[:3]} {pin[3:]}", "PIN"),)
    elif r < 0.5:
        new = others + ((f"{pin[:3]}-{pin[3:]}", "PIN"),)
    elif r < 0.65:
        new = others + ((rng.choice(["Pin:", "Pin", "Pincode", "PIN Code", "Pin -", "Pincode:"]), None), (pin, "PIN"))
    elif r < 0.75:
        new = others + (("-", None), (pin, "PIN"))
    elif r < 0.85 or ci is None:
        new = ((pin, "PIN"),) + others
    else:  # "Bhilai-23": the last two digits glued to the city
        new = others
        parts[ci] = replace(parts[ci], chunks=parts[ci].chunks + (("-", None), (pin[4:], "PIN")), sep="")
    parts[si] = replace(part, chunks=new)
    return replace(a, parts=tuple(parts))


def n8_numbers(a: Address, rng: random.Random) -> Address:
    hi = _idx(a.parts, "house")
    if hi is None:
        return a
    values = {"n": rng.randint(1, 250), "m": rng.randint(1, 30), "k": rng.randint(1, 99), "L": rng.choice("ABCDE")}
    text = rng.choice(HOUSE_FORMATS).format(**values)
    parts = list(a.parts)
    parts[hi] = replace(parts[hi], chunks=((text, "HOUSE"),))
    return replace(a, parts=tuple(parts))
