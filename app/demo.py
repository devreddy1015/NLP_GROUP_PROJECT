"""Demo: type an address, see its parts, the cleaned JSON and what the PIN directory says (build-plan step 15).

    uv sync --extra train --extra demo
    uv run python -m app.demo                        # CRF, trained on the full-noise synthetic set at start (about a minute)
    uv run python -m app.demo --model rules          # the rules baseline: starts at once
    uv run python -m app.demo --model models/muril   # a transformer saved with `src.models.transformer --save-dir models/muril`

Nothing typed into the page is stored or sent anywhere. It still runs on one machine for a talk:
use addresses of public places, as everywhere else in this project (CLAUDE.md rule 1).
"""
import argparse
from pathlib import Path
from typing import Callable

from src.data import read_jsonl, tags_to_spans, tokenize
from src.models import crf, rules
from src.normalize.gazetteer import Gazetteer
from src.normalize.normalize import normalise

EXAMPLE = "flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490023"  # initials.md §2
EXAMPLES = [EXAMPLE, "Apollo Pharmacy, Ground Floor, Surya Towers, Opp Bus Stand, Station Road, Raipur 492001 (C.G.)",
            "opp. ganesh temple, 2nd cross, hosur road, madivala, b'lore-68", "Shop 4, MG Road, Pune 999999"]
TRAIN = Path("data/synthetic/full/train.jsonl")
Tagger = Callable[[str], list[str]]  # address text -> one IOB2 tag per token of src.data.tokenize


def _tokens(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]


def rules_tagger(gaz: Gazetteer) -> Tagger:
    return lambda text: rules.predict(text, gaz)


def crf_tagger(gaz: Gazetteer, train_path=TRAIN, max_iter: int = 100) -> Tagger:
    lex = crf.Lexicon(gaz)
    model = crf.train(read_jsonl(train_path), lex, max_iter=max_iter)
    return lambda text: crf.predict(model, [{"tokens": _tokens(text)}], lex)[0]


def transformer_tagger(folder, device: str | None = None) -> Tagger:
    import torch  # heavy: only loaded when a transformer is asked for
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.models.transformer import predict
    tok = AutoTokenizer.from_pretrained(folder)
    net = AutoModelForTokenClassification.from_pretrained(folder).to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return lambda text: predict(net, tok, [{"tokens": _tokens(text)}])[0]


def _notes(parsed: dict) -> list[str]:
    """What the PIN directory says about the address, in plain words."""
    notes = {"missing": ["No PIN in this address."],
             "not_found": [f"PIN {parsed['pin']} is not in the India Post directory."],
             "mismatch": [f"PIN {parsed['pin']} belongs to a different state or district than the one written."]}.get(parsed["pin_check"], [])
    if parsed["filled"]:
        notes.append(f"{' and '.join(parsed['filled']).capitalize()} filled in from the PIN: "
                     f"{', '.join(parsed[field] for field in parsed['filled'])}.")
    if parsed["locality_known"] is False:
        notes.append("The locality is not a post-office name under this PIN. That is common and not an error.")
    return notes


def parse(text: str, tagger: Tagger, gaz: Gazetteer) -> dict:
    """spans: the whole text in order as (piece, label or None); json: the normaliser's output; notes: the PIN findings."""
    if not text.strip():
        return {"spans": [], "json": {}, "notes": ["Type an address first."]}
    tags = tagger(text)
    spans, at = [], 0
    for start, end, label in tags_to_spans(tokenize(text), tags):
        spans += [(text[at:start], None)] * (start > at) + [(text[start:end], label)]
        at = end
    spans += [(text[at:], None)] * (at < len(text))
    parsed = normalise(text, tags, gaz)
    return {"spans": spans, "json": parsed, "notes": _notes(parsed)}


def main() -> None:
    import gradio as gr  # the `demo` extra; not needed by anything else

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="crf", help="crf, rules, or a folder saved by src.models.transformer --save-dir")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    gaz = Gazetteer.load()
    tagger = {"crf": crf_tagger, "rules": rules_tagger}[a.model](gaz) if a.model in ("crf", "rules") else transformer_tagger(a.model)

    def respond(text: str):
        out = parse(text, tagger, gaz)
        return out["spans"], out["json"], "\n".join(f"- {note}" for note in out["notes"]) or "The PIN is in the directory and nothing contradicts it."

    gr.Interface(
        fn=respond, inputs=gr.Textbox(label="Address", lines=2, placeholder=EXAMPLE),
        outputs=[gr.HighlightedText(label="Parts"), gr.JSON(label="Cleaned"), gr.Markdown(label="PIN check")],
        examples=EXAMPLES, title="Indian address parser",
        description=f"Tags the parts of a messy Indian address and checks it against the India Post PIN directory. Model: {a.model}.",
        flagging_mode="never", analytics_enabled=False,  # nothing a visitor types is kept or reported
    ).launch(server_name="127.0.0.1", server_port=a.port)


if __name__ == "__main__":
    main()
