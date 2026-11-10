# CLAUDE.md: Indian Address Parser (Group 10 NLP project)

Instructions for Claude Code and for teammates working in this folder. Keep this file short; the details live in the other docs.

## The project in brief
Course project for DSL504 / DS504 Natural Language Processing (IIT Bhilai, Jul–Nov 2026), Group 10:
Agastya Nath, Y. Rahul Dev Reddy, Paritosh Lahre, Varun Rao.

We tag the parts of messy Indian addresses (house, building, street, landmark, locality, city, district, state, PIN) and normalise them against the India Post PIN directory.

    flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490023

**Research focus:** parsers trained on clean synthetic addresses collapse on real, human-written ones. We measure that gap on a new human-written test set and test which kinds of synthetic noise, and how many real examples, close it.

## Files in this folder
| File | What it holds | Update it when |
|---|---|---|
| `CLAUDE.md` | Rules and conventions (this file) | A convention changes |
| `initials.md` | The plan: problem, research questions, label schema, data, models, experiments, timeline, open decisions, decision log | A plan or decision changes (also add a row to its §13 decision log) |
| `research.md` | Prior work, existing models and datasets, how we differ | You find relevant new work |
| `resources.md` | Every external link (datasets, models, papers, tools) with licence and date checked | You use or cite any new link |
| `logs.md` | Dated work log + table of every training/eval run | End of every work session, and after every run |
| `build-plan.md` | Cross-check of the plan, decisions A1–B15, and the build order (steps 1–16) | A step finishes or the order changes |
| `data/real/GUIDELINES.md` | Annotation rules and edge cases | An edge case is decided |

**Before any task:** read `logs.md` (latest status), then `initials.md` and `research.md`.

## Status (10 Nov 2026, evening)
Steps 1–9 and 13 of `build-plan.md` §3.2 are built: data format and tokenizer, eval, gazetteer, generator with noise families N1–N8 (`clean`, `full`, `full-minus-Nx` configs), rules and CRF baselines, normaliser (E5), annotation plumbing (Label Studio config, import/export, agreement), 20 hand-labelled smoke items, guidelines v1. Training extras (torch 2.15 + CUDA, transformers 5.22, sklearn-crfsuite) are installed locally. No real data yet. **Blocked on the team:** the deadline (`initials.md` header) and sending the Google Form (with a respondent code). Next code: step 10 (MuRIL / XLM-R), step 11 (zero-shot E1).

## Ground rules
1. **Privacy.** Never collect, store or commit real residential addresses, people's names or phone numbers. Real data = addresses of public places only (shops, temples, schools, hospitals, offices). The generator does not need fake names or phones either; leave them out.
2. **Frozen test set.** Once the real test set is annotated, never train, tune or adjust prompts on it. Make every decision on the real dev split. Run the test set only for final runs.
3. **Real numbers only.** Every number in the report or slides must come from a file in `results/` and a row in the `logs.md` run table. Never estimate, round up or invent a result. If a run fails, log it as failed.
4. **Headline = real test set.** Synthetic test scores are near 99% and say little (`research.md` §1). Report them only to show the gap.
5. **Reproducibility.** Fix seeds; run the main models with 3 seeds and report mean ± std; save the config next to each result.
6. **Licences.** Attribute India Post data (GODL-India) and OpenStreetMap (ODbL). Check a model's or dataset's licence before redistributing it.
7. **Ask first** before downloading more than 2 GB, starting a job longer than 1 hour, or using a paid API.
8. **Keep the docs current.** After each session: add an entry to `logs.md`; add new links to `resources.md`; record changed decisions in `initials.md` §13. Cite sources in `research.md` only after opening them.

## Conventions
### Data format (JSONL, one address per line)
```json
{"id": "syn-clean-train-000042", "text": "flat 3b, shanti apts, ...", "tokens": ["flat", "3b", ","],
 "tags": ["B-HOUSE", "I-HOUSE", "O"], "source": "synthetic", "split": "train", "noise": ["N1", "N2"],
 "slices": [], "meta": {"pin": "490023", "city": "Bhilai", "district": "Durg", "state": "Chhattisgarh"}}
```
- `source` is `synthetic` or `real`; `split` is `train`, `dev` or `test`.
- `noise` lists the noise families applied (`N1`…`N8`); `slices` lists test-item traits (`data/real/GUIDELINES.md`); `meta` holds the known true PIN / city / district / state (from the form or the generator).
- Tokens come from `src.data.tokenize` only: split on whitespace, every punctuation or symbol character is its own token (`b'lore-68` → `b ' lore - 68`). `src.data.validate` enforces this on every read and write.
- Tags are BIO (IOB2) over word and punctuation tokens. Transformer code aligns them to subwords: label the first subword and ignore the rest (`-100`).
- Labels: `HOUSE BUILDING STREET LANDMARK LOCALITY CITY DISTRICT STATE PIN`. Definitions and edge cases are in `initials.md` §4.

### Folder layout (create folders as needed)
```
data/raw/          downloads (gitignored)
data/gazetteer/    cleaned PIN directory and place-name lists
data/synthetic/    generated train/dev JSONL, one folder per generator config
data/real/         human-written dev/test JSONL + GUIDELINES.md (annotation rules)
src/generate/      generator; one function per noise family (N1–N8)
src/models/        rules, crf, transformer, llm
src/normalize/     PIN / gazetteer normalisation
src/eval/          span F1 (seqeval), field accuracy, bootstrap CIs
notebooks/         Colab/Kaggle training notebooks
results/           one JSON per run (committed)
app/               demo (Gradio or Streamlit)
report/            report and slides
```

### Stack
Python 3.12 through `uv` (`uv sync` builds `.venv`; matches Colab). Core deps are pinned in `uv.lock` and exported to `requirements.txt`; training extras (`torch`, `transformers`, `datasets`, `sklearn-crfsuite`) install with `uv sync --extra train` and get pinned once first used. Main backbones: `google/muril-base-cased` and `FacebookAI/xlm-roberta-base`. Local GPU: RTX 4060 Laptop (8 GB); otherwise a free Colab/Kaggle T4.

### Run IDs
Use `YYYYMMDD-<model>-<data>-s<seed>`, e.g. `20261114-muril-synfull-s42`. The results file (`results/<run-id>.json`) and the `logs.md` row use the same ID.

## Commands
```bash
uv sync                                   # environment (add --extra train for torch/transformers)
uv run pytest                             # tests + coverage (src/)
uv run python -m src.normalize.gazetteer  # download pincode.csv -> data/gazetteer/pincodes.csv
uv run python -m src.generate --config full --seed 42         # 20k/2k/2k -> data/synthetic/full/ (configs: clean, full, full-minus-N1..N8)
uv run python -m src.models.rules --data <jsonl> --run-id YYYYMMDD-rules-<data>-s0   # -> results/<run-id>.json
uv run python -m src.models.crf --train data/synthetic/full/train.jsonl --eval <jsonl> --run-id YYYYMMDD-crf-synfull-<data>-s0 [--n-train N]
uv run python -m src.normalize.normalize --data <jsonl> --run-id YYYYMMDD-normalizer-<data>-s0   # E5 with gold tags
uv run python -m src.annotate tasks --csv form.csv --out data/real          # form CSV -> Label Studio tasks per split
uv run python -m src.annotate export --export ls.json --out data/real/dev.jsonl [--annotator N]
uv run python -m src.annotate agreement a.jsonl b.jsonl                      # span F1 + token kappa
```
Module map: `src/data.py` (labels, tokenizer, spans↔BIO, JSONL), `src/eval/` (seqeval strict F1, field P/R/F1, bootstrap CI, slices, results writer), `src/normalize/gazetteer.py`, `src/generate/` (templates, lexicon, `noise.py` N1–N8, CLI), `src/models/rules.py` (B0), `src/models/crf.py` (B1), `src/normalize/normalize.py` (fields → JSON, PIN checks, E5), `src/annotate.py` (Label Studio in/out, agreement).
