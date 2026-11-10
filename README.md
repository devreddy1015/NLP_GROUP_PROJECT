# Indian Address Parser

Group 10 project for DSL504 / DS504 Natural Language Processing, IIT Bhilai (Jul–Nov 2026).
Agastya Nath, Y. Rahul Dev Reddy, Paritosh Lahre, Varun Rao.

We tag the parts of messy Indian addresses (house, building, street, landmark, locality, city, district, state, PIN) and check them against the India Post PIN directory.

    flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490023

**Research question:** parsers trained on clean synthetic addresses do badly on real, human-written ones. We measure that gap and test which kinds of synthetic noise, and how many real examples, close it.

**Status:** all code is built and tested. No real addresses are collected yet, so every score so far is on synthetic data or a 20-address smoke set.

## Setup

```bash
uv sync --extra train --extra demo
uv run pytest
```

## Main commands

```bash
uv run python -m src.generate --config full --seed 42      # synthetic data
uv run python -m app.demo --model rules                    # demo page
scripts/run_experiments.sh --dry-run                       # list the planned experiment runs
uv run python -m scripts.make_tables                       # report tables from results/
```

## Where things are

| Path | Contents |
|---|---|
| `src/` | generator, models (rules, CRF, MuRIL / XLM-R, existing parsers, LLM), normaliser, evaluation |
| `tests/` | tests |
| `results/`, `report/tables/` | one file per run, and tables built from them |
| `initials.md`, `build-plan.md`, `logs.md` | the plan, the build order, and the work log |
| `research.md`, `resources.md` | prior work, and every external link with its licence |

## Rules

1. **Privacy:** addresses of public places only. No home addresses, names or phone numbers.
2. **Frozen test set:** never train, tune or adjust prompts on the real test set.
3. **Real numbers only:** every reported number comes from a file in `results/`.

Data: India Post PIN directory (GODL-India).
