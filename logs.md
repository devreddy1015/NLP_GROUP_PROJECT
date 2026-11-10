# Project log

Newest entries first. Add one entry per work session, and one row per training or evaluation run in the run table at the bottom. Keep entries short and factual.

**Entry template**
```
### YYYY-MM-DD: <who>
- Did: …
- Decided: … (also add a row to initials.md §13)
- Results: run IDs from the table below / files in results/
- Problems: …
- Next: …
```

---

### 2026-11-10 (late evening): Claude Code, steps 9 and 13 closed out
- **Did:** Steps 9 (`src/models/crf.py`, B1) and 13 (`src/normalize/normalize.py`, E5 plumbing) were built earlier this evening, but that session was cut off before it logged anything. Their four runs are now in the table below, copied from `results/`.
- **Did:** One normaliser test was failing (`Stn Rd` should expand to `Station Road`). Fixed `expand()`: a short form now also expands when the next word is itself a short form. Re-ran both normaliser runs; the numbers did not change.
- **Did:** Applied the rest of `build-plan.md` §4 to `initials.md`: §5.3 (form fields, split by respondent, test-set protection, thin labels), §7 (fixed hyperparameters, 3 seeds per config, the 40k-synthetic run, dev-only label mapping), §8 (paired bootstrap), §13.
- **Did:** Two independent reviews (python-reviewer on the CRF and normaliser; mle-reviewer on the plan). 15 findings, recorded as C1–C15 in `build-plan.md` §2.4. Fixed in code, tests first:
  - Normaliser: a PIN that covers several districts (1,168 of 19,300) accepts any of them and fills none; a misspelt state or district is no longer a mismatch; filled districts are scored as leniently as `pin_check` judges them; a state in Devanagari is left unchecked.
  - `src/annotate.py tasks`: the respondent → split map is saved in `data/real/splits.json` so nobody moves on a re-run; the test set fills before the pool; blank respondent codes are rejected. Label Studio now shows the place name.
  - CRF: `--n-train` above the train size no longer crashes.
- **Tests:** 171 passing, 92 % coverage. `torch` 2.15.0 with CUDA and `transformers` 5.22.2 import and see the RTX 4060.
- **Decided (provisional; rows in `initials.md` §13):** B3, B5, B6, B10–B14 as proposed in `build-plan.md`; C4, C5, C7–C11.
- **Results:** `20261110-crf-synfull-dev-s0`, `20261110-crf-synfull-hand20-s0`, `20261110-normalizer-synfull-dev-s0`, `20261110-normalizer-hand20-s0`. The two normaliser runs were re-run after the fixes and their rows show the new numbers.
- **Problems:** The CRF scores 0.992 on synthetic dev and 0.670 on the 20 hand-written items, below the rules baseline (0.764) on the same items. Those 20 items are ours, not form data: a smoke signal, not a result. The review's C1 (a place name on every synthetic address) and C2 (hand-written name lists of 20–40 items) are likely causes and are open. For the full-data CRF the seed changes nothing, so it has no seed spread to report. The paired, respondent-level bootstrap in `initials.md` §8 is not written yet. Deadline still blank and the form not sent.
- **Next:** Send the form. Team decisions C1–C3 (`initials.md` §12) before step 10, because C1 and C2 change the synthetic data. Then step 10 (MuRIL / XLM-R; time the first run), step 11 (zero-shot E1, with C6), step 12 with the bootstrap.

### 2026-11-10 (night): Claude Code, steps 7 and 8
- **Did:** Step 8, noise families N1–N8 as pure functions in `src/generate/noise.py`, switched per config (`clean`, `full`, `full-minus-Nx` for E3, `only-Nx` for tests); the exact transformation list is in `initials.md` §5.2. Each record's `noise` field lists the families that changed it. On `full`, 400 sampled addresses: firing rates N1 0.89, N2 0.96, N3 0.93, N4 0.88, N5 0.51, N6 0.59, N7 0.98, N8 0.81; 6.5 families per address; every label present in 77–88 % of records.
- **Did:** Step 7, `src/annotate.py` (form CSV → Label Studio tasks split by respondent and deduplicated by place; Label Studio export → our JSONL with spans snapped to tokens; agreement = strict span F1 + token Cohen's κ) and `data/real/labelstudio_config.xml`. Guidelines updated with the process.
- **Tests:** 117 passing, 93 % coverage. Every family has a test that labels stay valid and that it changes only the records it reports.
- **Decided (provisional):** Label Studio; the form export needs a `respondent` code column; Devanagari stays out of the generator (`initials.md` §13).
- **Results:** none (no training runs yet).
- **Problems:** The agreement number for step 7's gate needs real double-annotated items; the script is verified on the 20-item fixture only. Title Case noise turns `HP` into `Hp` and `B.R.` into `B.r.`; left as is, it is plausible noise. Code review (python-reviewer, 60k fuzzed addresses through N1–N8 with no failure) found no bug in the noise code; three fixes applied in `src/annotate.py` (dedupe keyed on place + city and never on blank places, every bad span reported at once, submitted all-O annotations kept).
- **Next:** Send the form. Step 9 (CRF on `full`), step 10 (MuRIL / XLM-R; `uv sync --extra train` first, ask before the torch download), then step 11 (zero-shot E1).

### 2026-11-10 (evening): Claude Code, resuming the build for the group
- **Did:** Built Phase 1 (`build-plan.md` §3.2 steps 1–6) on Python 3.12 / `uv`; git repo initialised.
  - `src/data.py`: labels, the one shared tokenizer, spans ↔ BIO, JSONL read/write with validation.
  - `src/eval/`: seqeval strict-IOB2 span F1 (micro / macro / per label), exact match, field-level set P/R/F1, bootstrap 95% CI, per-slice F1, `results/` writer.
  - `src/normalize/gazetteer.py`: downloads the PIN directory mirror (157,126 offices, 19,300 PINs, 36 states), cleans office names, canonical state table, PIN → offices / city / district / state. 490023 → Supela Bhilai, Durg, Chhattisgarh.
  - `src/generate/`: generator v1 with the `clean` config (fixed order, place-name prefix tagged O, fixed 90/5/5 held-out PIN split), deterministic by seed; 1,000 addresses in under a second.
  - `src/models/rules.py`: B0 chunk-based rules + gazetteer baseline with a CLI that writes `results/<run-id>.json`.
  - `tests/fixtures/handlabelled20.jsonl`: 20 hand-written public-place addresses (written by us, **not** form data) for the Phase 1 gate; `data/real/GUIDELINES.md` v1.
  - 94 tests, 93 % line coverage (`uv run pytest`).
- **Decided (provisional; team to confirm; rows in `initials.md` §13):** A2–A6, B2, B8, B9, B15 as proposed in `build-plan.md`; Python 3.12 via `uv`.
- **Results:** `20261110-rules-hand20-s0` (table below). A smoke run on our own hand-written items, not a real-data number.
- **Problems:** Deadline still blank and the form not sent; both need the team. The rules baseline is weak on CITY (F1 0.33: "Bhilai" is not a division name, so the gazetteer has no city entry for it) and LOCALITY (0.56); OSM names in step 8 should help. `torch` / `transformers` are not installed yet (optional extra; ask before the download). Known B0 limitation left as is (review finding): a bare `Delhi` / `Chandigarh` / `Goa` chunk is tagged STATE even in the city slot. Code review (python-reviewer) found no critical issues; six fixes applied (normalised gazetteer keys, offset-based text slices, JSONL reading, title-casing, pinned and atomic download, PIN split over all PINs).
- **Next:** Send the form. Step 7 (Label Studio config, export through `src.data`, agreement script), step 8 (noise families N1–N8 as config switches), step 9 (CRF). Before step 10: `uv sync --extra train` on the GPU machine and time one MuRIL run.

### 2026-11-10: Paritosh (with Claude Code)
- **Did:** The group picked project #8 from the shortlist (Indian address parsing). Created this folder with `CLAUDE.md`, `initials.md`, `research.md`, `resources.md` and `logs.md`.
- **Did:** Literature and resource scan (`research.md`, links in `resources.md`). Found:
  - the closest paper (Dhavase & Chaudhary, ETASR 2026), which shows the synthetic→real collapse;
  - six open Indian-address models on Hugging Face;
  - the abr-6024 synthetic dataset;
  - deepparse's zero-shot results for India;
  - the count of OSM India objects with a PIN tag (283,211).
- **Decided:**
  - Research focus: the synthetic→real gap, studied through a noise ablation, a real-data curve and a human-written test set.
  - Label schema v0.1.
  - Real data comes from public-place addresses only.
  - Details in `initials.md` §13.
- **Results:** none yet (no code).
- **Next:**
  - Fill in the deadline at the top of `initials.md` and work through §12 (open decisions).
  - Start Phase 1 (`initials.md` §10): repo setup, clean the PIN directory, annotation guidelines, send the Google Form, generator v1, eval script.
  - Paste the new title and description (`initials.md` §14) into the class sheet.

### 2026-10-22: Paritosh (with Claude Code)
- **Did:** Reviewed all 31 entries in the class sheet (`DOC-20260928-WA0000.xlsx`) to find topics no other group had taken. Shortlisted and ranked 9 ideas; address parsing was #8.
- **Outputs (parent folder):** `Group_10_Project_Shortlist.pdf` (detailed), `Group_10_Project_Shortlist_Brief.pdf` / `.tex`, `Group_10_Project_Shortlist.md`.
- **Also:** reviewed a ChatGPT-generated alternative report (`Group_10_NLP_Project_Analysis_and_Recommendations.pdf`).

### By 28 Sep 2026 (date of the class-sheet copy)
- The group's row in the class sheet held a placeholder: "NLP Modeling – Use NLP for language modeling". Replace it with `initials.md` §14.

---

## Run table
Log every training or evaluation run, including failed ones. Fill the **Test F1** column only for final runs (`CLAUDE.md` rule 2).

| Run ID | Date | Model | Train data | Seed | Dev F1 (micro / macro) | Test F1 | Results file | Notes |
|---|---|---|---|---|---|---|---|---|
| 20261110-rules-hand20-s0 | 2026-11-10 | B0 rules + gazetteer | none (no training) | 0 | hand20 smoke set, not real dev: 0.764 / 0.797 (CI95 0.689–0.828) | – | `results/20261110-rules-hand20-s0.json` | 20 hand-written public-place items; exact match 0.05; CITY 0.33, LOCALITY 0.56, PIN 0.97 |
| 20261110-crf-synfull-dev-s0 | 2026-11-10 | B1 CRF (features v1, c1 = c2 = 0.1, 100 iterations) | synthetic `full`, 20k | 0 | synthetic `full` dev (2,000), not real dev: 0.992 / 0.992 (CI95 0.991–0.994) | – | `results/20261110-crf-synfull-dev-s0.json` | exact match 0.95; lowest labels LOCALITY 0.971, CITY 0.974 |
| 20261110-crf-synfull-hand20-s0 | 2026-11-10 | B1 CRF (same model) | synthetic `full`, 20k | 0 | hand20 smoke set, not real dev: 0.670 / 0.653 (CI95 0.600–0.749) | – | `results/20261110-crf-synfull-hand20-s0.json` | exact match 0.10; PIN 1.00, LANDMARK 0.92, STATE 0.88, BUILDING 0.80, CITY 0.56, LOCALITY 0.50, STREET 0.48, HOUSE 0.40, DISTRICT 0.33 |
| 20261110-normalizer-synfull-dev-s0 | 2026-11-10 | Normaliser on gold tags (E5 plumbing) | none (no training) | 0 | no F1; synthetic `full` dev (2,000): PIN valid 1,383, missing 375, not found 234, mismatch 8 | – | `results/20261110-normalizer-synfull-dev-s0.json` | re-run after the review fixes (before: valid 1,266, mismatch 125, mostly typos and PINs that span districts); filled district 241 / 241, filled state 251 / 251; this data holds no real contradictions, so the 8 mismatches are false alarms |
| 20261110-normalizer-hand20-s0 | 2026-11-10 | Normaliser on gold tags (E5 plumbing) | none (no training) | 0 | no F1; hand20 smoke set: PIN valid 15, missing 4, not found 1, mismatch 0 | – | `results/20261110-normalizer-hand20-s0.json` | re-run after the review fixes (before: valid 14, mismatch 1); the fixture has no `meta`, so fill accuracy is not scored |
