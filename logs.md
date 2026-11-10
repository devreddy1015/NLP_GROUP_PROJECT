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

### 2026-11-11 (00:40–01:35): Claude Code, M1 hyperparameters fixed, mixed precision
- **Did:** Step 10 is committed (`f526159`). Mixed precision added to `src/models/transformer.py`, tests first: bf16 autocast when training on a GPU, recorded as `amp` in every results file; a CPU run stays in full precision. One test trains the tiny model on the GPU with it on.
- **Did:** Five tuning runs of MuRIL on `full` (20k, seed 42), each scored on synthetic dev and hand20. They were called from Python as `run(..., epochs=E, lr=LR)`; the command line has no flags for these on purpose. The rule was set before the runs: **the cheapest setting whose synthetic-dev micro F1 is within 0.002 of the best** (0.002 is the half-width of the interval on 2,000 addresses). hand20 was scored but decided nothing.

  | Epochs | Learning rate | Precision | Synthetic dev micro F1 | Training time | Mean loss, last epoch |
  |---|---|---|---|---|---|
  | 3 | 5e-5 | full (the first run, below) | 0.9880 | 508 s | 0.362 |
  | 3 | 5e-5 | mixed | 0.9830 | 318 s | 0.384 |
  | 5 | 5e-5 | mixed | 0.9910 | 527 s | 0.072 |
  | **3** | **1e-4** | **mixed** | **0.9903** | **314 s** | 0.054 |
  | 5 | 1e-4 | mixed | 0.9904 | 523 s | 0.014 |
  | 3 | 1e-4 | full | 0.9900 | 532 s | 0.054 |

- **Decided (provisional; row in `initials.md` §13):** 3 epochs, learning rate 1e-4, batch 32, mixed precision on. Three settings are inside the margin and this is the cheapest. Two more epochs at that rate change nothing (0.9904), so it has converged. Mixed precision costs nothing once the model converges (0.9903 against 0.9900, the same loss curve) and trains 1.7 times faster; the 0.005 gap at lr 5e-5 came from under-training.
- **Tests:** 186 passing, 92 % coverage.
- **Results:** the ten `muril-tune` rows below.
- **Timing:** 314 s to train and about 8 minutes end to end at the fixed settings. 48 runs need a little over 6 hours, inside the 4–8 hour estimate (`build-plan.md` §3.3).
- **Problems:**
  - MuRIL still does not beat the CRF: 0.990 against 0.992 on synthetic dev, and 0.737 against 0.796 on hand20 (intervals overlap, n = 20). Real data has to settle this.
  - hand20 cannot tell these models apart. The same setting scored 0.737 with mixed precision and 0.777 in full precision, with the same loss curve.
  - The choice rests on one seed, and the three settings inside the margin differ by less than the interval.
  - The rate 1e-4 was chosen on MuRIL, which starts slowly. XLM-R is not run yet and often trains badly at that rate; check its first run before reusing the settings for M2.
  - The first-guess run keeps its rows (`20261111-muril-synfull-*-s42`); it is not at the fixed settings.
  - The ten tuning `.pred.jsonl` files (2.9 MB) are on disk. No paired test will use them, so they need not be committed.
- **Next:** Step 11 (zero-shot E1). A first XLM-R run. Three seeds per config once real data exists. Send the form.

### 2026-11-11 (00:00–00:35): Claude Code, step 10 (MuRIL / XLM-R training script)
- **Did:** The generator fix below is committed (`dae3a88`).
- **Did (tests first):** `src/models/transformer.py` (M1, M2). A plain PyTorch loop (`accelerate` is not installed, so no HF `Trainer`): AdamW, linear warm-up over the first 10 % of steps, gradient clipping at 1.0, the tag on the first subword of each token. It trains once and scores every `--eval` file with that one model, because a GPU run does not repeat bit for bit. Next to each results file it writes `<run-id>.pred.jsonl` (one `{"id", "tags"}` line per address) for the paired bootstrap. The tests use a random two-layer BERT saved to a temp folder, so they download nothing.
- **Review (mle-reviewer):** no alignment or training-loop defect. Fixed after it: eval files are read before training (a mistyped path used to lose the run); a run ID must end in `-s<seed>` and cannot repeat; the model commit is pinned (`transformers` 5.22 no longer records it). The tests now fail if `predict` reads the wrong subword, skips eval mode, lets padding change a prediction, or if `run` retrains per eval file. Not done: saving the model (the demo needs it, step 15), a train / eval overlap check (belongs with the E4 mixing in step 12), counting cut-off or dropped tokens (none in the current data: at most 101 subwords against a limit of 256).
- **Tests:** 185 passing, 92 % coverage.
- **Results:** the two MuRIL rows below (seed 42, `full`, 20k). Synthetic dev 0.988 micro F1; hand20 0.745. Both are below the CRF (0.992 and 0.796); on hand20 the intervals overlap and n is 20.
- **Timing (RTX 4060 Laptop, fp32, batch 32):** 508 s to train (1,875 steps), about 11 minutes end to end with scoring. 48 runs at that rate is about 9 hours, above the 4–8 hour estimate (`build-plan.md` §3.3).
- **Decided (provisional; row in `initials.md` §13):** train once and score every eval file; pinned model commits; run IDs end in the seed; predictions saved per address.
- **Problems:**
  - The hyperparameters (3 epochs, lr 5e-5) are a first guess and **not settled**. Mean loss per epoch was 2.07, 0.70, 0.36, so the run was still improving when it stopped, and a 60-step smoke run learnt nothing (loss 2.90 → 2.84, F1 0). MuRIL starts slowly at this learning rate.
  - The first load downloaded about 1.9 GB, not 953 MB: `transformers` fetched the official `pytorch_model.bin` and also a bot-converted `model.safetensors` from a pull-request branch. The pinned commit now loads only the official file; the other copy sits unused in `~/.cache/huggingface`.
  - `results/20261111-muril-synfull-dev-s42.pred.jsonl` is 582 KB. 48 such files would add about 28 MB to git; decide before step 12 whether synthetic-dev predictions are committed.
  - XLM-R is not downloaded or run yet; only its commit is pinned.
- **Next:** Settle epochs and learning rate on synthetic dev (and try mixed precision for speed), then 3 seeds. Step 11 (zero-shot E1). Send the form.

### 2026-11-10 (night, 21:30–22:00): Claude Code, city and street names in the generator
- **Did:** Committed the work below (`7329519`). Then listed what the CRF gets wrong on the 20 hand-written items. Two causes stood out, both in the generator's geography (build-plan C2), not in the models:
  - `Bhilai` was tagged LOCALITY in 10 of 20 items. Cities came from postal division names, so `Bhilai` appeared in training only inside office names (`Supela Bhilai`), and metros appeared as `Mumbai City North West` or `Pune City East` (52 such names, 9,994 offices).
  - `Hosur Road`, `Junwani Road`, `Station Road` and three more were tagged LOCALITY. With 30 fixed street names the model memorised names instead of reading `Road`.
- **Did (tests first):** `clean_city` drops trailing direction words (`Mumbai City North West` → `Mumbai`). `gazetteer._towns` reads a town off the end of office names when several offices in a district share it and an office there is named after it; that town becomes the city (`490023` → `Bhilai`) and the generator drops it from the locality (`Supela, Bhilai`). Half the named streets now take a real place name. Bare `Plot 12` / `Shop 4` added to the house forms. Gazetteer rebuilt (524 city names, was 460); `clean` and `full` regenerated with seed 42; all five runs re-run.
- **Tests:** 177 passing, 92 % coverage.
- **Decided (provisional; row in `initials.md` §13):** the city and street rules above. The place-name prefix (C1) stays as it is: 17 of the 20 hand-written items do start with the place name.
- **Results:** the five rows below, re-run. On hand20 the CRF went from 0.670 to 0.796 micro F1 (CITY 0.56 → 0.93, STREET 0.48 → 0.62, exact match 0.10 → 0.40) and the rules baseline from 0.764 to 0.828 (CITY 0.33 → 0.79). Synthetic dev is unchanged at 0.992. The earlier numbers are in git (`7329519`).
- **Problems:** The 20 items were used to find these faults, so part of the gain is fitted to them; they are a smoke check, never a result. Still weak on hand20 for the CRF: HOUSE 0.40 (`Ground Floor`, `1st floor`), DISTRICT 0.40, STREET 0.62, LOCALITY 0.70 (`Sector 10`, `Phase 2` read as streets). Place, landmark and building names are still short hand-written lists. The town rule lets a little junk through (`Halli`, `Pipaliya`).
- **Next:** Send the form. Step 10 (MuRIL / XLM-R; time the first run). OSM names for places, landmarks and buildings once the cities are chosen.

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
| 20261110-rules-hand20-s0 | 2026-11-10 | B0 rules + gazetteer | none (no training) | 0 | hand20 smoke set, not real dev: 0.828 / 0.848 (CI95 0.745–0.897) | – | `results/20261110-rules-hand20-s0.json` | 20 hand-written public-place items; re-run after the city fix in the gazetteer (before: 0.764 / 0.797, CITY 0.33); exact match 0.25; CITY 0.79, LOCALITY 0.56, PIN 0.97 |
| 20261110-crf-synfull-dev-s0 | 2026-11-10 | B1 CRF (features v1, c1 = c2 = 0.1, 100 iterations) | synthetic `full`, 20k | 0 | synthetic `full` dev (2,000), not real dev: 0.992 / 0.992 (CI95 0.990–0.993) | – | `results/20261110-crf-synfull-dev-s0.json` | re-run on the regenerated data (same scores as before); exact match 0.95; lowest labels LOCALITY 0.97, CITY 0.97 |
| 20261110-crf-synfull-hand20-s0 | 2026-11-10 | B1 CRF (same model) | synthetic `full`, 20k | 0 | hand20 smoke set, not real dev: 0.796 / 0.724 (CI95 0.708–0.889) | – | `results/20261110-crf-synfull-hand20-s0.json` | re-run after the generator's city and street fix (before: 0.670 / 0.653, exact match 0.10); exact match 0.40; PIN 0.94, CITY 0.93, LANDMARK 0.92, STATE 0.81, BUILDING 0.80, LOCALITY 0.70, STREET 0.62, DISTRICT 0.40, HOUSE 0.40 |
| 20261110-normalizer-synfull-dev-s0 | 2026-11-10 | Normaliser on gold tags (E5 plumbing) | none (no training) | 0 | no F1; synthetic `full` dev (2,000): PIN valid 1,376, missing 377, not found 244, mismatch 3 | – | `results/20261110-normalizer-synfull-dev-s0.json` | re-run on the regenerated data (first run, before the review fixes: valid 1,266, mismatch 125, mostly typos and PINs that span districts); filled district 249 / 249, filled state 287 / 287; this data holds no real contradictions, so the 3 mismatches are false alarms |
| 20261110-normalizer-hand20-s0 | 2026-11-10 | Normaliser on gold tags (E5 plumbing) | none (no training) | 0 | no F1; hand20 smoke set: PIN valid 15, missing 4, not found 1, mismatch 0 | – | `results/20261110-normalizer-hand20-s0.json` | re-run after the review fixes (before: valid 14, mismatch 1); the fixture has no `meta`, so fill accuracy is not scored |
| 20261111-muril-synfull-dev-s42 | 2026-11-11 | M1 MuRIL (`google/muril-base-cased` at commit `afd9f36`; 3 epochs, lr 5e-5, batch 32, fp32) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.988 / 0.988 (CI95 0.986–0.990) | – | `results/20261111-muril-synfull-dev-s42.json` | exact match 0.93; CITY 0.96, LOCALITY 0.95, every other label 0.99 or more; trained in 508 s; mean loss per epoch 2.07 / 0.70 / 0.36, so still improving; first-guess hyperparameters, not the fixed set; tags per address in `results/20261111-muril-synfull-dev-s42.pred.jsonl` |
| 20261111-muril-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.745 / 0.676 (CI95 0.644–0.839) | – | `results/20261111-muril-synfull-hand20-s42.json` | exact match 0.40; PIN 1.00, STATE 0.89, STREET 0.79, CITY 0.78, LANDMARK 0.73, HOUSE 0.57, LOCALITY 0.48, BUILDING 0.44, DISTRICT 0.40; below the CRF's 0.796, but the intervals overlap and n = 20 |
| 20261111-muril-tune-e3lr5e5-synfull-dev-s42 | 2026-11-11 | M1 MuRIL, tuning (3 epochs, lr 5e-5, mixed precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.983 / 0.983 (CI95 0.981–0.985) | – | `results/20261111-muril-tune-e3lr5e5-synfull-dev-s42.json` | exact match 0.90; trained in 318 s; under-trained (mean loss per epoch 2.10 / 0.73 / 0.38) |
| 20261111-muril-tune-e3lr5e5-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL, tuning (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.783 / 0.734 (CI95 0.680–0.884) | – | `results/20261111-muril-tune-e3lr5e5-synfull-hand20-s42.json` | exact match 0.50; not used to choose |
| 20261111-muril-tune-e5lr5e5-synfull-dev-s42 | 2026-11-11 | M1 MuRIL, tuning (5 epochs, lr 5e-5, mixed precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.991 / 0.991 (CI95 0.989–0.993) | – | `results/20261111-muril-tune-e5lr5e5-synfull-dev-s42.json` | exact match 0.95; trained in 527 s; the best score, but 0.0007 above the 3-epoch setting at lr 1e-4 |
| 20261111-muril-tune-e5lr5e5-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL, tuning (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.777 / 0.694 (CI95 0.676–0.873) | – | `results/20261111-muril-tune-e5lr5e5-synfull-hand20-s42.json` | exact match 0.45; not used to choose |
| 20261111-muril-tune-e3lr1e4-synfull-dev-s42 | 2026-11-11 | M1 MuRIL at the **fixed settings** (3 epochs, lr 1e-4, batch 32, mixed precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.990 / 0.990 (CI95 0.988–0.992) | – | `results/20261111-muril-tune-e3lr1e4-synfull-dev-s42.json` | exact match 0.94; CITY 0.97, LOCALITY 0.96, every other label 0.996 or more; trained in 314 s; chosen as the cheapest setting within 0.002 of the best |
| 20261111-muril-tune-e3lr1e4-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL at the fixed settings (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.737 / 0.650 (CI95 0.635–0.828) | – | `results/20261111-muril-tune-e3lr1e4-synfull-hand20-s42.json` | exact match 0.35; PIN 1.00, STATE 0.85, CITY 0.83, STREET 0.75, LANDMARK 0.62, LOCALITY 0.57, BUILDING 0.50, HOUSE 0.40, DISTRICT 0.33; below the CRF's 0.796, intervals overlap |
| 20261111-muril-tune-e5lr1e4-synfull-dev-s42 | 2026-11-11 | M1 MuRIL, tuning (5 epochs, lr 1e-4, mixed precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.990 / 0.990 (CI95 0.989–0.992) | – | `results/20261111-muril-tune-e5lr1e4-synfull-dev-s42.json` | exact match 0.95; trained in 523 s; two more epochs add nothing |
| 20261111-muril-tune-e5lr1e4-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL, tuning (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.768 / 0.701 (CI95 0.667–0.859) | – | `results/20261111-muril-tune-e5lr1e4-synfull-hand20-s42.json` | exact match 0.40; not used to choose |
| 20261111-muril-tune-e3lr1e4fp32-synfull-dev-s42 | 2026-11-11 | M1 MuRIL, precision check (3 epochs, lr 1e-4, full precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.990 / 0.990 (CI95 0.988–0.992) | – | `results/20261111-muril-tune-e3lr1e4fp32-synfull-dev-s42.json` | exact match 0.95; trained in 532 s; the same score and loss curve as mixed precision, 1.7 times slower |
| 20261111-muril-tune-e3lr1e4fp32-synfull-hand20-s42 | 2026-11-11 | M1 MuRIL, precision check (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.777 / 0.700 (CI95 0.670–0.874) | – | `results/20261111-muril-tune-e3lr1e4fp32-synfull-hand20-s42.json` | exact match 0.45; 0.04 above its mixed-precision twin, which is the noise of a 20-item set |
