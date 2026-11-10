# Build plan and cross-check of the plan

**Status 10 Nov 2026 (late evening):** steps 1–9 and 13 of §3.2 are built and tested (171 tests pass; CRF and normaliser runs in `logs.md`); torch + CUDA installed locally. The §4 edits are applied. A second review found 15 more problems (§2.4): six are fixed in code (C8–C12, C14), three are now rules in `initials.md` (C4, C5, C7), and C1–C3 need a team decision before step 10. Step 9's "real-dev row" still waits for real data: its two runs are on synthetic dev and the 20-item fixture. Before that: steps 1–8 were built and tested; step 7's agreement number waits for real double-annotated items (the script runs on the 20-item fixture). Earlier: steps 1–6 were built and tested (Phase 1 gate passed: the rules baseline scores on 20 hand-labelled items, `results/20261110-rules-hand20-s0.json`). A1 (deadline) and sending the form are still open; A2–A6 were adopted as proposed and logged in `initials.md` §13. Details in `logs.md`.

Written 10 Nov 2026 from a full read of `CLAUDE.md`, `initials.md`, `research.md`, `resources.md` and `logs.md`, plus live checks of the resources the code depends on. Nothing here changes those five files; §4 lists the edits they need once the team agrees.

## 1. Checked against live sources today
| Claim in the docs | Result |
|---|---|
| The zip next to this folder is the same as the folder | Same, all five files |
| The eight Hugging Face model IDs exist and are open | All exist, none gated |
| Licences marked "check card" | Shiprocket TinyBERT, gagan1985, abr-6024 model and dataset: Apache-2.0. howdoiuse-keyboard: MIT. mukuls9971 and Shiprocket IndicBERT: no licence in the model metadata |
| Does Shiprocket TinyBERT use the same labels as IndicBERT? (`research.md` §10) | Yes, identical 11 labels |
| mukuls9971 MuRIL labels (`research.md` §2) | The doc misses one: the model also has `ADDRESS_DISTRICT`. Its label set is exactly our nine |
| gagan1985 labels | Match the doc, but the model is missing from the mapping table |
| howdoiuse-keyboard labels (doc says "?") | `HOUSE_NUMBER PLOT FLOOR BLOCK GALI AREA SUBAREA COLONY SECTOR CAMP KHASRA POLE CITY STATE PINCODE`. Delhi-style; no landmark, building or road label |
| PIN directory mirror | `pincode.csv`, 23.8 MB, branch `main`. Columns: `CircleName RegionName DivisionName OfficeName Pincode OfficeType Delivery District StateName Latitude Longitude` |
| 490023 belongs to Supela (`initials.md` §2) | Yes. India Post lists three offices under it: Kohka Bhilai, Supela Bhilai, Vaishali Nagar Bhilai; district Durg |
| Stack: Python 3.10+, free T4 | This machine has Python 3.14.8, an RTX 4060 Laptop GPU (8 GB), `uv`, `git`, `gh`. The folder is not a git repo yet |

**Not checked:** the numbers quoted from papers (ETASR table, deepparse accuracies), OSM coverage of any city, the data.gov.in portal, the Kaggle copy, and the course files in `resources.md` §9 (they are not on this machine).

## 2. Problems found

### 2.1 Decide before writing code or sending the form
| # | Problem | Where | Suggested fix |
|---|---|---|---|
| A1 | The deadline is blank. The course ends in November and the plan needs 20 days from today (30 Nov). Collecting real addresses is the slowest step and is not code | `initials.md` header, §10 | Fill in the date. Send the form today: it does not depend on the guidelines |
| A2 | Every public-place address starts with the place's own name ("Sharma Sweets, shop 4, …"). The schema has no rule for it and the generator never writes one, so the real test set differs from training in a way that is not "noise". This inflates the gap RQ1 measures | §4, §5.2, §5.3 | Tag the place name `O`, like a person's name. Ask for it in its own form field, before the address. Have the generator prefix an OSM place name tagged `O` in every config, clean included |
| A3 | No tokeniser is defined, and the generator, the annotation export, the CRF and the eval must all share one. `b'lore-68` only works if `-` splits. The obvious regex `\w+\|[^\w\s]` shreds Devanagari (`मंदिर` becomes 5 tokens; tested) | `CLAUDE.md` data format | One `tokenize()` used everywhere: split on whitespace, and make every Unicode punctuation or symbol character its own token. Keep character offsets |
| A4 | The PIN directory has no city column, and its district and state are in capitals with India Post spellings (`Chattisgarh`). One PIN covers several offices. So the generator has no source for CITY and the normaliser cannot check "PIN–city" as §8 says | §5.1, §8 | CITY comes from OSM for the chosen cities, and from `DivisionName` minus "Division" elsewhere. Add a 36-row canonical state table. `pin_check` passes if the locality matches any office under that PIN. Drop "PIN–city" from §8 or define it as locality match |
| A5 | DISTRICT or merge into CITY is still open, and it changes the label set every module uses | §12 | Keep DISTRICT (the mukuls model has it, so E1 loses nothing). Rule: DISTRICT only after a cue word (`Dist`, `Distt`, `Zila`); otherwise CITY |
| A6 | Devanagari in or out is still open. It decides how N6 is built. IndicXlit depends on fairseq, which is hard to install on current Python (not tested today) | §12, §5.2 N6 | Core scope: Roman script and Hinglish. N6 = a spelling-variant table. If Devanagari is wanted, take real names from OSM `name:hi` tags, not a transliteration model |

### 2.2 Experiment design
| # | Problem | Where | Suggested fix |
|---|---|---|---|
| B1 | Noise families overlap, so removing one does not remove what its name says. PIN dropping is in N3 and N7; `bhilai-23` and `B'lore` straddle N1 and N7; `ke paas` is N2 and N6 | §5.2, E3 | Each transformation belongs to exactly one family. Write the list in §5.2 |
| B2 | "Clean" is undefined. If clean data has no landmarks, a model trained on it can never output LANDMARK, and E2 measures a missing label, not noise | §5.2, E2 | Clean = all nine fields, one fixed order, commas, full words, one plain "near X" landmark. N2 = variety in landmark phrasing |
| B3 | The four people who annotate the test set also write the generator's word lists | Rule 2, §5.3 | Split the real data first. Annotate dev and the pool first. Tag the generator in git before anyone reads test items; later changes must cite a dev example |
| B4 | A random split leaks: classmates will describe the same campus places, and one person's style lands in both the pool and the test set | §5.3, E4 | Split by form respondent and deduplicate by place |
| B5 | 50 dev addresses cannot select a model or stop training. If dev picks the checkpoint, the "0 real examples" runs are not zero | §5.3, E4 | Fix hyperparameters and epoch count once, on the full-noise config. No per-run selection |
| B6 | H3 compares against "doubling the synthetic data" but no run does that | §3, E4 | Add one run: 40k synthetic, k = 0 |
| B7 | How k real examples are mixed into 20k synthetic is not stated. 25 in 20,000 is swamped | E4 | Choose before any result: upsample real to about 10% of the training set. Use nested subsets (25 ⊂ 50 ⊂ 100 ⊂ 200) |
| B8 | E2 scores on a "synthetic test" split that §5.2 never creates | §5.2, E2 | Generate 2k test addresses from PINs held out of train |
| B9 | "Field-level accuracy" is undefined, yet E1, E6 and §5.4 depend on it. The JSON-output systems (abr-6024, L1) cannot be scored with seqeval, and abr-6024 rewrites values (`MH` to `Maharashtra`) | §8 | One definition: per field, compare the sets of predicted and gold strings after lower-casing and stripping punctuation; report P / R / F1 on shared fields only. Canonicalise state names on both sides |
| B10 | E5 needs the true PIN, district and state of each real address. Span tags do not hold them when the writer left them out | §5.3, E5 | After the free-text box, the form asks for PIN, city and state in separate fields. Store them under `meta` |
| B11 | Public places rarely have `Flat` or `H.No.`, so HOUSE and BUILDING will be thin in the real test set, and HOUSE was the worst label in the ETASR paper | Rule 1, §4, H1 | Add `Shop No. 4`, `Plot 12`, `1st floor` to the HOUSE examples. Report span counts per label. Make no claim on a label with under about 30 test spans |
| B12 | Respondents are IIT Bhilai classmates, so the test set will lean to Bhilai, Durg and Raipur | §5.3 | Ask each person for five places in different towns, one in their home town. Report the state spread |
| B13 | Eight leave-one-out differences may be smaller than the confidence interval on 250 addresses | E3, §8 | Three seeds for every E3 and E4 config (cost in §3.3). Use a paired bootstrap for differences. Report a difference inside the interval as "no measurable effect" |
| B14 | The E1 label mappings will be adjusted while looking at model output | E1, Rule 2 | Build the mappings on dev. Run the test set once |
| B15 | Two landmark cases have no rule: back-to-back landmarks and a landmark with no relation word | §4 | Each relation word opens a new span (`behind SBI atm` / `nr hanuman mandir`). A bare landmark is still LANDMARK |

### 2.3 Small fixes
- `CLAUDE.md` JSONL example: id `real-0042` with `"source": "synthetic"`.
- The data format has no field for test-item slices (§5.3). Add `slices`.
- Training only in notebooks is not reproducible (Rule 5). Put training in a script; the notebook calls it.
- BiLSTM-CRF (B2) is optional and left out of the build order below. §9 wants a BiLSTM link to the syllabus, so decide whether the report needs it.

### 2.4 Found by the second review (10 Nov, late evening)
Two independent reviews: one of the plan as amended, one of the CRF and normaliser code. Counts marked "checked" were re-run here.

| # | Problem | Status |
|---|---|---|
| C1 | Every synthetic address starts with a place name tagged `O` (2,000 of 2,000 on `full` dev, checked). The form puts the name in its own box, so many real addresses will not repeat it, and a model can learn "the first chunk is `O`". This is the A2 fix overshooting | **Open.** Measure on real dev how often the name is repeated and prefix at that rate in every config; 0.5 until it is known. Settle before step 10: it changes the synthetic data |
| C2 | The generator's names are short hand-written lists (32 place names, 39 landmarks, 30 streets, 20 buildings; checked), not OSM names. Vocabulary, which no noise family touches, may dominate the gap and flatten E3 | **Open**, waits on the city choice (`initials.md` §12). Add OSM names before E2–E4; report the share of real landmark and street words never seen in training |
| C3 | Leave-one-out (E3) measures what is lost when the other seven families stay, so overlapping families hide each other. The review also found couplings: N2 off removes N6's `ke paas` variants; N3 off raises N8's firing rate | **Open.** Add clean + one family runs (the `only-Nx` configs exist; 24 more runs) and log firing rates per config |
| C4 | The bootstrap resamples single addresses, but real data comes in groups of about five per respondent, so the intervals are too narrow | In `initials.md` §8: resample respondents. Code it with step 12 |
| C5 | Macro F1 swings on rare labels, and DISTRICT may have almost no real spans | In `initials.md` §8: macro over the labels with at least 30 test spans |
| C6 | Scoring for E1 and E6 is not ready: `evaluate()` always scores all nine fields; state names are not yet canonicalised for JSON-output systems; gold LANDMARK includes the relation word, so an output of `Apartments` for `Apts` counts as a miss, which is unfair to L1 and abr-6024 | **Open**, build with step 11: score each system on its shared fields and rescore M1 and B0 on the same fields; canonicalise states; prompt L1 to copy text verbatim |
| C7 | E5 on real data: the form has no district box, so a filled district has nothing to be checked against; and the same person types the address and the PIN, so wrong PINs will be rare | In `initials.md` §7 and §8: district fill is scored on synthetic data only; add a test that swaps in wrong PINs at a known rate |
| C8 | 1,168 of the 19,300 PINs (checked) cover more than one district. The lookup returned only the commonest one, which gave false mismatches and wrong fills | **Fixed.** Any district of the PIN is accepted; the district is filled only when the PIN has exactly one |
| C9 | `pin_check` called a misspelt but correct state or district a mismatch. That was most of the 125 mismatches on synthetic dev | **Fixed.** Closest state name; near-identical district spelling (cut-off 0.8, not tuned). Synthetic dev now has 8 mismatches, all false alarms |
| C10 | Filled districts were scored more strictly than `pin_check` judges them (`Bangalore` against `Bengaluru Urban` counted as wrong). A state written in Devanagari gave a false mismatch | **Fixed** |
| C11 | `src.annotate tasks`: re-running after more responses moved respondents between splits, some from test into train; the pool filled before the test set; a blank respondent column sent every row to dev | **Fixed.** The split is kept in `data/real/splits.json` (commit it); fill order is dev 50, test 200, pool 50, test 250, pool 200; blank codes are rejected |
| C12 | Annotators could not see the place name they must leave unlabelled | **Fixed.** Label Studio shows it above the text |
| C13 | Duplicate places are matched on exact name and city, so `IIT Bhilai` and `Indian Institute of Technology Bhilai` both survive | **Open.** Look for them by eye while annotating |
| C14 | CRF: `--n-train` above the train size crashed. Without `--n-train` the seed changes nothing | Crash **fixed**. Do not report a seed spread for the full-data CRF |
| C15 | Unstated: E4's base config and whether the k-subset changes with the seed; where L1's few-shot examples come from. Synthetic test PINs are held out, but every test city, district and state also occurs in train | **Open**, listed in `initials.md` §12 |

## 3. What to code, in order

### 3.1 Before any code (today)
Settle A1–A6, then send the form (fields: place name, free-text address, PIN, city, state; no email collection).

### 3.2 Build order
Python 3.12 through `uv` (matches Colab; 3.14 wheels for `torch` and `python-crfsuite` are not checked). Every step ends with a test or a logged run.

| # | Module | What it does | Done when |
|---|---|---|---|
| 1 | Repo | `git init`, `.gitignore` (`data/raw/`), pinned `requirements.txt`, commands in `CLAUDE.md` | `uv run pytest` starts |
| 2 | `src/data.py` | Label list, `tokenize()`, spans ↔ BIO, JSONL read / write with validation | Round-trip test passes on 10 hand-written addresses, including `b'lore-68`, `490 023`, a Devanagari one and back-to-back landmarks |
| 3 | `src/eval/` | Span F1 per label, micro and macro (seqeval, strict IOB2); exact match; field-level P / R / F1 (B9); bootstrap CI; per-slice table; writes `results/<run-id>.json` | Runs on 20 hand-labelled items. Gold against gold scores 1.0; all-`O` scores 0 |
| 4 | `src/normalize/gazetteer.py` | Downloads `pincode.csv`; strips `B.O` / `S.O` / `H.O`; fixes case; canonical states; PIN → offices, district, state | Lookup of 490023 returns Supela, Durg, Chhattisgarh |
| 5 | `src/generate/` v1 | Templates and the clean config; `--seed`, `--config`; train / dev / test with held-out PINs | 1,000 addresses pass step 2's validation and are identical for the same seed |
| 6 | `src/models/rules.py` (B0) | PIN regex, cue words, gazetteer lookup | Its predictions go through step 3 and produce a results file |
| | **Phase 1 gate** | | Eval runs on 20 hand-labelled items (§10) |
| 7 | Annotation | Label Studio config; export to JSONL through step 2; agreement script (span F1 and token κ) | 50 double-annotated items give an agreement number |
| 8 | Noise N1–N8 | One function per family, one switch each | Each family has a test: labels are still valid after the noise |
| 9 | `src/models/crf.py` (B1) | Shape, suffix, cue-word and gazetteer features | Real-dev row in `logs.md` |
| 10 | `src/models/transformer.py` (M1, M2) | HF token classification, first-subword labels, fixed hyperparameters, `--seed`; writes predictions for step 3 | One full-noise MuRIL run logs a real-dev row |
| 11 | `src/models/zeroshot.py` (Z) | Runs the open models, maps labels, scores fields | E1 logged on dev |
| | **Phase 2 gate** | | First real-dev numbers logged (§10) |
| 12 | `scripts/run_experiments.sh` | A loop over configs × seeds for E2, E3, E4. No framework | Results tables fill from `results/` |
| 13 | `src/normalize/normalize.py` | Expands abbreviations, splits landmark relation from anchor, checks and fills PIN, district, state | E5 logged |
| 14 | `src/models/llm.py` (L1) | Few-shot prompt → JSON → field-level score | E6 logged |
| 15 | `app/` | Gradio demo: tags, JSON, PIN warnings | Runs on the §2 example |
| 16 | `scripts/make_tables.py` | Report tables and figures from `results/` only (Rule 3) | Every number in the report traces to a file |

Step 2 comes first and everyone waits on it (about half a day). After that the four roles in §11 run in parallel:
- **Data:** form, guidelines, step 7.
- **Generator:** steps 4, 5, 8.
- **Modelling:** steps 6, 9, 10, 14.
- **Eval and demo:** steps 3, 11, 13, 15, 16.

### 3.3 GPU budget (estimate, not measured)
About 48 MuRIL runs: E2 (2 configs) + E3 (8) + E4 (4 values of k, plus the 40k run), each with 3 seeds, plus XLM-R. A run is 20k short addresses for a few epochs, roughly 5–10 minutes on a T4 or the local 4060, so 4–8 GPU hours in total. Time the first run before deciding whether E3 must fall back to the CRF.

## 4. Edits the other docs need once the team agrees
- `initials.md`: deadline; §4 (place name, landmark rules, HOUSE examples); §5.2 (clean config, one family per transformation, test split); §5.3 (form fields, split by respondent); §7 (40k run, mixing rule); §8 (field-level metric, drop PIN–city); §12 and §13 for every decision above.
- `research.md` §2: add DISTRICT to the mukuls9971 row and mapping column; add gagan1985 and howdoiuse-keyboard to the mapping table; close §10 question 3.
- `resources.md`: replace "check card" with the licences in §1.
- `CLAUDE.md`: fix the JSONL example; add `slices` and `meta`; add `build-plan.md` to the file table; Python 3.12.
- `logs.md`: entry for this session.
