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

### 2026-11-11 (03:56–04:00): Claude Code, faster bootstrap
- **Did (tests first):** `src/eval.span_scorer` no longer asks seqeval to rescore every resample. seqeval's strict IOB2 reader extracts the spans once, three counts are kept per address (right, wrong, missed), and a resample adds them up.
- **Checked:** a test compares the scorer with seqeval's `f1_score` on 64 resamples of tags with a third replaced at random (wrong labels, dangling `I-` tags, spans cut in two): equal to the bit. The interval of every stored span run that has saved tags (21 files) was recomputed and is identical to the one in its results file.
- **Timing (measured, 2,000 synthetic dev addresses, 1,000 resamples):** one interval took 126.2 s before and 0.6 s after. The whole `evaluate()` call now takes 1.3 s.
- **Effect on the budget (an estimate, not timed end to end):** a training run should drop from about 8 minutes to about 5 and a half (314 s of training plus loading and prediction), so the 48 runs to about four and a half hours. `compare.md` no longer takes a minute per pair.
- **Tests:** 245 passing, 91 % coverage.

### 2026-11-11 (03:30–03:55): Claude Code, review of steps 12, 14, 15 and 16, and its fixes
- **Review (mle-reviewer):** the statistics are right (respondent resampling, percentile indices; the address-level intervals already logged are unchanged), the loop never scores the real test set without `--final`, and E4's subsets are nested. It found the problems below. Each is fixed with a test.
- **Fixed in the tables (`scripts/make_tables.py`):**
  - Runs were matched to an existing system by file name only. A zero-shot run on real dev would have been paired with our synthetic-dev runs and stopped the script with no tables. They are now matched on the full data name, and `rules.py` names its data like every other script (`fixtures/handlabelled20`).
  - A results file scored on a different number of addresses than the gold file now holds is refused.
  - A repeated run counts once: the newest of a configuration and seed. This matters for `--final` runs, which train again.
  - Tuning runs are no longer listed among our systems in `shared.md`. Side effect: the MuRIL row there is the first-guess run, because the run at the fixed settings has a tuning ID.
  - Our systems are rescored by exactly the other system's rules: commas trimmed, and the parser's city-or-district key counted once on both sides. Before, a tagger that named both scored 0.833 where the parser scored 0.800 for the same content (reviewer's example).
  - `shared.md` gives the difference to each existing system with a paired field-level interval. `runs.md` shows macro F1 over the labels with at least 30 spans.
- **Fixed for ground rule 2:** `src.data.require_train` refuses anything but train-split records as training data (`transformer`, `crf`, `mix`) or as few-shot examples (`llm`). Before, `--shots data/real/test.jsonl` would have passed. The ids of the examples shown are stored in the results file.
- **Fixed in scoring:** a predicted span of punctuation alone is no longer a field. The stored `field` scores of our MuRIL and XLM-R runs were computed under the old rule and can be about 0.004 lower than a rescoring (reviewer's measurement); their span scores are not affected. Rules, CRF and the three JSON-system runs were re-run and their logged scores did not change.
- **Fixed in the loop script:** several experiments can be named at once (`e2 e3` used to run only `e3`); E4 checks that the real pool holds 200 addresses before anything trains; `RESULTS=` sets the results folder.
- **Tests:** 244 passing, 91 % coverage. They now fail if respondent groups are ignored (in `evaluate`, `paired_bootstrap` or `compare_table`), if only the first seed is compared, if predictions are matched as an unordered set, if E4's subsets stop nesting in a large pool, or if a finished run is not skipped.
- **Results (hand20, `report/tables/shared.md`; these replace the same-label numbers quoted in the two entries below):** against the abr-6024 parser's 0.645 on its eight labels: CRF 0.848, rules 0.821, XLM-R 0.778, MuRIL (first-guess run) 0.777. Against the few-shot LLM's 0.484 on nine labels: CRF 0.842, rules 0.828, XLM-R 0.772, MuRIL 0.754. Every interval excludes zero. **This is not a finding:** the 20 addresses were written by us in our own conventions and were used to fix the generator, so they favour our systems.
- **Not fixed (known limits):**
  - The paired interval covers the choice of test addresses, not the training seed, and unequal numbers of seeds are not corrected.
  - The bootstrap calls seqeval once per resample. That is most of the 2.5 minutes of scoring after each training run, and about a minute per pair in `compare.md` on 260 addresses with three seeds a side (reviewer's timing). *Fixed in the 03:56–04:00 entry.*
  - The dry run prints the `generate full` command more than once on a fresh checkout. `app/demo.py` got only a light read.

### 2026-11-11 (03:18–03:25): Claude Code, step 14 (few-shot LLM)
- **Did (tests first):** `src/models/llm.py` gains the few-shot system `l1-qwen`: `Qwen/Qwen2.5-1.5B-Instruct` at a pinned commit (Apache-2.0, 3.1 GB; the base model the abr-6024 parser was fine-tuned from) with a prompt in our nine labels that asks for the pieces copied exactly as written. `--shots FILE --n-shots N --seed S` draws the examples. An example that is also in the eval file is refused (rule 2), and the run ID must end in `-s<seed>`. An answer may hold a list per key. Each results file records the generation time.
- **Results (hand20; a smoke check):** the two `l1-qwen` rows below.
  - Five examples drawn from synthetic `full` train, a stand-in because there is no real-train pool yet: valid JSON 20 of 20, field micro F1 0.484, 11.1 s for 20 addresses on the RTX 4060.
  - No examples: valid JSON 9 of 20, field micro F1 0.288.
  - Scored the same way on the same nine labels (`report/tables/shared.md` as it then was; superseded by the 03:30–03:55 entry): CRF 0.842, rules 0.828, XLM-R 0.768, MuRIL at the fixed settings 0.758.
- **Decided (provisional; `initials.md` §12, §13):** the L1 model and prompt above.
- **Problems:**
  - The examples must come from the real-train pool, and the prompt may be changed on real dev only. Both wait for data.
  - The prompt was reworded once on hand20 ("one JSON object") after the no-example run answered with bare lists. The first version's files were overwritten and are not cited.
  - M1's speed at inference is not recorded yet, so E6's speed comparison has only L1's side.
  - Generation stops at 200 new tokens; a very long address could be cut off.
- **Tests:** 236 passing, 91 % coverage.
- **Next:** all 16 build steps now exist in code. What is left needs real data: send the form, annotate, then `scripts/run_experiments.sh`, E1, E5 and E6 on real dev and test, and the report.

### 2026-11-11 (03:10–03:18): Claude Code, step 15 (demo)
- **Did (tests first):** `app/demo.py`. `parse(text, tagger, gaz)` returns the whole address as labelled pieces, the normaliser's JSON and plain notes about the PIN (missing, not in the directory, contradicts the written state or district, district and state filled in). Three taggers plug in: rules (starts at once), CRF (trained on `full` at start, about a minute; the default) and a saved transformer folder. The page is Gradio, as the plan says; it stores nothing a visitor types and reports no analytics.
- **Did:** `src/models/transformer.py --save-dir` keeps the trained model. A test checks that the saved model reproduces exactly the tags the run was scored on.
- **Checked in the browser:** the page was started with the rules tagger and given the §2 example. It showed `flat 3b` HOUSE, `shanti apts` BUILDING, two landmarks, `supela bhilai` LOCALITY, `490023` PIN, the cleaned JSON, and "District and state filled in from the PIN: Durg, Chhattisgarh." No console errors.
- **Dependencies:** a new optional extra, `demo = ["gradio>=5"]` (Gradio 6.31.0, Apache-2.0, installed). `uv.lock` updated. `models/` is git-ignored so a saved model (about 1 GB) is never committed.
- **Tests:** 233 passing, 91 % coverage (`app/` is not in the coverage count).
- **Problems:** no MuRIL or XLM-R model has been saved yet, so `--model <folder>` is tested only with a tiny random model; the CRF option was tested with a small training file, not started in the browser. The page has no way to choose the model while it runs.
- **Next:** Step 14 (few-shot LLM).

### 2026-11-11 (03:05–03:10): Claude Code, step 16 (tables) and a PIN fix
- **Did (tests first):** `scripts/make_tables.py` writes five Markdown tables to `report/tables/` from `results/` alone:
  - `runs.md`: one row per results file.
  - `configs.md`: the seeds of one configuration averaged.
  - `shared.md` (E1): each existing system on the labels it shares with us, and our systems rescored on those labels from their saved tags. This closes the second half of C6.
  - `compare.md` (E2–E4): differences with a paired bootstrap. Every row says "not run yet".
  - `normaliser.md` (E5).
- **Did:** rules and CRF re-run under their existing run IDs so that they have per-address tags. Their scores are unchanged; the files gained `field_ci95` and `ci_unit`.
- **Fixed (tests first; commit `27381c5`):** the India Post file has a `Test Office` row at PIN 999999 and two army base post offices at 9000xx. The normaliser took 999999 as valid and filled in a district from it. `Gazetteer.lookup` now returns nothing for a PIN that does not start with 1 to 8. The gazetteer file and the generator are untouched, because dropping the rows would reshuffle the PIN split. No address in `full` uses those PINs, so no logged number changes.
- **Tests:** 228 passing, 91 % coverage.
- **Results (hand20, from `report/tables/shared.md` as it then was; a smoke check, superseded by the 03:30–03:55 entry):** on the eight labels the abr-6024 parser shares with us it scores 0.645 field micro F1. Scored the same way, our CRF gets 0.853, rules 0.828, MuRIL at the fixed settings 0.781 and XLM-R 0.777.
- **Problems:** no figures (matplotlib is not installed). `shared.md` lists the tuning runs among our systems. The three rows whose PIN starts with 9 are still in the gazetteer and the generator could sample them; drop them the next time the synthetic sets are regenerated.
- **Next:** Step 15 (demo), step 14 (few-shot LLM).

### 2026-11-11 (02:55–03:05): Claude Code, step 12 (experiment loop and statistics)
- **Did (tests first), statistics in `src/eval/`:**
  - The bootstrap resamples whole respondents when records carry `meta.respondent` (real data), and single addresses otherwise. A test pins the address-level numbers, so the intervals already logged do not move.
  - `paired_bootstrap` gives the difference between two systems on the same addresses with a 95 % interval, for span F1 or field F1.
  - Every results file gains `field_ci95`, `ci_unit` and `span.macro_min30` (macro F1 over the labels with at least 30 gold spans).
- **Did:** every model script now saves its tags per address (`<run-id>.pred.jsonl`) through one helper, rules and CRF included, so any two systems can be compared and any system rescored on fewer labels. The trainer refuses to train on an address that is also in an eval file (rule 2).
- **Did:** `src/mix.py` (E4). It takes the first k addresses of the real-train pool after a seeded shuffle, so the subsets are nested within a seed, and repeats them to about a tenth of the training file.
- **Did:** `scripts/run_experiments.sh`. Loops for E2 (6 runs), E3 (24), E4 (12, plus 3 on 40k synthetic) and XLM-R (3): 48 in all. `--dry-run` lists them. The real test set is scored only with `--final`. It refuses to start without `data/real/dev.jsonl`, and it skips a run that is already finished.
- **Not run:** none of the 48. They need the real dev and test sets, and at about 8 minutes each they take a little over 6 hours.
- **Tests:** 218 passing, 91 % coverage.
- **Decided (provisional; row in `initials.md` §13):** the mixing rule as proposed; k = 0 is E2's full-noise run; the three seeds are training seeds over one synthetic set (data seed 42); run IDs `<date>-<model>-<train tag>-<syntest|realdev|realtest>-s<seed>`.
- **Problems:** the paired test and the rescoring on fewer labels exist as functions, but nothing calls them yet (step 16 will). Results files written before today have no `field_ci95` or `macro_min30`.
- **Next:** Step 16 (tables from `results/`), step 15 (demo), step 14 (few-shot LLM).

### 2026-11-11 (02:05–02:55): Claude Code, step 11 (existing parsers, no training) and the JSON path
- **Did (tests first):** `src/models/zeroshot.py` (Z, E1). It runs four open address taggers at pinned commits on the raw address text, as their model cards do, and reads each token's label by character offset. Their labels are mapped onto ours with the tables in `research.md` §2. A system is scored only on the labels it shares with us (gold spans of any other label are blanked). Stray commas and brackets at span ends are dropped by the rule human annotations get (`annotate.snap_span`). Each model's own tags are saved beside ours in `<run-id>.pred.jsonl`.
- **Did (tests first):** `src/models/llm.py`, the path for systems that answer in JSON. It runs the generative parser `abr-6024/indian-address-parser-1.5B` with the prompt from its model card, maps its keys onto our labels and scores at field level only. State names are written in full on both sides. Its `city` counts as right if it is our city or our district. The few-shot LLM (step 14) will reuse this path.
- **Left out:** `howdoiuse-keyboard/indian-address-parser-model` cannot be loaded with standard classes (no architecture in its config, pickle-only weights). `deepparse` is not run: it needs a new dependency and several GB of embeddings, and it was trained on other countries' addresses. The team should decide whether to drop it.
- **Review (mle-reviewer):** alignment, mappings and trimming are clean (four real tokenizers on adversarial strings; 20,000 fuzzed tag sequences). The tests were strengthened after it: they now fail if trimming is removed from `run`, if a mapping changes, if the wrong batch row is read, if the context length is ignored, or if the safe-weights flag is dropped.
- **Tests:** 202 passing, 91 % coverage.
- **Results:** the five rows below, all on hand20 (20 items: a check of the plumbing, never a result).

  | System | Shared labels | Field micro F1 | Span micro F1 |
  |---|---|---|---|
  | abr-6024 parser (JSON) | 8 | 0.645 | – |
  | Shiprocket IndicBERT | 8 | 0.591 | 0.588 |
  | Shiprocket TinyBERT | 8 | 0.480 | 0.469 |
  | mukuls9971 MuRIL | 9 | 0.420 | 0.418 |
  | gagan1985 TinyBERT | 8 | 0.386 | 0.386 |

  For scale, on all nine labels (span micro F1): rules 0.828, CRF 0.796, our XLM-R 0.759, our MuRIL 0.737. The label sets differ, so this is not yet a comparison.
- **Problems (each is a scoring decision for real dev; build-plan B14, C6; `initials.md` §12):**
  - *Commas.* Before trimming, the Shiprocket models scored 0.245 and 0.187 span F1, because they tag `Bhilai,` with its comma. Those first numbers were not kept as files; do not cite them.
  - *The place's own name.* Our gold leaves `Shree Sweets` untagged; these models tag it as a building or a house name. The reviewer measured, in a scratch run, that ignoring those predictions lifts the four taggers by 0.04 to 0.07 F1. Real data carries the name in `meta.place`, so it can be handled once we see how often people repeat it (build-plan C1).
  - *The parser's conventions.* It answers `4` where gold has `shop no 4`, and `Ganesh Temple` where gold has `opp. ganesh temple`. Its HOUSE and LANDMARK therefore score 0.00 at field level although the content is right.
  - *Same-label comparison.* Our own models are not yet rescored on each system's shared labels (the second half of C6). That belongs with the tables (step 16).
  - *Licences.* The hub metadata states none for Shiprocket IndicBERT (its card says Apache-2.0) and none for the mukuls9971 model.
- **Downloads:** 1.4 GB for the four taggers and 3.1 GB for the parser, all safetensors, cached in `~/.cache/huggingface`.
- **Next:** Step 12 (experiment loop; paired and respondent-level bootstrap).

### 2026-11-11 (01:40–01:55): Claude Code, first XLM-R run
- **Did:** The tuning work is committed (`19da262`). Downloaded `FacebookAI/xlm-roberta-base` at the pinned commit (1.1 GB) and checked its tokenizer on our tokens: nothing is dropped and an address has at most 93 subwords. A comma becomes a bare `▁` followed by `,`, so the tag of a punctuation token is read from the bare `▁`.
- **Did:** One XLM-R run on `full` (20k, seed 42) at the fixed settings, through the command line.
- **Results:** the two `xlmr` rows below. Synthetic dev 0.991 micro F1 (MuRIL 0.990, CRF 0.992; the three intervals overlap). hand20 0.759 (MuRIL 0.737, CRF 0.796; intervals overlap, n = 20). Mean loss per epoch 0.249, 0.015, 0.005: it trains cleanly at 1e-4, and much faster than MuRIL (1.475, 0.142, 0.054).
- **Decided (provisional; row in `initials.md` §13):** M2 reuses the M1 settings unchanged.
- **Timing:** 361 s to train, 8.5 minutes end to end.
- **Problems:**
  - GPU memory is tight for XLM-R: the card peaked at 7,194 MiB of 8,188 (about 540 MiB of that was in use before the run). Another GPU program or much longer addresses could make a run fail. Halving the batch would fix it, but that changes a fixed hyperparameter and needs a decision.
  - One seed. On synthetic dev the three systems cannot be told apart, so only real data can rank them.
- **Next:** Step 11 (zero-shot E1). Send the form.

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
| 20261111-xlmr-synfull-dev-s42 | 2026-11-11 | M2 XLM-R (`FacebookAI/xlm-roberta-base` at commit `e73636d`; fixed settings: 3 epochs, lr 1e-4, batch 32, mixed precision) | synthetic `full`, 20k | 42 | synthetic `full` dev (2,000), not real dev: 0.991 / 0.991 (CI95 0.989–0.993) | – | `results/20261111-xlmr-synfull-dev-s42.json` | exact match 0.95; CITY 0.97, LOCALITY 0.96, every other label 0.996 or more; trained in 361 s; mean loss per epoch 0.249 / 0.015 / 0.005; GPU peak 7,194 MiB of 8,188; tags per address in `results/20261111-xlmr-synfull-dev-s42.pred.jsonl` |
| 20261111-xlmr-synfull-hand20-s42 | 2026-11-11 | M2 XLM-R (same model) | synthetic `full`, 20k | 42 | hand20 smoke set, not real dev: 0.759 / 0.677 (CI95 0.677–0.840) | – | `results/20261111-xlmr-synfull-hand20-s42.json` | exact match 0.35; PIN 1.00, CITY 0.82, STATE 0.80, STREET 0.80, LANDMARK 0.67, LOCALITY 0.61, HOUSE 0.50, BUILDING 0.50, DISTRICT 0.40; between MuRIL (0.737) and the CRF (0.796), intervals overlap |
| 20261111-shiprocket-indicbert-hand20-s0 | 2026-11-11 | Z Shiprocket IndicBERT, no training (commit `4e5d716`; 8 shared labels, no DISTRICT) | none (no training) | 0 | hand20 smoke set, not real dev: 0.588 / 0.623 (CI95 0.489–0.684) | – | `results/20261111-shiprocket-indicbert-hand20-s0.json` | field micro F1 0.591; PIN 0.93, HOUSE 0.89, STREET 0.82, LANDMARK 0.71, STATE 0.63, CITY 0.50, LOCALITY 0.40, BUILDING 0.10; shared labels only, commas trimmed |
| 20261111-shiprocket-tinybert-hand20-s0 | 2026-11-11 | Z Shiprocket TinyBERT, no training (commit `fc11437`; 8 shared labels, no DISTRICT) | none (no training) | 0 | hand20 smoke set, not real dev: 0.469 / 0.536 (CI95 0.365–0.579) | – | `results/20261111-shiprocket-tinybert-hand20-s0.json` | field micro F1 0.480; HOUSE 1.00, PIN 0.90, LANDMARK 0.73, STREET 0.57, STATE 0.38, LOCALITY 0.33, CITY 0.21, BUILDING 0.17; shared labels only, commas trimmed |
| 20261111-mukuls-muril-hand20-s0 | 2026-11-11 | Z mukuls9971 MuRIL address NER, no training (commit `6cdb861`; all 9 labels) | none (no training) | 0 | hand20 smoke set, not real dev: 0.418 / 0.325 (CI95 0.355–0.500) | – | `results/20261111-mukuls-muril-hand20-s0.json` | field micro F1 0.420; PIN 0.80, STATE 0.47, STREET 0.47, LOCALITY 0.40, CITY 0.33, BUILDING 0.25, HOUSE 0.20, LANDMARK 0.00, DISTRICT 0.00; commas trimmed |
| 20261111-gagan-tinybert-hand20-s0 | 2026-11-11 | Z gagan1985 TinyBERT, no training (commit `20d532b`; 8 shared labels, no BUILDING) | none (no training) | 0 | hand20 smoke set, not real dev: 0.386 / 0.322 (CI95 0.316–0.466) | – | `results/20261111-gagan-tinybert-hand20-s0.json` | field micro F1 0.386; PIN 0.85, STREET 0.56, LOCALITY 0.41, CITY 0.31, LANDMARK 0.24, STATE 0.13, HOUSE 0.07, DISTRICT 0.00; shared labels only |
| 20261111-abr-parser-hand20-s0 | 2026-11-11 | Z abr-6024 generative parser 1.5B, no training (commit `483951f`; JSON output, 8 shared labels) | none (no training) | 0 | hand20 smoke set, not real dev: no span F1 (JSON output); field micro F1 0.645 (P 0.641, R 0.648) | – | `results/20261111-abr-parser-hand20-s0.json` | valid JSON 20 of 20; field F1: PIN 0.90, CITY 0.85, STATE 0.67, LOCALITY 0.63, STREET 0.50, BUILDING 0.33, HOUSE 0.00, LANDMARK 0.00 (it answers `4` for `shop no 4` and drops the relation word); raw answers in `results/20261111-abr-parser-hand20-s0.pred.jsonl` |
| 20261111-l1-qwen-syn5shot-hand20-s42 | 2026-11-11 | L1 few-shot LLM (`Qwen/Qwen2.5-1.5B-Instruct` at commit `989aa79`; 5 examples from synthetic `full` train, seed 42) | none (no training) | 42 | hand20 smoke set, not real dev: no span F1 (JSON output); field micro F1 0.484 (P 0.430, R 0.553) | – | `results/20261111-l1-qwen-syn5shot-hand20-s42.json` | valid JSON 20 of 20; 11.1 s for 20 addresses; PIN 0.86, STREET 0.65, CITY 0.65, LANDMARK 0.43, BUILDING 0.33, STATE 0.31, LOCALITY 0.29, DISTRICT 0.17, HOUSE 0.00; the examples are synthetic stand-ins for the real-train pool |
| 20261111-l1-qwen-0shot-hand20-s42 | 2026-11-11 | L1 LLM with no examples (same model and prompt) | none (no training) | 42 | hand20 smoke set, not real dev: no span F1 (JSON output); field micro F1 0.288 (P 0.373, R 0.234) | – | `results/20261111-l1-qwen-0shot-hand20-s42.json` | valid JSON 9 of 20; 6.7 s for 20 addresses; PIN 0.55, STREET 0.50, CITY 0.48, BUILDING 0.25, LOCALITY 0.19, LANDMARK 0.12, HOUSE, DISTRICT and STATE 0.00 |
