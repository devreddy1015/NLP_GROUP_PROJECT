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
