# Initial notes: Near Hanuman Mandir, Parsing Messy Indian Addresses

Group 10 · DSL504 / DS504 NLP · IIT Bhilai · Jul–Nov 2026
Team: Agastya Nath · Y. Rahul Dev Reddy · Paritosh Lahre · Varun Rao
Created 10 Nov 2026 · Final deadline: **____ (fill in)**

## 1. Problem
Indian addresses rarely follow a fixed format. They lean on landmarks, abbreviations, mixed scripts, transliterated words and free ordering, often with missing commas:

    opp. ganesh temple, 2nd cross, behind old bus stand, hosur road, madivala, b'lore-68

Delivery, banking KYC and emergency services all need these as structured fields. India Post's DIGIPIN (2025) gives every ~4 m × 4 m grid cell a 10-character code, but people will keep writing free-text addresses with landmarks, so turning text into structured fields still matters.

## 2. What we build
1. **Tagger:** labels each token with an address field.
2. **Normaliser:** cleans the fields and checks them against the India Post PIN directory.
3. **Demo:** paste an address, get coloured tags, clean JSON and warnings.

Example (illustrative):

    Input:  flat 3b, shanti apts, behind SBI atm nr hanuman mandir, supela bhilai 490023
    Tags:   HOUSE=flat 3b · BUILDING=shanti apts · LANDMARK=behind SBI atm · LANDMARK=nr hanuman mandir
            LOCALITY=supela · CITY=bhilai · PIN=490023

```json
{"house": "Flat 3B", "building": "Shanti Apartments",
 "landmarks": [{"relation": "behind", "anchor": "SBI ATM"},
               {"relation": "near", "anchor": "Hanuman Mandir"}],
 "locality": "Supela", "city": "Bhilai", "district": "Durg", "state": "Chhattisgarh",
 "pin": "490023", "pin_check": "valid | not_found | mismatch"}
```
District and state are filled from the PIN directory when the address leaves them out. Check that 490023 really belongs to Supela before using this example on a slide.

## 3. Research questions
- **RQ1 (the gap):** how much do parsers trained on synthetic data, both ours and existing open models, lose on real human-written addresses?
- **RQ2 (which noise helps):** which kinds of synthetic noise (§5.2) close the gap most?
- **RQ3 (real data):** how much does adding a few real labelled addresses (0 / 25 / 50 / 100 / 200) help, compared with more synthetic data?
- **Secondary:** how many errors does PIN-directory normalisation catch or fix?

**Hypotheses**
- **H1:** everything trained only on clean synthetic data drops sharply on real text, worst on STREET, LOCALITY and LANDMARK.
- **H2:** landmark and abbreviation noise give the biggest gains; transliteration noise matters mostly for code-mixed addresses.
- **H3:** about 100 real examples help more than doubling the synthetic data.

**Why this angle:** the closest paper (Dhavase & Chaudhary, ETASR 2026) has already shown the gap, and it patched it with retrieval at prediction time. That paper has no LANDMARK label, its training data has no noise, and its "real" test set was generated from OpenStreetMap by a script. We fix the training data instead, test on genuinely human-written text, and treat landmarks as a full field. Details in `research.md`.

## 4. Label schema (v0.1)
| Label | Covers | Examples |
|---|---|---|
| HOUSE | house / flat / plot / door / shop number, floor | `Flat 3B`, `H.No. 12/3`, `D.No 5-6-7/A`, `#12`, `2nd floor`, `Shop No. 4`, `Plot 12` |
| BUILDING | apartment, society, complex, named building | `Shanti Apts`, `Sunshine Enclave`, `Tower C` |
| STREET | road, street, lane, cross, main, highway | `MG Road`, `2nd cross`, `3rd main`, `Gali No. 4`, `NH 44` |
| LANDMARK | relation word + the landmark itself | `near Hanuman Mandir`, `opp. petrol pump`, `SBI ATM ke paas` |
| LOCALITY | area, colony, nagar, sector, layout, village, sub-locality | `Supela`, `Sector 21`, `Gandhi Nagar`, `Koramangala 4th Block` |
| CITY | city or town | `Bhilai`, `B'lore`, `Hyd` |
| DISTRICT | district, only next to a cue word (`Dist`, `Distt`, `District`, `Zila`), before or after | `Durg` in `Distt. Durg` or `Durg Dist`; a bare `Durg` is CITY |
| STATE | state or UT, including short forms | `Chhattisgarh`, `C.G.`, `MH` |
| PIN | 6-digit PIN, spaced or partial | `490023`, `490 023`, `68` in `b'lore-68` |

**Span rules**
- **Inside the span:** words that are part of how the field is written. `near` / `opp.` / `behind` belong to LANDMARK, `Flat` / `H.No.` / `#` to HOUSE, `Road` / `Rd` / `Cross` / `Gali` to STREET, `Nagar` / `Sector` / `Layout` to LOCALITY, `Apts` / `Society` to BUILDING.
- **Tag as O:** words that only announce a field (`Pin:`, `Pincode`, `Dist.`, `Distt.`, `State:`), filler (`c/o`, `India`), punctuation, and any names or phone numbers.
- **The place's own name is O** (`Shree Sweets`, `IIT Bhilai`): the form asks for it separately and the generator prefixes one in every config, so it is not a difference between synthetic and real data.
- **Each relation word opens a new LANDMARK span** (`behind SBI atm` / `nr hanuman mandir`); a landmark with no relation word is still LANDMARK.
- **Sub-locality** is merged into LOCALITY, because the boundary between them is too fuzzy to annotate consistently.
- **Block / Tower** inside a society is BUILDING; a city sector such as `4th Block` is LOCALITY.
- Record every new edge case in `data/real/GUIDELINES.md` (v1 written 10 Nov) and in the §13 decision log.

## 5. Data plan
### 5.1 Gazetteer
The India Post **All India Pincode Directory** (data.gov.in, GODL-India licence; use the GitHub mirror if the portal is down). It maps each PIN to its post office or locality, district and state. Three parts use it:
- the generator, as real geography;
- the rules and CRF baselines, as features;
- the normaliser, for checking and filling fields.

### 5.2 Synthetic generator
The generator takes real geography (the PIN directory, plus OSM street, building and POI names for 3–5 cities including Bhilai / Durg / Raipur) and fills address templates with it. Labels come free because the generator knows every span. Each noise family is a switch, so it can be ablated:

| ID | Noise family | Examples |
|---|---|---|
| N1 | Abbreviations | `opp.`, `nr.`, `bh.`, `apts`, `rd`, `sec.`, `B'lore`, `Hyd` |
| N2 | Landmark phrases | near / opp / behind / beside + temple, ATM, petrol pump, bus stand; `ke paas`, `ke saamne` |
| N3 | Reordering and dropped fields | state before city; no PIN; no state |
| N4 | Delimiters and case | no commas, ALL CAPS, all lowercase, `-` and `/` separators |
| N5 | Typos | inserted, deleted, substituted or swapped characters (edit operations) |
| N6 | Script and transliteration | some words in Devanagari; spelling variants (mandir / mandeer, gali / galli) |
| N7 | PIN variants | `490 023`, `bhilai-23`, PIN missing |
| N8 | Indian number formats | `D.No 5-6-7/A`, `H.No 12/3`, `#12, 3rd main` |

**Clean config** = all nine fields, one fixed order, commas, full words, one plain `near X` landmark, prefixed by a place name tagged O. Each transformation belongs to exactly one family (so N7 owns everything about PINs, N1 owns `B'lore`, N6 owns `ke paas` spelling variants). PINs are split 90/5/5 into train/dev/test pools with a fixed seed, so synthetic test addresses always use held-out PINs.

Target: about 20k train + 2k dev + 2k test addresses per generator config.

### 5.3 Real, human-written data (the most important asset)
- **Collect:** a Google Form asks classmates and friends to write the address of a **public place** the way they would for a courier, in free text and any style: a shop, temple, college, hospital or office. **No home addresses, names or phone numbers.**
- **Target:** about 500 addresses: 250 test, 50 dev and a 200-address real-train pool for RQ3. The minimum workable set is 200 test, 50 dev and a 50-address pool.
- **Annotate:** in Label Studio or doccano, following `GUIDELINES.md`. Two people label the same 50 items first and report agreement (span-level F1 or Cohen's κ). Fix the guidelines, then label the rest.
- **Slice tags:** mark each test item with its traits (has a landmark, code-mixed, Devanagari, no PIN, ALL CAPS, …) so results can be broken down by trait.

### 5.4 Secondary test set
Use the 800-item test split of `abr-6024/indian-address-parsing-synthetic`, which includes held-out districts (`research.md` §3). Its text is synthetic, so use it for field-level checks only, never as a headline result.

## 6. Models
| ID | Model | Notes |
|---|---|---|
| B0 | Rules + gazetteer | Regex for PIN, cue-word lists, place-name lookup |
| B1 | CRF (`sklearn-crfsuite`) | Word shape, suffixes, cue words, gazetteer features |
| B2 | BiLSTM-CRF (optional) | Character + word embeddings; links to the RNN lectures |
| M1 | MuRIL, fine-tuned | `google/muril-base-cased`; the main model; 3 seeds |
| M2 | XLM-R, fine-tuned | `FacebookAI/xlm-roberta-base`; the best baseline in the closest paper |
| L1 | Few-shot LLM → JSON | Small open model that runs on a free T4 (e.g. Qwen2.5-3B-Instruct) |
| Z | Existing open parsers, zero-shot | Shiprocket IndicBERT/TinyBERT, MuRIL address NER, abr-6024 1.5B, deepparse (`research.md` §2) |

The zero-shot systems use different label sets. Map them onto ours and compare shared fields only; the mapping table is in `research.md` §2.

## 7. Experiments
| ID | Answers | Setup |
|---|---|---|
| E1 | RQ1 | All Z systems + B0 on the real test set (field level) |
| E2 | RQ1 | M1 trained on clean vs full-noise synthetic data; scored on synthetic test and real test |
| E3 | RQ2 | Full noise minus one family at a time (N1–N8) with M1; use B1 if GPU time runs short |
| E4 | RQ3 | Synthetic + k real addresses (k = 0 / 25 / 50 / 100 / 200) with M1 |
| E5 | Normaliser | PIN / district / state errors caught and fixed on the real test set |
| E6 | LLM | L1 vs M1: F1, speed and cost |

## 8. Evaluation
- **Main metric:** entity-level micro and macro F1 (`seqeval`, strict IOB2), reported per label.
- **Other metrics:** whole-address exact match; **field-level P / R / F1**: per field, compare the *sets* of predicted and gold strings after lower-casing and stripping punctuation, summed over addresses, reported on shared fields only (state names canonicalised on both sides for JSON-output systems).
- **Breakdowns:** results per slice (§5.3) and F1 vs noise level.
- **Stats:** 3 seeds (mean ± std) plus a bootstrap 95% CI on the test set.
- **Normaliser:** PIN validity, PIN–locality consistency (`pin_check` passes if the locality matches any office under that PIN), PIN–district consistency, and accuracy of filled-in district / state.

## 9. Deliverables
- Report, slides and a demo (paste an address, get tags, JSON and PIN warnings).
- The code, plus a release of the synthetic generator and the real test set (public places only).
- Updated class spreadsheet entry (§14).
- The report should tie each part to the syllabus: edit distance (typos and fuzzy matching), tokenisation and subword alignment, BiLSTM, transformers / BERT fine-tuning, and in-context learning (the LLM baseline).

## 10. Plan (about 3 weeks; compress if the deadline is earlier)
| Phase | Days | Work | Done when |
|---|---|---|---|
| 1 | 1–5 | Repo and environment; clean the PIN directory; guidelines v1; send the Google Form; generator v1 (no noise); rules baseline; eval script | Eval script runs on 20 hand-labelled items |
| 2 | 5–10 | Annotate real data (double-annotate a subset first); noise families N1–N8; CRF; MuRIL on full noise; E1 zero-shot | First real-dev numbers are logged |
| 3 | 10–15 | Ablations E2–E4; normaliser and E5; LLM (E6) | Results tables are filled from `results/` |
| 4 | 15–20 | Error analysis; demo; report; slides | All four members have reviewed the report draft |

## 11. Suggested roles (adjust freely)
- **Data lead:** form, guidelines, annotation, agreement check, freezing the test set.
- **Generator lead:** gazetteer, generator, noise families.
- **Modelling lead:** CRF, MuRIL / XLM-R, LLM, seeds, GPU time.
- **Eval & demo lead:** eval scripts, normaliser, demo, figures.
- **Everyone:** annotation, report, slides.

## 12. Open decisions
- [ ] Final deadline and presentation format.
- [x] Script scope (provisional, 10 Nov): core = Roman script and Hinglish; Devanagari items are kept in the test set as a slice; N6 is a spelling-variant table, not a transliteration model.
- [ ] Which 3–5 cities do we take OSM names from?
- [ ] Annotation tool: Label Studio or doccano?
- [ ] Which LLM for L1 (it must run on a free T4)?
- [x] DISTRICT stays (10 Nov): tagged only next to a cue word; otherwise CITY.
- [ ] Mixing rule for E4 (proposed: upsample the k real addresses to about 10% of the training set; nested subsets 25 ⊂ 50 ⊂ 100 ⊂ 200). Confirm before the first E4 run.

## 13. Decision log
| Date | Decision | Why |
|---|---|---|
| 10 Nov 2026 | Project chosen: #8 from the shortlist (Indian address parsing). | Group decision. |
| 10 Nov 2026 | Research focus = synthetic→real gap: noise ablation, real-data curve, human-written test set. | Prior work has already shown the gap; this is what it leaves open (`research.md` §1). |
| 10 Nov 2026 | Label schema v0.1 (§4). LANDMARK spans include the relation word; sub-locality is merged into LOCALITY. | Simpler, consistent annotation; the normaliser splits relation from landmark. |
| 10 Nov 2026 | Real data = public-place addresses only. | Privacy. |
| 10 Nov 2026 | *Provisional, made during the Phase 1 build; team to confirm or overturn.* One shared tokenizer: whitespace + every punctuation character its own token (`src.data.tokenize`). | Generator, annotation export, CRF and eval must agree on tokens; a `\w+` regex shreds Devanagari (build-plan A3). |
| 10 Nov 2026 | The place's own name is tagged O and the generator prefixes one in every config. | Otherwise the real test set differs from training in a way that is not noise (A2). |
| 10 Nov 2026 | DISTRICT kept; only next to a cue word. Each relation word opens a new LANDMARK span. | A5, B15. |
| 10 Nov 2026 | CITY comes from the PIN directory's `DivisionName` (minus "Division") until OSM names are added; 36-row canonical state table; `pin_check` = locality matches any office under the PIN. | The directory has no city column (A4). |
| 10 Nov 2026 | Core script scope = Roman + Hinglish; N6 = spelling variants. | IndicXlit needs fairseq, which does not install on current Python (A6). |
| 10 Nov 2026 | Clean config defined (all fields, fixed order, commas, full words, one `near X`); PIN split 90/5/5 with a fixed seed. | B2, B8. |
| 10 Nov 2026 | Field-level metric = set P/R/F1 over normalised strings on shared fields. | B9. |
| 10 Nov 2026 | Python 3.12 via `uv`; training in scripts, notebooks only call them. | Colab parity; Rule 5. |

## 14. Class spreadsheet entry (paste into Group 10's row: sheet row 51)
**Title:** Near Hanuman Mandir: Parsing and Normalising Messy Indian Addresses

**Description:** Indian addresses mix landmarks ("near Hanuman Mandir"), abbreviations, transliterated words and free ordering, so parsers trained on clean synthetic data fail on real text. We tag address components (house, building, street, landmark, locality, city, district, state, PIN) and normalise them against the India Post PIN directory. We build a noise-controlled synthetic data generator and a new human-written test set of public-place addresses, benchmark existing open parsers, compare CRF, MuRIL/XLM-R and few-shot LLMs, and measure which noise types and how many real examples close the synthetic-to-real gap.
