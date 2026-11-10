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

| ID | Noise family | Transformations (each belongs to this family only; `src/generate/noise.py`) |
|---|---|---|
| N1 | Abbreviations | word table (`Road→Rd`, `Apartments→Apts`, `Sector→Sec`, `near→nr`, `opposite→Opp`, `District→Distt`, `Government→Govt`, …); city short forms (`B'lore`, `Hyd`, `Bom`, `Kol`); state codes (`CG`, `C.G.`, `MH`) |
| N2 | Landmark phrases | relation variety (`opposite / behind / beside / next to / in front of / adjacent to / close to / back side of`); Hinglish postpositions (`X ke paas / ke samne / ke peeche / ke bagal me`); 0, 1 or 2 landmarks; two in one breath (`behind X near Y`) |
| N3 | Reordering and dropped fields | drop house / building / street / district / city / locality / state (p = 0.2 each; never the PIN, never the place name); landmarks first, city ↔ locality, street ↔ building, district before city, or a full shuffle |
| N4 | Delimiters and case | separator `, ` / space / ` - ` / `,` / ` / `; all lower / ALL CAPS / Title Case |
| N5 | Typos | one or two character edits (insert, delete, substitute, swap) in words of 4+ letters; never in the PIN |
| N6 | Spelling variants (transliteration) | `Mandir→Mandeer`, `Gali→Galli`, `Chowk→Chauk`, `Road→Rod`, `Station→Steshan`, `Hospital→Haspatal`, `Bhavan→Bhawan`, `ke paas→ke pass`, … No Devanagari (§12) |
| N7 | PIN variants | `490 023`, `490-023`, `Pin: / Pincode 490023`, `State - 490023`, PIN before the state, `Bhilai-23`, PIN missing |
| N8 | Indian number formats | house as `#12`, `H.No 12/3`, `D.No 5-6-7/A`, `12-3-45`, `Door No 12`, `Flat No. 3B`, `Plot No 7/2`, `Khasra No 12`, … |

Families apply in the order N2, N3, N8, N7, N1, N6, N5, N4 (structure, then wording, then typos, then case and delimiters). Configs: `clean`, `full` (all eight), `full-minus-Nx` (E3), `only-Nx` (tests). Each record's `noise` field lists the families that actually changed it; on `full` each family fires on 50–98 % of addresses and every label survives in about 80 % of them.

**Clean config** = all nine fields, one fixed order, commas, full words, one plain `near X` landmark, prefixed by a place name tagged O. Each transformation belongs to exactly one family (so N7 owns everything about PINs, N1 owns `B'lore`, N6 owns `ke paas` spelling variants). PINs are split 90/5/5 into train/dev/test pools with a fixed seed, so synthetic test addresses always use held-out PINs.

**Geography.** City = the postal division's town with direction words removed (`Mumbai City North West` → `Mumbai`), or the town that office names end in (`Supela Bhilai` → locality `Supela`, city `Bhilai`). Half the named streets take a real place name (`Junwani Road`); the rest use the fixed list. Place, landmark and building names are still short hand-written lists until OSM names are added (§12).

Target: about 20k train + 2k dev + 2k test addresses per generator config.

### 5.3 Real, human-written data (the most important asset)
- **Collect:** a Google Form asks classmates and friends to write the address of a **public place** the way they would for a courier, in free text and any style: a shop, temple, college, hospital or office. **No home addresses, names or phone numbers.**
- **Form fields, in this order:** the place's name; the free-text address; then its PIN, city and state in separate boxes. The last three are the true values E5 scores against and go into `meta`. The form collects no email; each response gets a `respondent` code. Ask each person for five places in different towns, one of them in their home town, so the set is not all Bhilai, Durg and Raipur. Report the spread by state.
- **Target:** about 500 addresses: 250 test, 50 dev and a 200-address real-train pool for RQ3. The minimum workable set is 200 test, 50 dev and a 50-address pool.
- **Split:** by respondent, never by row, after dropping duplicate places, so one person's style and one place never sit on both sides (`src/annotate.py tasks`).
- **Protect the test set:** split before anyone reads the addresses, and annotate dev and the pool first. Tag the generator in git before the test items are opened; after that, a change to its word lists or templates must cite a dev example.
- **Annotate:** in Label Studio, following `GUIDELINES.md`. Two people label the same 50 items first and report agreement (span-level F1 and token-level Cohen's κ). Fix the guidelines, then label the rest.
- **Slice tags:** mark each test item with its traits (has a landmark, code-mixed, Devanagari, no PIN, ALL CAPS, …) so results can be broken down by trait.
- **Thin labels:** public places rarely have `Flat` or `H.No.`, so HOUSE and BUILDING may be rare. Report the span count per label and make no claim about a label with fewer than about 30 test spans.

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
| E1 | RQ1 | All Z systems + B0 on the real test set (field level). Fix each label mapping on real dev first, then run the test set once |
| E2 | RQ1 | M1 trained on clean vs full-noise synthetic data; scored on synthetic test and real test |
| E3 | RQ2 | Full noise minus one family at a time (N1–N8) with M1, 3 seeds per config; use B1 if GPU time runs short |
| E4 | RQ3 | Synthetic + k real addresses (k = 0 / 25 / 50 / 100 / 200) with M1, 3 seeds each, nested subsets (mixing rule: §12). Plus one run on 40k synthetic with k = 0, the "double the synthetic data" side of H3 |
| E5 | Normaliser | PIN / district / state errors caught and fixed on the real test set, scored against the form's PIN, city and state (`meta`); plus wrong PINs swapped in at a known rate (§8) |
| E6 | LLM | L1 vs M1: field-level F1 (L1 outputs JSON, not tags), speed and cost |

Hyperparameters and the number of epochs are fixed once, on the full-noise config, and reused for every M1 and M2 run. No run picks a checkpoint on real dev, so the k = 0 runs train on no real address. Real dev can still shape the generator (§5.3); the report must say so.

## 8. Evaluation
- **Main metric:** entity-level micro and macro F1 (`seqeval`, strict IOB2), reported per label.
- **Other metrics:** whole-address exact match; **field-level P / R / F1**: per field, compare the *sets* of predicted and gold strings after lower-casing and stripping punctuation, summed over addresses, reported on shared fields only (state names canonicalised on both sides for JSON-output systems).
- **Breakdowns:** results per slice (§5.3) and F1 vs noise level.
- **Stats:** 3 seeds (mean ± std) plus a bootstrap 95% CI on the test set. Differences between two systems use a paired bootstrap over the same test items (to be added to `src/eval/` with step 12); a difference whose interval includes zero is reported as "no measurable effect". On real data both bootstraps resample respondents, not single addresses: one person's five addresses are not independent. Macro F1 is taken over the labels with at least 30 test spans, the same set for every system.
- **Normaliser:** PIN validity, PIN–locality consistency (`pin_check` passes if the locality matches any office under that PIN), PIN–district consistency (any district the PIN covers; a misspelt name is not a mismatch), and accuracy of the filled-in state. Filled districts are scored on synthetic data only, because the form has no district box. The same person types the address and the PIN, so real wrong-PIN cases will be rare: E5 also swaps in a wrong PIN at a known rate and reports how many `pin_check` catches and how many correct addresses it flags by mistake.

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
- [ ] Which 3–5 cities do we take OSM names from? Needed before E2–E4: place, landmark and building names are still hand-written lists of 20–40 items (build-plan C2). City and street names now come from the PIN directory.
- [ ] Place-name prefix rate. Every synthetic address now starts with a place name, but the form asks for the name separately, so many real addresses will not repeat it. Proposed: no change until real dev shows how often the name is repeated (17 of the 20 hand-written items have it), then prefix at that rate (build-plan C1).
- [ ] E3: add clean + one family runs next to leave-one-out? 24 more runs (build-plan C3).
- [ ] E4 base config (proposed: `full`) and whether the k-subset changes with the seed (proposed: yes, nested within each seed).
- [x] Annotation tool (10 Nov): Label Studio; config in `data/real/labelstudio_config.xml`, import/export/agreement in `src/annotate.py`.
- [x] LLM for L1 (provisional, 11 Nov): `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 3.1 GB; it runs on a T4 and on the 4060), the base model of the abr-6024 parser; system `l1-qwen` in `src/models/llm.py`. Its few-shot examples come from the real-train pool, never from dev or test.
- [ ] L1: how many examples, and the final prompt wording. Fix both on real dev; the smoke run used five synthetic examples.
- [x] DISTRICT stays (10 Nov): tagged only next to a cue word; otherwise CITY.
- [ ] Mixing rule for E4 (proposed: upsample the k real addresses to about 10% of the training set; nested subsets 25 ⊂ 50 ⊂ 100 ⊂ 200). Built as proposed in `src/mix.py` (11 Nov); confirm before the first E4 run.
- [x] M1 hyperparameters (provisional, 11 Nov): 3 epochs, learning rate 1e-4, batch 32, mixed precision; fixed on synthetic dev with one seed (`logs.md`, §13).
- [ ] E1: the place's own name. The existing parsers tag `Shree Sweets` as a building or house name; our gold leaves it untagged, which costs them 0.04 to 0.07 F1 on the 20 smoke items. Ignore predictions that fall inside `meta.place`? Decide on real dev, with the place-name prefix rate above.
- [ ] E1 / E6: conventions for JSON systems. The abr-6024 parser answers `4` for `shop no 4` and `Ganesh Temple` for `opp. ganesh temple`, so HOUSE and LANDMARK score 0 at field level. Compare the number alone, and the landmark without its relation word? Decide on real dev (build-plan C6).
- [ ] Drop `deepparse` from the Z systems? It needs a new dependency and several GB of embeddings and was trained on other countries' addresses. `howdoiuse-keyboard` is already out: it cannot be loaded without its author's code.
- [x] M2 (XLM-R) reuses the M1 settings (provisional, 11 Nov): its first run at learning rate 1e-4 trains cleanly (synthetic dev 0.991; `logs.md`). GPU memory is tight: 7.2 of 8.2 GB.

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
| 10 Nov 2026 | Noise families N1–N8 defined as the transformation list in §5.2, one family per transformation; Devanagari out. | B1, A6. |
| 10 Nov 2026 | Normaliser: `pin_check` ∈ {valid, mismatch, not_found, missing}; mismatch only when the written state or district contradicts the directory; district / state are filled only on `valid`; an unknown locality is flagged (`locality_known`), not a mismatch. Abbreviation expansion is the inverse of the N1 table. | A4, B9; locality names are too varied to count as errors. |
| 10 Nov 2026 | Label Studio for annotation. The form export must carry a `respondent` code per row so whole respondents go to one split; sloppy spans are snapped to tokens and stray commas trimmed on export. | B3, B4; the form collects no email. |
| 10 Nov 2026 | *Provisional.* Form asks for five places in different towns per person and for PIN, city and state after the free text. The generator is tagged in git before the test items are opened. No claim on a label with fewer than about 30 test spans. | B3, B10, B11, B12. |
| 10 Nov 2026 | *Provisional.* Hyperparameters and epochs fixed once on full noise, no checkpoint picked on real dev; 3 seeds for every E3 and E4 config; a 40k-synthetic run added for H3; E1 label mappings fixed on dev; paired bootstrap for differences. | B5, B6, B13, B14. |
| 10 Nov 2026 | *Provisional, after the second review.* Respondent split kept in `data/real/splits.json`; test set filled before the pool. Any district a PIN covers is accepted and a district is filled only when the PIN has one; a misspelt state or district is not a mismatch (similarity cut-off 0.8, untuned). Real-data intervals resample respondents; macro F1 over labels with at least 30 test spans; E5 adds swapped-in wrong PINs. | build-plan C4, C5, C7–C11. |
| 10 Nov 2026 | *Provisional.* City names: division names lose trailing direction words; a town that several office names in a district end in becomes the city and leaves the locality. Half the named streets take a real place name. The place-name prefix stays on every synthetic address for now. | On 20 hand-written items the CRF tagged `Bhilai` as LOCALITY in 10 and `X Road` as LOCALITY in 6; 17 of the 20 start with the place name (build-plan C1, C2). |
| 11 Nov 2026 | *Provisional.* M1 / M2 script: one training run scores every eval file; model commits are pinned; a run ID must end in `-s<seed>`; the tags predicted for each address are saved next to the results file. The hyperparameters in the script are a first guess, not yet the fixed set (§12). | A GPU run does not repeat bit for bit, and the paired bootstrap needs both systems' tags. The first run was still improving after 3 epochs. |
| 11 Nov 2026 | *Provisional.* M1 hyperparameters fixed: 3 epochs, learning rate 1e-4, batch 32, linear warm-up over 10 % of steps, mixed precision (bf16) on a GPU. | Five MuRIL runs on `full`, seed 42, scored on synthetic dev. Rule set beforehand: the cheapest setting within 0.002 micro F1 of the best (0.9903 against 0.9910). Full precision gives the same score (0.9900) and takes 1.7 times as long. |
| 11 Nov 2026 | *Provisional.* M2 (XLM-R) reuses the M1 settings unchanged. | One XLM-R run on `full`, seed 42: mean loss per epoch 0.249, 0.015, 0.005 and synthetic dev 0.991, so it does not diverge at learning rate 1e-4. §7 asks for one fixed set for both models. |
| 11 Nov 2026 | *Provisional.* Z systems for E1: four taggers and the abr-6024 parser at pinned commits, safetensors weights only. Raw text goes in and labels are read by character offset. Each system is scored on the labels it shares with us; stray commas and brackets are trimmed from predicted spans as they are from human annotations; for JSON systems states are written in full, and the parser's `city` counts as our CITY or DISTRICT. Mappings as in `research.md` §2, to be fixed on real dev. | B9, B14, C6, and each model card. Untrimmed, two taggers lost more than half their span F1 to a comma habit. |
| 11 Nov 2026 | *Provisional.* Experiment loop (`scripts/run_experiments.sh`): 48 runs; the three seeds are training seeds over one synthetic set (data seed 42); k = 0 in E4 is E2's full-noise run; the real test set is scored only with `--final`. E4 mixing as proposed: nested seeded subsets, repeated to about a tenth of the file. Statistics: bootstraps resample respondents on real data; differences use a paired bootstrap; macro F1 over labels with at least 30 gold spans is stored beside the plain macro. | B7, B13, C4, C5, rule 2. |
| 11 Nov 2026 | *Provisional.* Demo: Gradio, run locally; CRF by default (no GPU), rules or a saved transformer on request; it stores nothing typed into it. | The plan named Gradio or Streamlit; Gradio has a highlighted-text view. Privacy (rule 1). |
| 11 Nov 2026 | *Provisional.* L1 = `Qwen/Qwen2.5-1.5B-Instruct`, few-shot, with a prompt in our nine labels that asks for text copied exactly as written; examples drawn with a seed from a file that must not overlap the eval set. | An open, ungated model that fits a free T4, and the base of the abr-6024 parser, so E6 compares few-shot against fine-tuned on one base. Verbatim copying keeps field scoring fair (C6). |
| 11 Nov 2026 | *Provisional, after the third review.* Only train-split records may be trained on or shown as few-shot examples; the code refuses anything else. In the tables a repeated run counts once (the newest). Our systems are compared with an existing one by that system's own scoring rules, with the city-or-district key counted once on both sides. A span of punctuation alone is not a field. | Rule 2; a `--final` run trains again; fairness in both directions (C6). |

## 14. Class spreadsheet entry (paste into Group 10's row: sheet row 51)
**Title:** Near Hanuman Mandir: Parsing and Normalising Messy Indian Addresses

**Description:** Indian addresses mix landmarks ("near Hanuman Mandir"), abbreviations, transliterated words and free ordering, so parsers trained on clean synthetic data fail on real text. We tag address components (house, building, street, landmark, locality, city, district, state, PIN) and normalise them against the India Post PIN directory. We build a noise-controlled synthetic data generator and a new human-written test set of public-place addresses, benchmark existing open parsers, compare CRF, MuRIL/XLM-R and few-shot LLMs, and measure which noise types and how many real examples close the synthetic-to-real gap.
