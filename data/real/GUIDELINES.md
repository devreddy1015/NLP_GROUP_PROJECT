# Annotation guidelines v1 (10 Nov 2026)

How to label the human-written addresses. Schema v0.1 is in `initials.md` §4; this file holds the rules an
annotator needs at hand and every edge case we have decided. Add new cases at the bottom and in `initials.md` §13.

## What the data is
- Addresses of **public places only** (shops, temples, schools, hospitals, offices), written by classmates and
  friends in a Google Form. Never a home address, a person's name or a phone number. If one slips in, delete the item.
- The form asks, in this order: place name · free-text address · PIN · city · state. The last three go into `meta`.
- Tokens come from `src.data.tokenize` (whitespace split, every punctuation mark its own token). Label whole tokens.

## Labels
| Label | Covers | Inside the span | Examples |
|---|---|---|---|
| HOUSE | house / flat / plot / door / shop number, floor | the cue word (`Flat`, `H.No.`, `Shop No.`, `Plot`, `#`) | `Flat 3B`, `Shop No. 4`, `Plot 12`, `1st floor`, `D.No 5-6-7/A` |
| BUILDING | apartment, society, complex, mall, named building, tower or block *inside* a society | the type word | `Shanti Apts`, `Surya Mall`, `Tower C` |
| STREET | road, street, lane, cross, main, gali, marg, highway | the type word | `MG Road`, `2nd cross`, `Gali No. 4`, `NH 53` |
| LANDMARK | relation word + the landmark itself | the relation word (`near`, `opp.`, `behind`, `ke paas`) | `near Hanuman Mandir`, `opp. petrol pump`, `SBI ATM ke paas` |
| LOCALITY | area, colony, nagar, sector, layout, village, sub-locality, city sector or block | the type word (`Nagar`, `Sector`, `Chowk`) | `Supela`, `Sector 6`, `Koramangala 4th Block`, `Smriti Nagar` |
| CITY | city or town, in any spelling | | `Bhilai`, `B'lore`, `Hyd`, `Bhilai Nagar` |
| DISTRICT | district, only next to a cue word | | `Durg` in `Distt. Durg` or `Durg Dist` |
| STATE | state or UT, including short forms | | `Chhattisgarh`, `C.G.`, `MH` |
| PIN | 6-digit PIN, spaced or partial | | `490023`, `490 023`, `68` in `b'lore-68` |

## Rules
1. **The place's own name is `O`** (`Shree Sweets`, `IIT Bhilai`, `Hanuman Mandir` when it *is* the place, not a landmark).
   The form collects it in its own field; if the writer repeats it in the address, it stays `O`.
2. **Announcer words are `O`:** `Pin:`, `Pincode`, `Dist.`, `Distt.`, `District`, `Vill.`, `State:`, `c/o`, `India`, brackets.
3. **Each relation word opens a new LANDMARK span:** `behind SBI atm` / `nr hanuman mandir` are two spans. A landmark
   with no relation word is still LANDMARK. Hinglish postpositions belong to the span (`hanuman mandir ke paas`).
4. **DISTRICT only with a cue** (`Dist`, `Distt`, `District`, `Zila`), before or after the name. A bare `Durg` after
   `Bhilai` is CITY, even though Durg is the district: tag what is written, not what is true.
5. **Sub-locality, village, colony, sector, chowk, phase and industrial area are all LOCALITY.** Two adjacent ones are
   two spans (`Phase 2` / `Industrial Area`).
6. **Tower / block / wing inside a society is BUILDING; a city sector or block is LOCALITY.**
7. **Numbers:** a floor is HOUSE. A partial PIN (`-68`) is PIN. A spaced PIN (`490 020`) is one PIN span.
8. **Scripts:** Devanagari and Hinglish are labelled the same way as English. Do not translate.
9. **Spelling mistakes stay** in the text and get the label they would have if spelled correctly.
10. **Unsure?** Leave the token `O`, add the item id to the "to discuss" list below, and keep going.

## Slice tags (set per test item in `slices`)
`landmark` · `code_mixed` (Hindi words in Roman script) · `devanagari` · `no_pin` · `all_caps` · `no_commas` · `abbrev` · `typo`

## Process
1. Export the form to a CSV with one row per address and the columns `respondent, place, address, pin, city, state` (the respondent is a code, not a name), then `uv run python -m src.annotate tasks --csv form.csv --out data/real`. It drops duplicate places, keeps each respondent in one split, and writes `tasks_dev.json`, `tasks_train.json` (the real-train pool) and `tasks_test.json` for Label Studio, **before** anyone reads them (`build-plan.md` B3, B4). Load `labelstudio_config.xml` as the labelling interface; it shows the place name above the address so you know what to leave unlabelled.
   - **Respondent code:** use the form's response number (`R001`, `R002`, …). A response with five places becomes five rows with the same code. A blank code is rejected.
   - **Re-running:** the respondent → split map is saved in `data/real/splits.json`. Commit it. When more responses arrive, append them to the same CSV and run the command again: nobody already assigned moves, so a test item can never drift into training.
   - **Order of filling:** dev to 50, test to 200, pool to 50, then test to 250 and pool to 200 (`initials.md` §5.3).
   - **Duplicates:** only an exact match of place and city is dropped. `IIT Bhilai` and `Indian Institute of Technology Bhilai` both survive, so delete near-duplicates by eye.
2. Two people label the same 50 dev items. Export each annotator's JSON, convert with `uv run python -m src.annotate export --export ls.json --out a.jsonl --annotator <id>`, then `uv run python -m src.annotate agreement a.jsonl b.jsonl`. Fix these guidelines, then label the rest.
3. Never read test items while editing the generator or the models.

## To discuss
- Tehsil / taluk names (`Tah. Patan`): no label yet.
