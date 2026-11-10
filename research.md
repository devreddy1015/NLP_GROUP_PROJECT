# Research notes

Literature and resource scan done on 10 Nov 2026. Links, licences and check dates for everything below are in `resources.md`. Re-check anything before citing it in the report.

## TL;DR
- **The problem is not untouched.** A 2026 paper and six open Hugging Face models already parse Indian addresses, so don't claim "the first Indian address parser".
- **What is still open:**
  1. Evaluation on **genuinely human-written** addresses. Existing work tests on synthetic or script-generated text.
  2. **Landmarks** and other descriptive phrases ("behind old bus stand"), which the closest paper calls an open problem.
  3. Closing the gap with **better synthetic data plus a little real data**, which that paper lists as future work.
- **Free baselines:** the open models below can be run zero-shot on our test set. That is experiment E1 (`initials.md` §7).

## 1. Closest prior work
**Dhavase & Chaudhary (2026).** "Improving Indian Address Parsing in Data-Scarce Environments Using Chunk-Based Retrieval-Augmented Transformer Models." *ETASR*, CC-BY 4.0.
- **Training data:** 15,000 synthetic addresses from a probabilistic grammar (PCFG) over lists of states, cities and PINs. By the authors' own account, it has no misspellings, abbreviations or colloquial shorthand.
- **Labels:** HOUSE, ROAD, SUBURB, CITY, STATE, POSTAL CODE, COUNTRY. **There is no landmark label.**
- **"Real-world" test set:** 500 addresses built from OpenStreetMap data, then passed through a script that adds noise. Tags were derived from OSM tags and reviewed by hand; no inter-annotator agreement was measured.
- **Models:** IndicBERT, MuRIL, DeBERTa-v3 and XLM-RoBERTa, 3 seeds each.
- **Findings:** all four score F1 0.989–0.995 on synthetic test data (same grammar as training), then collapse on the real test set:

| Entity | XLM-R, real test | + their retrieval fix | libpostal, real test |
|---|---|---|---|
| HOUSE | 0.006 | 0.007 | – |
| ROAD | 0.011 | 0.151 | 0.089 |
| SUBURB | 0.162 | 0.519 | 0.019 |
| CITY | 0.508 | 0.726 | 0.381 |
| STATE | 0.671 | 0.803 | 0.576 |
| POSTAL CODE | 0.629 | 0.734 | 0.877 |

(F1 values; XLM-R columns are means over 3 seeds.)
- **Their fix:** embed each predicted multi-word span (MiniLM sentence embeddings), look it up in a FAISS index of OSM place names, and relabel it if the match is close enough (L2 distance < 1.2).
- **Open problems they name:** descriptive spatial language ("road leading to the temple"), vernacular landmarks, and mixing synthetic training data with small amounts of real data.
- **What we borrow:** their reporting style (3 seeds + McNemar + bootstrap CIs); libpostal as a reference point; their real-address example.
- **What we do differently:** a human-written test set with an agreement check; LANDMARK as a field; fixing the *training data* (noise ablation, real-data curve) instead of adding retrieval at prediction time.

## 2. Existing open models (zero-shot baselines for E1)
| Model (Hugging Face) | Base | Output fields | Trained on | Licence | Notes |
|---|---|---|---|---|---|
| `shiprocket-ai/open-indicbert-indian-address-ner` | `ai4bharat/indic-bert` | building_name, city, country, floor, house_details, landmarks, locality, pincode, road, state, sub_locality | "300% augmented Indian address dataset" (not released) | Apache-2.0 | Card reports no metrics; 128-token context; English-script addresses |
| `shiprocket-ai/open-tinybert-indian-address-ner` | TinyBERT | same 11 labels as the IndicBERT model (checked 10 Nov) | same family | Apache-2.0 | Smaller, faster sibling |
| `mukuls9971/muril-indian-address-ner-v1` | `google/muril-base-cased` | ADDRESS_ HOUSE / BUILDING / STREET / LANDMARK / LOCALITY / CITY / DISTRICT / STATE / PIN (exactly our nine) | 26,728 synthetic addresses (Devanagari, English, Hinglish) | none in the model metadata | F1 0.815 on its own synthetic benchmark; small gazetteer (~50 cities) |
| `gagan1985/tinybert-4l-312d-indian-address-parser` | TinyBERT 4L-312D | houseNumber, houseName, poi, street, subsubLocality, subLocality, locality, village, subDistrict, district, city, state, pincode | not described | Apache-2.0 | 228-sample eval: locality recall 30.5%; adds district/state from a PIN gazetteer |
| `abr-6024/indian-address-parser-1.5B` | ~1.5B generative LM | JSON: flat, building, street, landmark, area, city, state, pincode, phone | abr-6024 synthetic dataset (§3) | Apache-2.0 (model and dataset) | Also available as `pip install india-address-parser` and in GGUF format |
| `howdoiuse-keyboard/indian-address-parser-model` | ? | HOUSE_NUMBER PLOT FLOOR BLOCK GALI AREA SUBAREA COLONY SECTOR CAMP KHASRA POLE CITY STATE PINCODE (Delhi-style; no landmark, building or road) | ? | MIT | Card read 10 Nov |

**Mapping their labels onto ours** (E1 compares only the fields both sides have):

| Ours | Shiprocket | MuRIL (mukuls) | abr-6024 JSON | deepparse |
|---|---|---|---|---|
| HOUSE | house_details, floor | ADDRESS_HOUSE | flat | StreetNumber, Unit |
| BUILDING | building_name | ADDRESS_BUILDING | building | – |
| STREET | road | ADDRESS_STREET | street | StreetName |
| LANDMARK | landmarks | ADDRESS_LANDMARK | landmark | – |
| LOCALITY | locality, sub_locality | ADDRESS_LOCALITY | area | – |
| CITY | city | ADDRESS_CITY | city* | Municipality |
| DISTRICT | – | ADDRESS_DISTRICT | city* | – |
| STATE | state | ADDRESS_STATE | state | Province |
| PIN | pincode | ADDRESS_PIN | pincode | PostalCode |

\* abr-6024's `city` holds the district name (or the metro name for big cities), so score it against CITY ∪ DISTRICT.

**gagan1985:** houseNumber + houseName → HOUSE; poi → LANDMARK; street → STREET; subsubLocality + subLocality + locality + village → LOCALITY; district + subDistrict → DISTRICT; city → CITY; state → STATE; pincode → PIN. **howdoiuse-keyboard:** HOUSE_NUMBER + PLOT + FLOOR → HOUSE; GALI → STREET; AREA + SUBAREA + COLONY + SECTOR + BLOCK + CAMP → LOCALITY; CITY, STATE, PINCODE → CITY, STATE, PIN; no BUILDING or LANDMARK.

There are also hobby projects (a DistilBERT parser on GitHub and a Streamlit app). They're not worth using as baselines.

## 3. Datasets and gazetteers
- **India Post All India Pincode Directory** (data.gov.in, GODL-India). Each row has the office name, PIN, office type, delivery status, district and state; some copies also include coordinates. The portal is sometimes down: use the GitHub mirror `dropdevrahul/pincodes-india` (`pincode.csv`) or the Kaggle copy (Aug 2021).
- **`abr-6024/indian-address-parsing-synthetic`** (Hugging Face): 9,000 train / 800 test messy addresses, each paired with JSON.
  - **Real parts:** the locality, PIN, district and state come from the PIN directory.
  - **Synthetic parts:** house, building, street and landmark names, plus the noise.
  - **Test tags:** `in_pool`, `unseen_office`, `unseen_district`.
  - **Skew:** about 44% of training rows are from Tamil Nadu.
  - **Caveat:** the JSON values are normalised (e.g. `MH` → `Maharashtra`), so they don't map one-to-one onto text spans. Use it for field-level evaluation; it can't be used for BIO training without alignment work.
- **OpenStreetMap India:** 283,211 objects carry `addr:postcode` (taginfo, 10 Nov 2026). Coverage varies a lot by city, so check the chosen cities with Overpass before relying on them. Licence: ODbL (attribution; share-alike for derived databases).
- **Not found:** any public dataset of real, human-written Indian addresses with span labels. That's expected given privacy rules, and it's the gap our test set fills.

## 4. General-purpose address parsers
- **deepparse** (Yassine et al. 2020; LGPL-3.0): a seq2seq parser trained on 20 countries.
  - **India is zero-shot only.** Reported accuracy on clean data: FastText 70.3%, FastTextAtt 75.31%, BPEmb 63.68%, BPEmbAtt 80.56%.
  - **Tags:** StreetNumber, StreetName, Unit, Municipality, Province, PostalCode, Orientation, GeneralDelivery. There is no landmark or locality tag.
  - It can be fine-tuned on new data, but our data would need converting to its format.
- **libpostal:** a CRF with large dictionaries, trained on OSM and OpenAddresses. It's strong on Western formats but weak on Indian free text (ETASR numbers in §1). It needs about 2 GB of data to install. Optional.

## 5. Industry research on Indian addresses (context only; their data is not public)
- **GeoIndia** (Singhal et al., EMNLP 2024 Industry): seq2seq geocoding of Indian e-commerce addresses into hierarchical H3 grid cells.
  - Uses Flan-T5-base and Llama-3-8B, with one model per state.
  - Reports over 50% lower mean distance error than Google Maps on their data.
  - A follow-up, **GeoIndia V2** (ACM, 2025), combines a graph model with a language model.
- **Mangalgi et al. (2020)**, *Deep Contextual Embeddings for Address Classification in E-commerce* (arXiv 2007.03020): contextual embeddings learned from Indian e-commerce addresses, used to classify addresses.
- **Babu et al. (2017)**, *Monkey Typed Address Classification for e-Commerce* (CEUR): detects junk addresses typed at random.
- **Swamy et al. (2025)**, *An address intelligence framework for e-commerce deliveries* (Amazon Science).
- **Delhivery "Naksha LLM"**: industry posts describe an LLM for Indian addresses and landmarks. We found no paper.

Takeaway for the report: companies clearly treat this as an important problem, but they keep their data and models private, which is why an open benchmark is useful.

## 6. LLMs for address parsing
- **AddrLLM** (KDD 2025): an LLM that rewrites and normalises addresses, trained on large-scale logistics data from China.
- *A System for Name and Address Parsing with Large Language Models* (arXiv 2601.18014, Jan 2026).
- *Empirical analysis of address parsing methods in payment data* (arXiv 2404.05632, 2024): compares classical, fine-tuned transformer and generative LLM approaches. A later paper summarises it as favouring fine-tuned transformers; read it before citing.

Use: these justify the LLM baseline L1. Expect the fine-tuned encoder to be both stronger and cheaper, but measure it.

## 7. Building blocks
- **MuRIL** (`google/muril-base-cased`): a BERT model for Indian languages that was also trained on transliterated text, which suits Hinglish and romanised input.
- **IndicBERT** (`ai4bharat/indic-bert`; ALBERT-based) and **IndicBERT v2** (`ai4bharat/IndicBERTv2-MLM-only`; verify the ID).
- **XLM-RoBERTa** (`FacebookAI/xlm-roberta-base`): the strongest baseline in the ETASR paper.
- **IndicXlit** (AI4Bharat): transliteration between Roman and Indic scripts, for noise family N6.
- **seqeval** for span-level metrics; **sklearn-crfsuite** for the CRF baseline; **Label Studio** or **doccano** for annotation.

## 8. Context: India's digital addressing push
- **DIGIPIN:** built by India Post with IIT Hyderabad and NRSC (ISRO), 2025. It's an open national grid that gives each ~4 m × 4 m cell a 10-character alphanumeric code.
- **DHRUVA:** India Post's 2025 framework for a national digital-address layer built on DIGIPIN ("Address as a Service").

Relevance: the codes help, but free-text addresses with landmarks will stay common, and converting text into structured fields, and then into codes, needs exactly this kind of parser. (This is our own argument, not a cited claim.)

## 9. How we differ
| | ETASR 2026 | Open HF models | **Ours** |
|---|---|---|---|
| Training data | Grammar-generated, no noise | Synthetic, mostly undisclosed | Generator with each noise family switchable and ablated |
| Test data | OSM + noising script | Their own synthetic splits | **Human-written**, public places, agreement-checked |
| Landmarks | No label | Some have one | LANDMARK label + relation normalisation |
| Fix for the gap | Retrieval at prediction time | – | Better synthetic data + a little real data |
| Normalisation | – | One adds district/state from a PIN gazetteer | PIN-directory validation and filling |

## 10. Open questions to check
- [ ] Did the ETASR authors release their data or code? We saw no link in the paper text.
- [x] Licences (checked 10 Nov): gagan1985 and abr-6024 (model and dataset) Apache-2.0; Shiprocket TinyBERT Apache-2.0; howdoiuse-keyboard MIT; mukuls9971 and Shiprocket IndicBERT have no licence in the model metadata (ask before redistributing outputs).
- [x] Shiprocket TinyBERT uses the same 11 labels as the IndicBERT model (checked 10 Nov).
- [ ] How good is OSM coverage for Bhilai / Durg / Raipur and the other chosen cities?
- [ ] Is there any 2025–26 paper with a real, human-written Indian address test set? Search Google Scholar and the ACL Anthology before writing the report.
