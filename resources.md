# Resources

The master list of external links. Add every new link to its section; if a link dies, mark it ✗ and add the replacement.

**Check column** (all links tested on 10 Nov 2026):
- **read** = we opened it and read the relevant content.
- **loads** = the link opens (HTTP 200), but we haven't reviewed the content in detail.
- **⚠** = blocks automated checks (403 / timeout); open it in a browser.

## 1. Closest prior work
| Resource | Link | Used for | Licence | Check |
|---|---|---|---|---|
| Dhavase & Chaudhary (2026), chunk-based retrieval for Indian address parsing, *ETASR* | PDF: https://etasr.com/index.php/ETASR/article/download/18510/7001 · DOI: https://doi.org/10.48084/etasr.18510 | Main related work; numbers to compare against | CC-BY 4.0 | read (PDF) · ⚠ DOI redirect timed out |

## 2. Open Indian-address models (zero-shot baselines)
| Resource | Link | Used for | Licence | Check |
|---|---|---|---|---|
| Shiprocket IndicBERT address NER | https://huggingface.co/shiprocket-ai/open-indicbert-indian-address-ner | E1 baseline | Apache-2.0 | read |
| Shiprocket TinyBERT address NER | https://huggingface.co/shiprocket-ai/open-tinybert-indian-address-ner | E1 baseline | Apache-2.0 | read (labels + licence, 10 Nov) |
| MuRIL Indian address NER v1 (mukuls9971) | https://huggingface.co/mukuls9971/muril-indian-address-ner-v1 | E1 baseline; handles Hindi and Hinglish | not stated on card | read |
| TinyBERT Indian address parser (gagan1985) | https://huggingface.co/gagan1985/tinybert-4l-312d-indian-address-parser | E1 baseline | Apache-2.0 | read (card excerpt) |
| abr-6024 Indian address parser 1.5B (LLM) | Model: https://huggingface.co/abr-6024/indian-address-parser-1.5B · GGUF: https://huggingface.co/abr-6024/indian-address-parser-1.5B-GGUF · pip: https://pypi.org/project/india-address-parser/ | E1 baseline (generative) | Apache-2.0 (model and dataset) | loads |
| howdoiuse-keyboard Indian address parser | https://huggingface.co/howdoiuse-keyboard/indian-address-parser-model | Possible extra baseline | MIT | read (labels, 10 Nov) |

## 3. Data and gazetteers
| Resource | Link | Used for | Licence | Check |
|---|---|---|---|---|
| All India Pincode Directory (India Post, data.gov.in) | Catalogue: https://www.data.gov.in/catalog/all-india-pincode-directory · Resource / API: https://www.data.gov.in/resource/all-india-pincode-directory-till-last-month | Gazetteer, generator, normaliser | GODL-India: https://www.data.gov.in/Godl | loads (the portal was flaky during search) |
| Pincode directory GitHub mirror | https://github.com/dropdevrahul/pincodes-india (`pincode.csv`) | **The copy we use** (`src/normalize/gazetteer.py` downloads it; 157,126 rows, 19,300 PINs, 36 states on 10 Nov 2026) | Repo MIT; data GODL-India | read + downloaded |
| Pincode directory Kaggle copy (Aug 2021) | https://www.kaggle.com/datasets/mascarenhasneil/all-india-pincode-directory-aug21 | Fallback download | check page | ⚠ |
| abr-6024 synthetic Indian addresses (9k train / 800 test) | https://huggingface.co/datasets/abr-6024/indian-address-parsing-synthetic | Secondary test set; generator ideas | check card (geography: GODL-India attribution) | read |
| OSM taginfo, India `addr:postcode` (283,211 objects) | https://taginfo.geofabrik.de/asia:india/keys/addr:postcode | Judging OSM coverage | ODbL | read |
| Geofabrik India OSM extracts | https://download.geofabrik.de/asia/india.html | Street, building and POI names for the generator | ODbL | loads |
| Overpass Turbo (query OSM by city) | https://overpass-turbo.eu | Pulling names for chosen cities | ODbL (data) | loads |
| OSM `addr:*` tagging docs | https://wiki.openstreetmap.org/wiki/Key:addr:* | Understanding OSM address tags | CC BY-SA (wiki) | loads |
| India Pincode boundary maps 2025 (OpenCity) | https://data.opencity.in/dataset/india-pincode-maps-2025 | Optional: PIN boundaries | check page | loads |

## 4. General address parsers
| Resource | Link | Used for | Licence | Check |
|---|---|---|---|---|
| deepparse (code and docs) | https://github.com/GRAAL-Research/deepparse · https://deepparse.org | E1 baseline (India is zero-shot) | LGPL-3.0 | read (README) |
| Yassine et al. 2020, *Leveraging Subword Embeddings for Multinational Address Parsing* | https://arxiv.org/abs/2006.16152 | deepparse paper | arXiv | loads |
| Yassine et al., *Multinational Address Parsing: A Zero-Shot Evaluation* | https://arxiv.org/abs/2112.04008 | India zero-shot results | arXiv | loads |
| Beauchemin et al. 2023, *Deepparse* (NLP-OSS) | https://arxiv.org/abs/2311.11846 | Library paper | arXiv | loads |
| libpostal | https://github.com/openvenues/libpostal | Optional baseline (~2 GB install) | MIT | loads |

## 5. Industry research on Indian addresses
| Resource | Link | Check |
|---|---|---|
| Singhal et al., *GeoIndia: A Seq2Seq Geocoding Approach for Indian Addresses* (EMNLP 2024 Industry) | https://aclanthology.org/2024.emnlp-industry.29.pdf | read (summary) |
| GeoIndia V2 (ACM, 2025) | https://dl.acm.org/doi/10.1145/3746252.3761512 | ⚠ (403 to scripts) |
| Mangalgi et al. 2020, *Deep Contextual Embeddings for Address Classification in E-commerce* | https://arxiv.org/abs/2007.03020 | loads |
| Babu et al. 2017, *Monkey Typed Address Classification for e-Commerce* | https://ceur-ws.org/Vol-2311/paper_21.pdf | loads |
| Swamy et al. 2025, *An address intelligence framework for e-commerce deliveries* (Amazon Science) | https://www.amazon.science/publications/an-address-intelligence-framework-for-e-commerce-deliveries | loads |
| Delhivery Naksha LLM (industry article, not a paper) | https://www.linkedin.com/pulse/delhivery-maps-naksha-llm-how-ai-solving-indias-address-problem-dv4yc | loads |

## 6. LLMs for address parsing
| Resource | Link | Check |
|---|---|---|
| *AddrLLM: Address Rewriting via Large Language Model on Nationwide Logistics Data* (KDD 2025) | https://arxiv.org/abs/2411.13584 | loads |
| *A System for Name and Address Parsing with Large Language Models* (2026) | https://arxiv.org/abs/2601.18014 | loads |
| *Empirical analysis of address parsing methods in payment data* (2024) | https://arxiv.org/abs/2404.05632 | loads |

## 7. Models, libraries and tools
| Resource | Link | Used for | Check |
|---|---|---|---|
| MuRIL base (cased) | https://huggingface.co/google/muril-base-cased | M1 backbone | loads |
| XLM-RoBERTa base | https://huggingface.co/FacebookAI/xlm-roberta-base | M2 backbone | loads |
| IndicBERT v1 | https://huggingface.co/ai4bharat/indic-bert | Optional backbone | loads |
| IndicBERT v2 | https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only | Optional backbone | loads |
| IndicXlit (transliteration) | https://github.com/AI4Bharat/IndicXlit | Noise family N6 | loads |
| Naamapadam (Indic NER dataset) | https://huggingface.co/datasets/ai4bharat/naamapadam | Optional: general Indic NER data | loads |
| seqeval | https://github.com/chakki-works/seqeval | Span-level F1 | loads |
| sklearn-crfsuite | https://github.com/TeamHG-Memex/sklearn-crfsuite | CRF baseline | loads |
| Label Studio | https://labelstud.io | Annotation | loads |
| doccano | https://github.com/doccano/doccano | Annotation (alternative) | loads |
| Hugging Face token-classification guide | https://huggingface.co/docs/transformers/tasks/token_classification | Fine-tuning recipe | loads |

## 8. Context: India's digital addressing
| Resource | Link | Check |
|---|---|---|
| India Post DIGIPIN page | https://www.indiapost.gov.in/digipin | read |
| India Post DHRUVA / DIGIPIN document (May 2025) | https://www.indiapost.gov.in/documents/offerings/intiatives/IP_30052025_Digipin_English.pdf | loads |
| UPU news: *India's DIGIPIN and the future of postal addressing* (Aug 2026) | https://www.upu.int/en/news/2026/august/indias-digipin-and-the-future-of-postal-addressing | loads |
| Wikipedia: National Level Addressing Grid | https://en.wikipedia.org/wiki/National_Level_Addressing_Grid | loads |

## 9. Course and project files (in the parent folder on Paritosh's machine, not inside this folder)
| File | What it is |
|---|---|
| `../nlp-course-topics.md` | Course syllabus topics (for linking the report to lectures) |
| `../DOC-20260928-WA0000.xlsx` | Class sheet of all group projects (Group 10 is sheet row 51) |
| `../Group_10_Project_Shortlist.pdf` | Detailed write-up of the 9 shortlisted ideas (address parser = #8) |
| `../Group_10_Project_Shortlist_Brief.pdf` / `.tex` | Short LaTeX version of the shortlist |
