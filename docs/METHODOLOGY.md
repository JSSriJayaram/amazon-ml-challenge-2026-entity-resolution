# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

**Team Name:** LowkeyLegends
**Written:** 27 September 2026

---

## 1. Executive Summary
A cascade entity-resolution system: three complementary TF-IDF retrievers (word, consonant-skeleton, address-only) feed a cheap stage-1 LightGBM filter (≈74 → ≈6 candidates per Source-1 entity), a stage-2 LightGBM matcher with ~80 name/address/number/context features plus *reverse competition* (every target record searched against the whole Source-1 universe), a graph stage over near-duplicate Source-2/3 records (2-hop expansion + graph-support features), and a Ditto-style multilingual cross-encoder (XLM-RoBERTa) that re-scores the uncertain pairs. Decoding enforces one owner per target record and is tuned directly on macro F0.5. v10 adds a LightGBM combiner over the cross-encoder scores and pair evidence (held-out validation **0.9839** vs v9 0.9831); the final system v11 applies slightly stricter decision thresholds to France only (the unseen country, where v10 predicted fewer no-match entities than the 5.59% rate seen in training). Best public leaderboard: **0.976** (v11, highest in the 4th decimal).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale:** train 2.21M S1 / 10.3M S2+S3; test 1.73M S1 / 10.0M S2+S3; test adds **France (15%)**, absent from training.
- **Label structure:** 5.6% singletons; 3.46 matches per S1 on average (up to 11); each S2/S3 record belongs to at most one S1 (0 exceptions); matched pairs always share the country; **26% of S2/S3 records are orphans** (match no S1); 30% of S1 names are exact duplicates. No leakage in row order or IDs (Spearman ≈ 0.000).
- **Noise catalogue** (30k true pairs; share of true pairs): legal suffix dropped 23%, generic business word swapped 19% (Group/Services/Center…), native Indic script 7.5% (India 18%), legal form changed 9%, generic words added 10%, renamed/DBA 8%, typos 6%, accents 7%, brackets/domain/repeated word/OCR digits 3–6%, honorific prefixes (Smt/Shri/Dr). Addresses: components dropped 22% (India 43%), extra numbers 10%, number prefixes (#/No/H.NO) 9%, **house number changed 5%** (typos such as 8209→209), number dropped 5%, empty 4%, zero padding 4%, `<NULL>` 2.5%.
- **Decoys:** of confident false merges, 32–49% truly belong to another S1 and the rest are orphans; typical decoys are same-name records without address, different businesses at the same address, and same name at another house number.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — blocking + cascade gradient boosting + graph reasoning + transformer re-scoring.
**Core Innovation:** (1) reverse competition features that ask whether a candidate fits *another* Source-1 entity better; (2) a sibling graph over S2∪S3 that lets confidently matched copies vouch for, and recover, their near-duplicates; (3) a Ditto-style multilingual cross-encoder applied only where the tabular model is uncertain, cross-fitted by entity.

---

## 3. Candidate Generation (Blocking)
- **Normalization (multi-view):** Unicode NFKC + transliteration of all scripts; legal-form extraction (US/India/France); abbreviation tables; context rules (`St Joseph`→saint vs `Nethaji St`→street, `Dr Ambedkar`→doctor vs `Blanca Dr`→drive); French address rules (R./Rue, bis/ter, N°); **512 variant spellings mined from training pairs** (e.g. `ddevlprs`→developers, `mhaaraassttr`→MH); consonant-skeleton and loose-phonetic keys.
- **Blocking keys used:** three sparse TF-IDF retrievers per country and per target source — *combo* (name + address word tokens and bigrams), *skel* (skeleton tokens, transliteration-robust), *addr* (address only; finds renamed businesses). Hashed features, streamed in chunks, frequent tokens capped (26× faster; bigrams restore specificity). Top-20 per retriever per source. **Plus a fourth, reverse retriever:** every S2/S3 record is searched against the whole S1 universe and contributes its top-5 S1s as candidates — this recovers true copies crowded out of the S1-side top-20 by look-alike names (pair recall 97.57% → 98.16%, candidate-set oracle macro F0.5 0.9914 → 0.9936).
- **Candidate pairs generated:** ≈82 raw candidates per S1 (4 retrievers) → stage-1 LightGBM filter (p ≥ 0.01, top-10 per S1) → ≈5.6 per S1 (pair recall 97.5%) → graph 2-hop expansion (+≈0.5–0.7, recall 97.85%) → **≈7.2 per S1 in `candidate_pairs.tsv`**.
- **How true matches were not lost:** union recall 97.6% of true pairs (combo 96.3%, skel 96.4%, addr 86.6%; each retriever contributes unique pairs); the stage-1 filter was chosen on a measured candidate-size/score trade-off (35.7 → 5.5 candidates at unchanged F0.5); graph expansion recovers +0.6 pt pair recall. Macro-F0.5 oracle of the final candidate set: 0.990.

---

## 4. Matching Model

**Features used:**
- Name features: Jaro-Winkler, Levenshtein, token-set/sort, partial ratio, token Jaccard/overlap, no-space and skeleton similarities, **distinctive-name view** (generator noise words learned from true pairs removed; country-specific French list), **loose phonetic key** for Indic transliteration (b/p, d/t, g/k), legal-form agreement/conflict, domain-name flag, name frequency in each universe.
- Address features: token-set/Levenshtein/partial similarity, street-word overlap, skeleton overlap, first-number equality, **digit-level house-number similarity and containment**, number overlap/conflict, city equality, missing-address flags.
- Other: retrieval scores/ranks per retriever, rank and gap within S1, **reverse competition** (rank of this S1 among the record's nearest S1s, best competing score, gap), graph-support features (strength/count of confident sibling copies vouching for the record), 2-hop flag, cross-encoder score.

**Model type:** LightGBM (stage-1, stage-2, stage-3), chosen from an 8-model comparison under identical folds (XGBoost 0.9610 ≈ LightGBM 0.9607 > MLP 0.9501 > RandomForest 0.9426 > ExtraTrees 0.9421 > LogisticRegression 0.9196 > rule 0.8314); plus Ditto-style cross-encoders — **XLM-RoBERTa-base** (MIT, 278M) and **mDeBERTa-v3-base** (MIT, 276M), ensembled — serialization `[COL] name [VAL] … [COL] address [VAL] …`, number/legal-form tagging, span-deletion/shuffle/attribute-drop/swap augmentation, 2-fold cross-fitting by S1 entity, trained on uncertain pairs + a confident sample + pseudo-labelled French pairs; combined with the LightGBM probability by logistic regression (v9), then by a small LightGBM that also sees the cross-encoder disagreement and pair evidence (reverse competition, house-number agreement, name frequency, missing-address flags) — v10; thresholds t1 0.80, t2 0.75.
**Threshold selection method:** grid search on out-of-fold predictions maximizing the exact macro F0.5 (singletons included); one-owner-per-record rule, first-match threshold t1, additional-match threshold t2 and relative threshold α·top score.

---

## 5. Results & Error Analysis

| Version | Main change | Validation macro F0.5 | Public LB |
|---|---|---|---|
| v1 | baseline cascade | 0.9607 | 0.949 |
| v2 | mined variants, French rules, 200k train, candidate filter | 0.9657 | 0.954 |
| v3 | top-20 blocking, context rules, distinctive-name & number features | 0.9707 | 0.960 |
| v4 | reverse competition | 0.9729 | 0.961 |
| v6 | 400k train, graph stage, XLM-R cross-encoder | **0.9797** | **0.971** |
| v7 | v6 + cross-encoder retrained on 2× data, wider band (0.01–0.99) incl. graph pairs | 0.9809* | not submitted |
| v8 | v7 + reverse-retrieval candidates (4th retriever), cascade retrained | 0.9826* | not submitted |
| v9 | v8 + mDeBERTa-v3-base cross-encoder (ensemble with XLM-R) | 0.9831* | 0.976 |
| v10 | v9 + LightGBM combiner over [logistic output, graph probability, both cross-encoder scores, their disagreement, reverse-competition / number / name-frequency evidence] | 0.9839* | 0.976 |
| **v11_fr_strict (final)** | v10 with a France-only decision offset (stricter, logit +0.75); US/India unchanged. France has no labels, and an unseen-country simulation (train on one country, test on the other) showed the best offset differs by country (US→India +0.75, India→US −0.75), so the direction was chosen on the public leaderboard | = v10 on US/India | **0.976** (best; +0.0001–0.0003 over v10 in the 4th decimal) |

\* held-out confirmation half of the CE-covered entities (combiner and thresholds fitted on the other half); frozen v6 scores 0.9798 on the same entities. v9 − v6 = +0.0033 (95% paired bootstrap CI [+0.0030, +0.0036]; US +0.0026, India +0.0043, singletons +0.0060); v9 − v8 = +0.0005 [+0.0004, +0.0007]; **v10 − v9 = +0.0008 [+0.0006, +0.0009]**.

- **F_0.5 Score (macro):** **0.9839** for the final system (v10; v9 0.9831) (held-out confirmation half; v6 0.9798 on the same entities). Cross-encoder uncertain-pair AUC: mDeBERTa 0.9629, XLM-R 0.9590, graph LightGBM 0.9508; India gains most from the cross-encoders.
- **Common false positives (wrong merges):** same-name copies without an address that belong to another S1 or to no S1; neighbouring house numbers on the same street; generic names (e.g. French "Amicale/Club + city").
- **Common false negatives (missed matches):** copies never retrieved (≈0.0085 of the remaining loss is the retrieval ceiling; mostly Indian native-script names and name-only records crowded out by identical names); name-only copies rejected as ambiguous.
- **Experiments that did not help (measured):** LightGBM+XGBoost ensemble (+0.0004), collective similarity features (+0.0008), removing absolute retrieval features for country robustness (worse), self-training on an unseen-country simulation (+0.0007), eligibility-aware reassignment decoder (±0.0000), GBM stacker on logits only (+0.0001; adding pair evidence made it work: v10 +0.0008).
- **Unseen-country simulation (France proxy):** train on US only → India 0.878 vs 0.948 in-domain; motivates multilingual cross-encoder and French pseudo-labels.

---

## 6. Conclusion
Most of the gain came from understanding the data generator (noise catalogue, decoys, singletons) and turning it into targeted normalization, reverse-competition and graph features; the largest single step was a multilingual cross-encoder applied only to uncertain pairs. The remaining loss is dominated by copies never retrieved and by genuinely ambiguous name-only records.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/ber/` (config, io, lexicons, normalize, mine, blocking, features, dataset, reverse, graph, decode, metrics, models), `scripts/` (entry point `make_submission.py`; `eval_blocking.py`; cross-encoder export/stacking `export_ce*.py`, `stage3_ce.py`, `gate_v7.py`; every experiment script), `kaggle/` (GPU cross-encoder scripts), `tests/`, `README.md` (exact commands), `requirements.txt` (pinned). Entry point: `python scripts/make_submission.py --stages train,block,score,write ...` then the cross-encoder stage; outputs `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
Charts: version progress, model comparison, error decomposition, noise catalogue, candidate-size trade-off, unseen-country simulation, blocking analysis (retriever recall/overlap/rank/score distributions).

**Compliance:** no external data, APIs or lookups; only the provided files; models LightGBM (MIT), XLM-RoBERTa-base (MIT, 278M), multilingual MiniLM (Apache-2.0, 118M), mDeBERTa-v3-base (MIT, 276M) — all ≤ 8B parameters.
