# Business Entity Resolution — reproducible pipeline

Finds, for every Source-1 business, its matching Source-2/3 records.
Blocking → cascade matcher (LightGBM) → graph expansion → decoding tuned on macro F0.5.
Optional GPU stage: a Ditto-style multilingual cross-encoder for uncertain pairs.

## Environment
- Python 3.11, macOS/Linux, 16 GB RAM is enough (every stage streams in chunks).
- `python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- GPU stage only (Kaggle/Colab): `torch`, `transformers` (preinstalled on Kaggle).
- Models used: LightGBM (MIT); cross-encoders `xlm-roberta-base` (MIT, 278M) and `microsoft/mdeberta-v3-base` (MIT, 276M);
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M) in experiments. No external data or APIs.

## Data layout
```
<repo>/student_resource/dataset/{train,test}/*.tsv     # as provided (override with BER_DATA_DIR)
<repo>/artifacts/                                       # all intermediate files (BER_ARTIFACT_DIR)
<repo>/output/                                          # matching_results.tsv, candidate_pairs.tsv
```

## Run end to end
All commands from `code/business_entity_resolution/` with `export PYTHONPATH=src`.

```bash
# 1. normalize all records (multi-view names/addresses)            ~7 min
python -m ber.run prepare --split train
python -m ber.run prepare --split test
# 2. learn variant spellings from training pairs (Layer-2 lexicon)   ~3 min
python -m ber.mine            # then re-run step 1 so the mined aliases are applied
# 3. training-sample blocking (two independent entity samples)       ~20 min
python scripts/eval_blocking.py --n 200000 --retrievers combo,skel,addr
python scripts/eval_blocking.py --n 220000 --seed 7 --tag B --retrievers combo,skel,addr
# 4. reverse competition + sibling (graph) indexes, train and test   ~60 min
python -m ber.reverse train test
python -m ber.graph train test
# 5. train cascade (400k entities), block test, score, decode, write both TSVs   ~2 h
python scripts/make_submission.py --stages train,block,score,write \
       --n_train 200000 --extra eval_train_220000B.parquet --k 20 --prune_top 10 --reverse --graph --rev --force
#    (--rev adds the reverse-retrieval candidates: each S2/S3 record's top-5 S1s over the whole S1 universe)
# 6. validate
cd ../../student_resource && python3 utils/validate_submission.py \
   --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

Cross-encoder stage (GPU; Kaggle T4×2, `kaggle/*.py` headers explain each step):
```bash
python scripts/export_ce_v7.py          # text pairs gated by the final graph model (0.01<p<0.99), 2 entity folds
# run kaggle/ce_kaggle_v7_xlmr.py and kaggle/ce_kaggle_v7_mdeberta.py on the exported dataset,
# download ce_oof_<arch>.parquet / ce_test_<arch>.parquet into artifacts/ce_v7/
python scripts/gate_v7.py                            # compare variants against frozen v6 / submitted v9 on a held-out half
python scripts/gate_v7.py --write "v8_xlmr+mdeberta" # v9: v8 cascade + XLM-R + mDeBERTa, logistic combiner (LB 0.976)
python scripts/combiner_v10.py                       # v10: LightGBM combiner on CE scores + pair evidence, vs v9
python scripts/combiner_v10.py --write               # writes both TSVs only if v10 beats v9 on the held-out half
```
(The submitted v6 used the same recipe with an earlier export, `scripts/export_ce.py` +
`kaggle/ce_kaggle.py` + `scripts/stage3_ce.py --arch xlmr --write`, gated by the v4 model's probabilities.)

Package: `python scripts/build_package.py --team <TEAM> --version <v9|v10>` validates both TSVs and
writes `<TEAM>_submission.zip` (output/, code/business_entity_resolution/, Documentation_template.md).

## Shipped lexicons (`lexicon/`)
`noise_words.json` (28 generator noise words learned from true training pairs, used by the distinctive-name view) and
`mined_aliases.json` (512 spelling variants mined by `ber.mine`) are included so the normaliser and features behave as in
the submitted runs. Copies in `artifacts/lexicon/` (written by `ber.mine`) take precedence.

## Source map (`src/ber/`)
| module | role |
|---|---|
| `config.py`, `io.py` | paths; safe TSV loading; submission writers |
| `lexicons.py`, `normalize.py` | hand-seeded variant tables; multi-view normalization (transliteration, legal forms, context rules, French address rules) |
| `mine.py` | variant spellings learned from training ground-truth pairs |
| `blocking.py` | 3 TF-IDF retrievers (combo / skeleton / address), hashed & streamed, sparse top-k per country |
| `features.py`, `dataset.py` | ~80 pair features (name, distinctive name, loose phonetic key, address, house-number digits, context) |
| `reverse.py` | reverse competition: each S2/S3 record vs the whole S1 universe |
| `graph.py` | sibling index over S2∪S3; 2-hop candidate expansion; graph-support features |
| `decode.py`, `metrics.py` | decoding (one-record-one-entity, two thresholds) tuned on exact macro F0.5 |
| `models.py` | model zoo used for the model comparison |

`scripts/` holds the entry points and every experiment that informed a design decision
(model comparison, cascade trade-off, unseen-country simulation, noise catalogue, blocking EDA).
Unit tests: `pytest` (scorer reproduces the official 0.714 example; normalizer cases).
