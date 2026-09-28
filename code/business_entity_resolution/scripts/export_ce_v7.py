"""v7 cross-encoder export (writes to artifacts/ce_v7/, v6 files untouched).

Differences from v6 (export_ce.py):
  * training pairs from BOTH entity samples (A+B, ~400k entities) instead of A only
  * the uncertainty gate is taken from the v6 model itself (graph stage-3), on both sides:
      train: stage-3 OOF probability    test: v6 base (graph stage-3) test probability
  * wider band 0.01 < p < 0.99, graph-added (2-hop) pairs included on both sides
Folds: 2, by S1 entity (hash), so every pair's CE score comes from a model that never saw
its entity."""
import glob
import shutil

import numpy as np
import pandas as pd

from ber import config, io

LO, HI = 0.01, 0.99


def fields(split, ids1, ids23):
    s1 = io.load_source(split, 1)
    s1 = s1[s1.entity_id.isin(set(ids1))].set_index("entity_id")
    t = pd.concat([io.load_source(split, s) for s in (2, 3)])
    t = t[t.entity_id.isin(set(ids23))].set_index("entity_id")
    cut = lambda d: d[["business_name", "business_address"]].apply(lambda c: c.str.slice(0, 200))
    return cut(s1), s1["country"], cut(t)


def attach(df, split):
    a, ctry, b = fields(split, df.s1_id, df.cand_id)
    A, B = a.reindex(df.s1_id), b.reindex(df.cand_id)
    df["name_a"], df["addr_a"] = A.business_name.values, A.business_address.values
    df["name_b"], df["addr_b"] = B.business_name.values, B.business_address.values
    df["country"] = ctry.reindex(df.s1_id).values
    return df


def main():
    out = config.artifact("ce_v7", "x").parent
    tr = pd.read_parquet(config.artifact("oof", "stage3_oof.parquet"))
    tr["uncertain"] = (tr.p > LO) & (tr.p < HI)
    tr["fold"] = (pd.util.hash_array(tr["s1_id"].values.astype(object)) % 2).astype(np.int8)
    tr = attach(tr, "train")
    tr.to_parquet(out / "ce_train.parquet", index=False, compression="zstd")
    print(f"ce_train: {len(tr):,} pairs from {tr.s1_id.nunique():,} entities | uncertain {tr.uncertain.sum():,} "
          f"| positives {tr.label.mean():.3f}")
    fs = sorted(glob.glob(str(config.ARTIFACT_DIR / "scores" / "test_cascade_k20" / "chunk_*.parquet")))
    assert len(fs) == 18, f"expected 18 v6 score chunks, found {len(fs)}"
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te = te[(te.p > LO) & (te.p < HI)].reset_index(drop=True)
    te = attach(te, "test")
    te.to_parquet(out / "ce_test.parquet", index=False, compression="zstd")
    print(f"ce_test: {len(te):,} pairs | by country {te.country.value_counts().to_dict()}")
    shutil.copy(config.artifact("ce", "ce_train_fr.parquet"), out / "ce_train_fr.parquet")
    for f in ("ce_train.parquet", "ce_test.parquet", "ce_train_fr.parquet"):
        print(f"  {f}: {(out / f).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
