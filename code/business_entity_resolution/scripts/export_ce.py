"""Export text pairs for the GPU cross-encoder (runs locally, after v4 scoring).

ce_train.parquet : stage-1 survivors of the 200k training entities, with label, the
                   stage-2 OUT-OF-FOLD probability, a 2-fold split by S1 entity, and an
                   'uncertain' flag. Cross-encoder trains on fold A and scores fold B (and
                   vice versa), so its scores on training pairs are honest.
ce_test.parquet  : uncertain test pairs (stage-2 p in (LO, HI)).
Raw fields are exported separately (original scripts/accents) for Ditto-style serialization."""
import glob

import numpy as np
import pandas as pd

from ber import config, io

LO, HI = 0.02, 0.98


def texts(split: str, ids1, ids23):
    s1 = io.load_source(split, 1)
    s1 = s1[s1.entity_id.isin(set(ids1))].set_index("entity_id")
    t = pd.concat([io.load_source(split, s) for s in (2, 3)])
    t = t[t.entity_id.isin(set(ids23))].set_index("entity_id")
    f = lambda d: d[["business_name", "business_address"]].apply(lambda c: c.str.slice(0, 200))
    return f(s1), s1["country"], f(t)


def main():
    out = config.artifact("ce", "x").parent
    tr = pd.read_parquet(config.artifact("oof", "stage2_oof.parquet"))
    tr["uncertain"] = (tr.p > LO) & (tr.p < HI)
    tr["fold"] = (pd.util.hash_array(tr["s1_id"].values.astype(object)) % 2).astype(np.int8)
    a, ctry, b = texts("train", tr.s1_id, tr.cand_id)
    A, B = a.reindex(tr.s1_id), b.reindex(tr.cand_id)
    tr["name_a"], tr["addr_a"] = A.business_name.values, A.business_address.values
    tr["name_b"], tr["addr_b"] = B.business_name.values, B.business_address.values
    tr["country"] = ctry.reindex(tr.s1_id).values
    tr.to_parquet(out / "ce_train.parquet", index=False, compression="zstd")
    print(f"ce_train: {len(tr):,} pairs, uncertain {tr.uncertain.mean():.1%}, positives {tr.label.mean():.3f}")

    fs = sorted(glob.glob(str(config.ARTIFACT_DIR / "scores" / "test_cascade_k20" / "chunk_*.parquet")))
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te = te[(te.p > LO) & (te.p < HI)].reset_index(drop=True)
    a, ctry, b = texts("test", te.s1_id, te.cand_id)
    A, B = a.reindex(te.s1_id), b.reindex(te.cand_id)
    te["name_a"], te["addr_a"] = A.business_name.values, A.business_address.values
    te["name_b"], te["addr_b"] = B.business_name.values, B.business_address.values
    te["country"] = ctry.reindex(te.s1_id).values
    te.to_parquet(out / "ce_test.parquet", index=False, compression="zstd")
    print(f"ce_test: {len(te):,} uncertain pairs | by country {te.country.value_counts().to_dict()}")
    for f in ("ce_train.parquet", "ce_test.parquet"):
        print(f, f"{(out / f).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
