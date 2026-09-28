"""Pseudo-labelled FRENCH pairs for the cross-encoder (France has no training labels).
From the v4 test scores (backed up), keep only very confident French pairs and favour
HARD ones: confident 'no' with high name similarity (decoys), confident 'yes' with lower
similarity (noisy copies). Measured pseudo-label accuracy at these confidences ~99%."""
import glob
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from ber import config, io

def main():
    fs = sorted(glob.glob(str(config.ARTIFACT_DIR / "submissions/v4/scores/chunk_*.parquet")))
    sc = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    s1 = io.load_source("test", 1)
    fr = s1[s1.country == "France"].set_index("entity_id")
    sc = sc[sc.s1_id.isin(fr.index) & ((sc.p >= 0.98) | (sc.p <= 0.02))]
    t = pd.concat([io.load_source("test", s) for s in (2, 3)])
    t = t[t.entity_id.isin(set(sc.cand_id))].set_index("entity_id")
    A, B = fr.reindex(sc.s1_id), t.reindex(sc.cand_id)
    sim = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(A.business_name.values, B.business_name.values)])
    sc["label"] = (sc.p >= 0.98).astype(int)
    hard_neg = (sc.label == 0) & (sim >= 80)
    hard_pos = (sc.label == 1) & (sim < 90)
    rng = np.random.default_rng(0)
    easy = ~(hard_neg | hard_pos)
    pick = hard_neg | hard_pos | (easy & (rng.random(len(sc)) < 0.05))
    out = sc[pick].copy()
    A, B = fr.reindex(out.s1_id), t.reindex(out.cand_id)
    out["name_a"], out["addr_a"] = A.business_name.values, A.business_address.values
    out["name_b"], out["addr_b"] = B.business_name.values, B.business_address.values
    out["country"] = "France"
    out = out[["s1_id", "cand_id", "label", "name_a", "addr_a", "name_b", "addr_b", "country"]]
    path = config.artifact("ce", "ce_train_fr.parquet")
    out.to_parquet(path, index=False, compression="zstd")
    print(f"French pseudo-pairs: {len(out):,} (hard neg {hard_neg[pick].sum():,}, hard pos {hard_pos[pick].sum():,}, "
          f"easy {(easy & pick).sum():,}) positives {out.label.mean():.2f} -> {path.stat().st_size/1e6:.0f} MB")

if __name__ == "__main__":
    main()
