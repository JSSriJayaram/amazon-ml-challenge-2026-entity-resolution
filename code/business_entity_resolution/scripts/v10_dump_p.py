"""Re-create the v10 deployment scores on test and SAVE them (s1_id, cand_id, p, country), so that
alternative decodes (e.g. France-specific thresholds) can be written without re-scoring.
Same code path as combiner_v10.py --write (models refit on all sample-A CE rows)."""
import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import combiner_v10 as C
from combiner_v10 import G, config, io, pair_evidence, design_rich


def main():
    A = set(pd.read_parquet(G.CE6 / "ce_train.parquet", columns=["s1_id"])["s1_id"])
    base = pd.read_parquet(G.BASES["live"][0])
    base = base[base["s1_id"].isin(A)].reset_index(drop=True)
    spec = G.VARIANTS["v8_xlmr+mdeberta"]
    cols = [c for c, _, _ in spec]
    df = G.attach(base, spec, "oof")
    has = df[cols].notna().any(axis=1).values
    df = df.merge(pair_evidence(df[has], "train"), on=["s1_id", "cand_id"], how="left")
    y = df["label"].values
    lr_all = LogisticRegression(C=1.0, max_iter=3000).fit(G.design(df[has], cols), y[has])
    p_lr_all = G.combine(df, cols, lr_all)
    gb_all = lgb.train(C.PARAMS, lgb.Dataset(design_rich(df, cols, p_lr_all)[has], y[has]), C.ROUNDS)
    # sample-A OOF of the deployed stack is not honest; keep train p only for count statistics
    df["p_v10"] = p_lr_all
    df.loc[has, "p_v10"] = gb_all.predict(design_rich(df, cols, p_lr_all)[has])
    df[["s1_id", "cand_id", "label", "p_v10"]].to_parquet(config.artifact("submissions", "v10", "trainA_p_insample.parquet"))
    fs = sorted(glob.glob(str(G.BASES["live"][1] / "chunk_*.parquet")))
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te["p"] = te["p"].astype(np.float64)
    te = G.attach(te, spec, "test")
    mt = te[cols].notna().any(axis=1).values
    te = te.merge(pair_evidence(te[mt], "test"), on=["s1_id", "cand_id"], how="left")
    p_lr_te = G.combine(te, cols, lr_all)
    p_final = p_lr_te.copy()
    p_final[mt] = gb_all.predict(design_rich(te, cols, p_lr_te)[mt])
    out = pd.DataFrame({"s1_id": te["s1_id"].values, "cand_id": te["cand_id"].values, "p": p_final,
                        "p_graph": te["p"].values, "has_ce": mt})
    cty = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id", "country"])
    out = out.merge(cty.rename(columns={"entity_id": "s1_id"}), on="s1_id", how="left")
    out.to_parquet(config.artifact("submissions", "v10", "test_p.parquet"))
    print(f"[dump] saved {len(out):,} test pairs with v10 p", flush=True)


if __name__ == "__main__":
    main()
