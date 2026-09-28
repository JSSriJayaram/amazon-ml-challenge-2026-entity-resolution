"""Richer combiner experiment (no GPU): let the stacker see pair evidence the logistic
combiner cannot — reverse competition, address/number agreement, name frequency, CE
disagreement. Same protocol as gate_v7.py: v8 base, sample A, FIT/CONFIRM split, thresholds
tuned on FIT, paired bootstrap on CONFIRM against the submitted v9 combiner (logistic)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "scripts")
import gate_v7 as G  # noqa: E402
from ber import config, io  # noqa: E402
from ber.decode import decode, tune  # noqa: E402
from ber.metrics import entity_f05  # noqa: E402

FEATS = ["rev_rank", "rev_self", "rev_best", "rev_gap", "rev_is_best", "rev_second", "name_freq_s1", "name_freq_tgt",
         "addr_miss_a", "addr_miss_b", "num_first_eq", "num_first_lev", "num_first_contains", "dn_eq", "lsk_eq",
         "n_retrievers", "rank_in_src", "gap_to_best_src", "score_rev", "rank_rev", "n_tset", "a_tset", "st_overlap",
         "legal_conflict", "city_eq"]


def main():
    A = set(pd.read_parquet(G.CE6 / "ce_train.parquet", columns=["s1_id"])["s1_id"])
    base = pd.read_parquet(G.BASES["live"][0])
    base = base[base["s1_id"].isin(A)].reset_index(drop=True)
    spec = G.VARIANTS["v8_xlmr+mdeberta"]
    cols = [c for c, _, _ in spec]
    df = G.attach(base, spec, "oof")
    ft = pd.read_parquet(config.artifact("features", "stage2_table.parquet"), columns=["s1_id", "cand_id"] + FEATS)
    df = df.merge(ft, on=["s1_id", "cand_id"], how="left")
    ids = np.array(sorted(A))
    confirm = set(ids[pd.util.hash_array(ids.astype(object), hash_key="gatev7split20260") % 2 == 1])
    fitm = ~df["s1_id"].isin(confirm).values
    has = df[cols].notna().any(axis=1).values
    gt = io.load_ground_truth()
    truth = io.truth_dict(gt[gt["s1_id"].isin(A)], ids)
    conf_ids = [s for s in ids if s in confirm]
    t_fit = {s: truth[s] for s in ids if s not in confirm}
    y = df["label"].values

    def evaluate(p, name):
        d = df.assign(p=p)
        prm, _ = tune(d[fitm][["s1_id", "cand_id", "label", "p"]], t_fit)
        pred = decode(d[~fitm][["s1_id", "cand_id", "p"]], prm)
        f = np.array([entity_f05(pred.get(s, []), truth[s]) for s in conf_ids])
        print(f"[{name}] CONFIRM F0.5 {f.mean():.4f}", flush=True)
        return f

    # reference: the submitted v9 combiner (logistic on logits)
    rows = fitm & has
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(G.design(df[rows], cols), y[rows])
    p_v9 = G.combine(df, cols, lr)
    f_v9 = evaluate(p_v9, "v9 logistic (submitted)")

    # richer: small LightGBM on [logistic output, CE scores, CE disagreement, pair evidence]
    X = pd.DataFrame({"lr": G.logit(p_v9), "p_graph": G.logit(df["p"].values)})
    for c in cols:
        X[c] = G.logit(df[c].fillna(0.5).values)
    X["ce_diff"] = (df[cols[0]] - df[cols[1]]).abs().values
    for c in FEATS:
        X[c] = df[c].values
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, num_threads=8, seed=42)
    # out-of-fold on the FIT half (2 folds by entity), then a FIT-trained model for CONFIRM
    p_gb = p_v9.copy()
    fit_rows = np.flatnonzero(rows)
    grp = pd.util.hash_array(df["s1_id"].values[fit_rows].astype(object)) % 2
    for k in (0, 1):
        tr, va = fit_rows[grp != k], fit_rows[grp == k]
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), 400)
        p_gb[va] = m.predict(X.iloc[va])
    m = lgb.train(params, lgb.Dataset(X.iloc[fit_rows], y[fit_rows]), 400)
    conf_rows = np.flatnonzero(~fitm & has)
    p_gb[conf_rows] = m.predict(X.iloc[conf_rows])
    f_gb = evaluate(p_gb, "richer combiner (LightGBM + pair evidence)")
    d = f_gb - f_v9
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f"richer - v9 = {d.mean():+.4f}  95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print("top inputs:", list((imp / imp.sum()).round(3).head(10).items()))


if __name__ == "__main__":
    main()
