"""How small can the candidate set get? Stage-1 = cheap LightGBM on retrieval
features only; prune by stage-1 prob / per-source top-n; stage-2 (existing OOF)
decides on survivors. Reports candidates per S1 vs pair recall vs final macro F0.5."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from ber import config, io
from ber.decode import tune

def main():
    X = pd.read_parquet(config.artifact("features", "feat_train_50000_k10.parquet"))
    oof2 = np.load(config.artifact("oof", "lightgbm.npy"))
    cheap = [c for c in X.columns if c.startswith(("score_", "rank_"))] + [
        "n_retrievers", "best_score", "cand_src", "rank_in_src", "gap_to_best_src", "n_cands_src",
        "name_freq_s1", "name_freq_tgt"]
    cheap = list(dict.fromkeys(cheap)) + ["ns_lev", "n_tset", "sk_tset", "a_tset", "st_overlap", "num_first_eq", "num_overlap", "city_eq"]
    y = X["label"].values
    oof1 = np.zeros(len(X))
    for tr, va in GroupKFold(5).split(X, y, X["s1_id"]):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, verbose=-1, n_jobs=9)
        m.fit(X.iloc[tr][cheap], y[tr])
        oof1[va] = m.predict_proba(X.iloc[va][cheap])[:, 1]
    gt = io.load_ground_truth(); ids = X["s1_id"].unique()
    truth = io.truth_dict(gt[gt.s1_id.isin(set(ids))], ids)
    n_true = sum(len(v) for v in truth.values())
    X["p1"] = oof1
    X["r1"] = X.groupby("s1_id")["p1"].rank(ascending=False, method="first")
    rows = []
    base = X[["s1_id", "cand_id", "label"]].assign(p=oof2)
    _, f_all = tune(base, truth)
    rows.append(("all (current)", len(X) / len(ids), X.label.sum() / n_true, f_all))
    for thr in (0.002, 0.01, 0.03):
        for cap in (5, 6, 8, 10):
            keep = (X["p1"] >= thr) & (X["r1"] <= cap)
            P = base[keep.values]
            _, f = tune(P, truth, grid={"t1": np.round(np.arange(0.5, 0.86, 0.05), 2),
                                        "t2": np.round(np.arange(0.1, 0.61, 0.1), 2), "alpha": [0.0, 0.5, 0.7]})
            rows.append((f"p1>={thr}, top{cap}", keep.sum() / len(ids), P.label.sum() / n_true, f))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows, columns=["rule", "cands_per_s1", "pair_recall", "final_f05"])
    print(r.to_string(index=False, float_format="%.4f"))
    r.to_csv(config.artifact("reports", "cascade_tradeoff_strfeat.csv"), index=False)

if __name__ == "__main__":
    main()
