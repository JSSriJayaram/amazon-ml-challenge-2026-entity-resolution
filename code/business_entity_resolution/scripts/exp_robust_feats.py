"""Which feature set transfers best to an UNSEEN country (the France situation)?
For each variant: unseen F0.5 both directions (thresholds tuned on the source country)
and in-domain 5-fold F0.5, all on the 50k training table."""
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from ber import config, io
from ber.decode import tune
P = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, num_threads=3, seed=config.SEED, verbose=-1)
def main():
    X = pd.read_parquet(config.artifact("features", "feat_train_50000_k10.parquet"))
    c = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"]).set_index("entity_id")["country"]
    X["country"] = c.reindex(X["s1_id"]).values
    allf = [f for f in X.columns if f not in ("s1_id", "cand_id", "label", "country")]
    absolute = [f for f in allf if f.startswith("score_")] + ["best_score"]
    freq = ["name_freq_s1", "name_freq_tgt"]
    lens = ["n_len_a", "n_len_b", "ns_len_diff"]
    variants = {"all": allf,
                "no_abs_scores": [f for f in allf if f not in absolute],
                "no_abs_no_freq": [f for f in allf if f not in absolute + freq],
                "no_abs_no_freq_no_len": [f for f in allf if f not in absolute + freq + lens]}
    gt = io.load_ground_truth(); key = ["s1_id", "cand_id", "label"]
    tr_of = lambda d: io.truth_dict(gt[gt.s1_id.isin(set(d.s1_id))], d.s1_id.unique())
    for name, feats in variants.items():
        out = []
        for s, t in (("US", "India"), ("India", "US")):
            src, tgt = X[X.country == s], X[X.country == t]
            oof = np.zeros(len(src))
            for a, b in GroupKFold(3).split(src, src.label, src.s1_id):
                oof[b] = lgb.train(P, lgb.Dataset(src.iloc[a][feats], src.label.values[a]), 500).predict(src.iloc[b][feats])
            prm, _ = tune(src[key].assign(p=oof), tr_of(src))
            p = lgb.train(P, lgb.Dataset(src[feats], src.label.values), 500).predict(tgt[feats])
            _, f = tune(tgt[key].assign(p=p), tr_of(tgt), grid={"t1": [prm.t1], "t2": [prm.t2], "alpha": [prm.alpha]})
            out.append(f)
        oof = np.zeros(len(X))
        for a, b in GroupKFold(5).split(X, X.label, X.s1_id):
            oof[b] = lgb.train(P, lgb.Dataset(X.iloc[a][feats], X.label.values[a]), 500).predict(X.iloc[b][feats])
        _, fin = tune(X[key].assign(p=oof), tr_of(X))
        print(f"{name:24s} unseen US->India {out[0]:.4f} | unseen India->US {out[1]:.4f} | in-domain {fin:.4f}", flush=True)
if __name__ == "__main__":
    main()
