"""Is the unseen-country loss a THRESHOLD problem or a RANKING problem?"""
import lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from ber import config, io
from ber.decode import tune
P = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, num_threads=4, seed=config.SEED, verbose=-1)
def main():
    X = pd.read_parquet(config.artifact("features", "feat_train_50000_k10.parquet"))
    c = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"]).set_index("entity_id")["country"]
    X["country"] = c.reindex(X["s1_id"]).values
    feats = [f for f in X.columns if f not in ("s1_id", "cand_id", "label", "country")]
    gt = io.load_ground_truth(); key = ["s1_id", "cand_id", "label"]
    for s, t in (("US", "India"), ("India", "US")):
        src, tgt = X[X.country == s], X[X.country == t]
        tr = io.truth_dict(gt[gt.s1_id.isin(set(tgt.s1_id))], tgt.s1_id.unique())
        m = lgb.train(P, lgb.Dataset(src[feats], src.label.values), 700)
        p = m.predict(tgt[feats])
        own = lgb.train(P, lgb.Dataset(tgt[feats], tgt.label.values), 700)  # in-sample, only for AUC reference
        prm_or, f_or = tune(tgt[key].assign(p=p), tr)
        prm_src = {"US": (0.7, 0.75, 0.0), "India": None}[s]
        print(f"{s}->{t}: pair AUC unseen={roc_auc_score(tgt.label, p):.5f} | F0.5 with thresholds tuned ON {t} (oracle) = {f_or:.4f} {prm_or}")
        # label-free threshold: choose t1/t2 so predicted matches/entity ~ training average (3.46)
        best = None
        for t1 in np.arange(0.3, 0.96, 0.05):
            for t2 in np.arange(0.1, 0.96, 0.05):
                d = tgt[key].assign(p=p).sort_values(["s1_id", "p"], ascending=[True, False])
                break
            break
        imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
        print("   top features:", list(imp.head(8).index))
if __name__ == "__main__":
    main()
