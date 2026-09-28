"""Experiment: does a second pass with collective features beat single-pass OOF?
Uses the 50k v2 feature table and its single-pass LightGBM OOF (same folds)."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ber import config, io
from ber.collective import NB_COLS, collective_features
from ber.dataset import load_records
from ber.decode import tune

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, num_threads=config.N_JOBS, seed=config.SEED, verbose=-1)


def main():
    X = pd.read_parquet(config.artifact("features", "feat_train_50000_k10.parquet"))
    X["p"] = np.load(config.artifact("oof", "lightgbm.npy"))
    base = [c for c in X.columns if c not in ("s1_id", "cand_id", "label", "p")]
    gt = io.load_ground_truth()
    ids = X["s1_id"].unique()
    truth = io.truth_dict(gt[gt.s1_id.isin(set(ids))], ids)
    t = time.time()
    rec = load_records("train", [2, 3], X["cand_id"], ["entity_id", "name_core", "addr_clean"]).set_index("entity_id")
    C = collective_features(X[["s1_id", "cand_id", "p"]], rec["name_core"], rec["addr_clean"])
    print(f"collective features in {time.time() - t:.0f}s")
    X = pd.concat([X, C.drop(columns=["p1st"])], axis=1)
    feats = base + ["p"] + [c for c in NB_COLS if c != "p1st"]
    y = X["label"].values
    oof = np.zeros(len(X))
    for tr, va in GroupKFold(5).split(X, y, X["s1_id"]):  # same folds as the pass-1 OOF
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr][feats], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(X.iloc[va][feats], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X.iloc[va][feats], num_iteration=m.best_iteration)
    _, f1 = tune(X[["s1_id", "cand_id", "label", "p"]], truth)
    prm, f2 = tune(X[["s1_id", "cand_id", "label"]].assign(p=oof), truth)
    print(f"pass-1 F0.5 = {f1:.4f}   pass-2 (collective) F0.5 = {f2:.4f}   {prm}")
    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print((imp / imp.sum()).head(12).round(3).to_string())


if __name__ == "__main__":
    main()
