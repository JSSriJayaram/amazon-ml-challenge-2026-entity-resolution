"""Simulate the France situation inside TRAIN: one country is unseen.

A) in-domain  : 5-fold CV on both countries, score the held-out country's entities
B) unseen     : train on SOURCE country only, predict TARGET country (like France)
C) self-train : B + pseudo-labels from B's confident predictions on TARGET, retrain
All decode thresholds are tuned on the SOURCE country only (as we must for France)."""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ber import config, io
from ber.decode import tune

P = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, num_threads=4, seed=config.SEED, verbose=-1)


def fit(X, y, rounds=700):
    return lgb.train(P, lgb.Dataset(X, y), rounds)


def score(df, p, truth, prm=None):
    grid = None if prm is None else {"t1": [prm.t1], "t2": [prm.t2], "alpha": [prm.alpha]}
    return tune(df.assign(p=p), truth, grid=grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    a = ap.parse_args()
    X = pd.read_parquet(config.artifact("features", "feat_train_50000_k10.parquet"))
    c = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"]).set_index(
        "entity_id")["country"]
    X["country"] = c.reindex(X["s1_id"]).values
    feats = [f for f in X.columns if f not in ("s1_id", "cand_id", "label", "country")]
    gt = io.load_ground_truth()
    src, tgt = X[X.country == a.source].reset_index(drop=True), X[X.country == a.target].reset_index(drop=True)
    truth = {k: io.truth_dict(gt[gt.s1_id.isin(set(d.s1_id))], d.s1_id.unique()) for k, d in (("s", src), ("t", tgt))}
    key = ["s1_id", "cand_id", "label"]

    # thresholds from SOURCE only (OOF on source)
    oof = np.zeros(len(src))
    for tr, va in GroupKFold(5).split(src, src.label, src.s1_id):
        oof[va] = fit(src.iloc[tr][feats], src.label.values[tr]).predict(src.iloc[va][feats])
    prm, f_src = score(src[key], oof, truth["s"])
    print(f"source {a.source} OOF F0.5 = {f_src:.4f}   thresholds {prm}")

    # A) in-domain reference on target (5-fold within target)
    oof_t = np.zeros(len(tgt))
    for tr, va in GroupKFold(5).split(tgt, tgt.label, tgt.s1_id):
        oof_t[va] = fit(tgt.iloc[tr][feats], tgt.label.values[tr]).predict(tgt.iloc[va][feats])
    _, fA = score(tgt[key], oof_t, truth["t"], prm)
    print(f"A) in-domain  {a.target}: F0.5 = {fA:.4f}")

    # B) unseen target
    m = fit(src[feats], src.label.values)
    pB = m.predict(tgt[feats])
    _, fB = score(tgt[key], pB, truth["t"], prm)
    print(f"B) unseen     {a.target}: F0.5 = {fB:.4f}   (gap {fA - fB:+.4f})")

    # C) self-training rounds with confident pseudo-labels
    p = pB
    for it, (hi, lo) in enumerate([(0.95, 0.05), (0.9, 0.05), (0.9, 0.1)], 1):
        conf = (p >= hi) | (p <= lo)
        pseudo = tgt[conf].copy()
        pseudo["label"] = (p[conf] >= hi).astype(int)
        acc = (pseudo["label"].values == tgt.label.values[conf]).mean()
        Xc = pd.concat([src, pseudo], ignore_index=True)
        m = fit(Xc[feats], Xc.label.values)
        p = m.predict(tgt[feats])
        _, fC = score(tgt[key], p, truth["t"], prm)
        print(f"C{it}) self-train (p>={hi}|p<={lo}: {conf.mean():.1%} of target pairs, "
              f"pseudo-label accuracy {acc:.4f}): F0.5 = {fC:.4f}   ({fC - fB:+.4f} vs unseen)")


if __name__ == "__main__":
    main()
