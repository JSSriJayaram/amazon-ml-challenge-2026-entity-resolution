"""Label-free calibration for an unseen country (France proxy): train on SOURCE country only,
predict TARGET, then shift the target's probabilities (logit offset) so the target's predicted
empty-list rate equals the source's OOF predicted empty rate (the generator gives every country
the same singleton rate, 5.59% US and India). Compare with no shift and the oracle shift."""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ber import config, io
from ber.decode import decode, tune
from ber.metrics import entity_f05

P = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=50, subsample=0.8,
         subsample_freq=1, colsample_bytree=0.8, num_threads=3, seed=42, verbose=-1)


def lg(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def main(source, target, n_ent=40_000):
    meta = json.load(open(config.artifact("model", "cascade_meta.json")))
    feats = meta["stage2_features"]
    X = pd.read_parquet(config.artifact("features", "stage2_table.parquet"))
    c = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"]).set_index("entity_id")["country"]
    X["country"] = c.reindex(X["s1_id"]).values
    rng = np.random.default_rng(0)
    parts = {}
    for k, cty in (("s", source), ("t", target)):
        ids = X.loc[X.country == cty, "s1_id"].unique()
        keep = set(rng.choice(ids, min(n_ent, len(ids)), replace=False))
        parts[k] = X[X.s1_id.isin(keep)].reset_index(drop=True)
    gt = io.load_ground_truth()
    truth = {k: io.truth_dict(gt[gt.s1_id.isin(set(d.s1_id))], d.s1_id.unique()) for k, d in parts.items()}
    src, tgt = parts["s"], parts["t"]
    oof = np.zeros(len(src))
    for tr, va in GroupKFold(4).split(src, src.label, src.s1_id):
        oof[va] = lgb.train(P, lgb.Dataset(src.iloc[tr][feats], src.label.values[tr]), 300).predict(src.iloc[va][feats])
    prm, f_src = tune(src[["s1_id", "cand_id", "label"]].assign(p=oof), truth["s"])
    def stats(df, p, tr):
        pred = decode(df[["s1_id", "cand_id"]].assign(p=p), prm)
        ids = list(tr)
        n = np.array([len(pred.get(s, ())) for s in ids])
        f = np.mean([entity_f05(pred.get(s, []), tr[s]) for s in ids])
        return f, (n == 0).mean(), n.mean()
    f0, e_src, m_src = stats(src, oof, truth["s"])
    print(f"{source} OOF F0.5 {f0:.4f} | predicted empty {e_src:.4f} mean {m_src:.3f} | {prm}", flush=True)
    pt = lgb.train(P, lgb.Dataset(src[feats], src.label.values), 300).predict(tgt[feats])
    rows = []
    for d in np.round(np.arange(-2.0, 2.01, 0.25), 2):
        f, e, m = stats(tgt, 1 / (1 + np.exp(-(lg(pt) - d))), truth["t"])
        rows.append((d, f, e, m))
        print(f"  {target} offset {d:+.2f}: F0.5 {f:.4f} empty {e:.4f} mean {m:.3f}", flush=True)
    r = pd.DataFrame(rows, columns=["d", "f", "empty", "mean"])
    cal = r.iloc[(r["empty"] - e_src).abs().argmin()]
    calm = r.iloc[(r["mean"] - m_src).abs().argmin()]
    print(f"RESULT {source}->{target}: no shift {r.loc[r.d == 0, 'f'].item():.4f} | empty-matched (d={cal.d:+.2f}) {cal.f:.4f} "
          f"| mean-matched (d={calm.d:+.2f}) {calm.f:.4f} | oracle (d={r.loc[r.f.idxmax(), 'd']:+.2f}) {r.f.max():.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
