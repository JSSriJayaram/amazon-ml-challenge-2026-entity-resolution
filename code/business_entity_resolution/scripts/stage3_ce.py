"""Stage 3: combine the final model's probability with cross-encoder score(s) on
UNCERTAIN pairs; confident pairs and graph-added pairs keep their base probability.

Provenance (verified): base = graph stage-3 OOF probabilities for training and graph
stage-3 test probabilities for deployment; the CE gate (0.02<p<0.98) was taken from the
same v4 model on both sides; fitting/tuning uses only CE-covered entities (sample A), as
every test entity is covered. Combiner = logistic regression on logits (simple, robust).

    python scripts/stage3_ce.py --arch minilm            # report only
    python scripts/stage3_ce.py --arch minilm,xlmr       # combine both CEs
    python scripts/stage3_ce.py --arch ... --write       # write only if the CI is > 0
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ber import config, io
from ber.decode import decode, tune
from ber.metrics import entity_f05

EPS = 1e-5


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def design(p, ces):
    a = logit(p)
    cols = [a]
    for ce in ces:
        b = logit(ce)
        cols += [b, a * b]
    return np.column_stack(cols)


def load_ce(ce_dir, archs, kind):
    out = None
    for a in archs:
        c = pd.read_parquet(ce_dir / f"ce_{kind}_{a}.parquet").rename(columns={"ce": f"ce_{a}"})
        out = c if out is None else out.merge(c, on=["s1_id", "cand_id"], how="inner")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="minilm")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    archs = args.arch.split(",")
    ce_cols = [f"ce_{a}" for a in archs]
    ce_dir = config.artifact("ce", "x").parent
    s3 = config.artifact("oof", "stage3_oof.parquet")
    tr = pd.read_parquet(s3 if s3.exists() else config.artifact("oof", "stage2_oof.parquet"))
    covered = set(pd.read_parquet(ce_dir / "ce_train.parquet", columns=["s1_id"])["s1_id"])
    tr = tr[tr["s1_id"].isin(covered)].merge(load_ce(ce_dir, archs, "oof"), on=["s1_id", "cand_id"], how="left")
    tr = tr.reset_index(drop=True)
    unc = tr[ce_cols].notna().all(1).values
    gt = io.load_ground_truth()
    ids = tr.s1_id.unique()
    truth = io.truth_dict(gt[gt.s1_id.isin(set(ids))], ids)
    key = ["s1_id", "cand_id", "label"]
    prm0, f0 = tune(tr[key].assign(p=tr.p), truth)
    U = tr[unc]
    Xu, yu = design(U.p.values, [U[c].values for c in ce_cols]), U.label.values
    pu = np.zeros(len(U))
    for a, b in GroupKFold(5).split(Xu, yu, U.s1_id):
        pu[b] = LogisticRegression(C=1.0, max_iter=2000).fit(Xu[a], yu[a]).predict_proba(Xu[b])[:, 1]
    p3 = tr.p.values.copy()
    p3[unc] = pu
    prm3, f3 = tune(tr[key].assign(p=p3), truth)
    d0, d3 = decode(tr.assign(p=tr.p), prm0), decode(tr.assign(p=p3), prm3)
    diff = np.array([entity_f05(d3.get(s, []), truth[s]) - entity_f05(d0.get(s, []), truth[s]) for s in ids])
    rng = np.random.default_rng(0)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"[{args.arch}] {len(ids):,} covered entities | base F0.5 {f0:.4f} -> with CE {f3:.4f} "
          f"({f3 - f0:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]) | decode {prm3}")
    if not args.write:
        return
    assert lo > 0, "cross-encoder gain not significant - not writing a submission"
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xu, yu)
    meta = json.load(open(config.artifact("model", "cascade_meta.json")))
    fs = sorted(glob.glob(str(config.ARTIFACT_DIR / "scores" / f"test_cascade_k{meta['k']}" / "chunk_*.parquet")))
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    assert len(fs) == (len(s1_ids) + 99_999) // 100_000, "incomplete score chunks"
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te = te.merge(load_ce(ce_dir, archs, "test"), on=["s1_id", "cand_id"], how="left")
    te["p"] = te["p"].astype(np.float64)
    m = te[ce_cols].notna().all(1).values
    te.loc[m, "p"] = lr.predict_proba(design(te.p.values[m], [te[c].values[m] for c in ce_cols]))[:, 1]
    matches = decode(te[["s1_id", "cand_id", "p"]], prm3)
    owners = pd.Series([c for v in matches.values() for c in v])
    assert not owners.duplicated().any(), "a target record was assigned to more than one S1"
    io.write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    json.dump({"arch": args.arch, "val_f05_base": f0, "val_f05_with_ce": f3, "ci95": [lo, hi], "decode": {
        k: (bool(v) if k == "one_to_one" else float(v)) for k, v in prm3.__dict__.items()}},
              open(config.artifact("model", "stage3_ce_meta.json"), "w"), indent=1)
    n_match = np.array([len(matches.get(s, ())) for s in s1_ids])
    print(f"[write] matching_results.tsv: {m.sum():,} test pairs re-scored with CE | empty={np.mean(n_match == 0):.3f} "
          f"mean matches={n_match.mean():.2f}; candidate_pairs.tsv unchanged (same scored set)")


if __name__ == "__main__":
    main()
