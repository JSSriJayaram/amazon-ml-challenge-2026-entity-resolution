"""Compare every model in the zoo under identical conditions:
same candidates, same features, same GroupKFold(5) by S1, own decode tuning.

Reports OOF macro F0.5 (overall / per country), pair AUC, and timings."""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from ber import config, io
from ber.dataset import add_labels, build_features, union_candidates
from ber.decode import tune
from ber.models import zoo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocking", default="eval_train_50000.parquet")
    ap.add_argument("--models", default=",".join(zoo()))
    ap.add_argument("--max_rows_slow", type=int, default=600_000, help="subsample train rows for slow models")
    ap.add_argument("--k", type=int, default=10, help="keep top-k per retriever per source")
    args = ap.parse_args()

    feat_path = config.artifact("features", args.blocking.replace("eval_", "feat_").replace(".parquet", f"_k{args.k}.parquet"))
    s1 = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"])
    gt = io.load_ground_truth()
    raw = pd.read_parquet(config.artifact("blocking", args.blocking))
    raw = raw[raw["rank"] <= args.k]
    q_ids = raw["s1_id"].unique()
    truth = io.truth_dict(gt[gt.s1_id.isin(set(q_ids))], q_ids)
    country = s1.set_index("entity_id")["country"]

    if feat_path.exists():
        X = pd.read_parquet(feat_path)
    else:
        t = time.time()
        pairs = union_candidates(raw)
        X = add_labels(build_features(pairs, "train"), gt)
        X.to_parquet(feat_path, index=False)
        print(f"features: {X.shape} in {time.time() - t:.0f}s, positives={X.label.mean():.3f}")

    if args.models == "none":  # feature table only
        return
    feats = [c for c in X.columns if c not in ("s1_id", "cand_id", "label")]
    y = X["label"].values
    groups = X["s1_id"].values
    folds = list(GroupKFold(5).split(X, y, groups))
    X["country"] = country.reindex(X["s1_id"]).values

    results = []
    factories = zoo()
    for name in args.models.split(","):
        print(f"--- {name}", flush=True)
        oof = np.zeros(len(X), dtype=np.float32)
        t = time.time()
        pred_time = 0.0
        for tr, va in folds:
            if name in ("extratrees", "randomforest", "mlp", "logreg") and len(tr) > args.max_rows_slow:
                tr = np.random.default_rng(config.SEED).choice(tr, args.max_rows_slow, replace=False)
            m = factories[name]().fit(X.iloc[tr][feats], y[tr], X.iloc[va][feats], y[va])
            tp = time.time()
            oof[va] = m.predict(X.iloc[va][feats])
            pred_time += time.time() - tp
        fit_time = time.time() - t - pred_time
        P = X[["s1_id", "cand_id", "label"]].assign(p=oof)
        prm, f05 = tune(P, truth)
        per_c = {}
        for c in sorted(X["country"].unique()):
            ids = set(X.loc[X["country"] == c, "s1_id"])
            tc = {s: v for s, v in truth.items() if country[s] == c}
            _, per_c[c] = tune(P[P.s1_id.isin(ids)], tc, grid={"t1": [prm.t1], "t2": [prm.t2], "alpha": [prm.alpha]})
        r = {"model": name, "oof_f05": f05, **{f"f05_{c}": v for c, v in per_c.items()},
             "auc": roc_auc_score(y, oof), "ap": average_precision_score(y, oof),
             "fit_s": fit_time, "pred_s_per_1M": pred_time / len(X) * 1e6,
             "t1": prm.t1, "t2": prm.t2, "alpha": prm.alpha, "one_to_one": prm.one_to_one}
        results.append(r)
        np.save(config.artifact("oof", f"{name}.npy"), oof)
        print(json.dumps(r, default=float))

    res = pd.DataFrame(results).sort_values("oof_f05", ascending=False)
    print("\n" + res.to_string(index=False, float_format="%.4f"))
    res.to_csv(config.artifact("reports", "zoo_results.csv"), index=False)


# Guard is required: on macOS, multiprocessing workers re-import this module.
if __name__ == "__main__":
    main()
