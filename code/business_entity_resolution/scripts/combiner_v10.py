"""v10 = v9 with a richer combiner. Pair evidence is computed by ONE function for both
train (sample-A CE-covered pairs) and test (all CE-covered pairs), so train and test
features are identical by construction (no retrieval-context features, which the test
scoring did not persist).

    python scripts/combiner_v10.py            # validate vs v9 on the CONFIRM half
    python scripts/combiner_v10.py --write    # write both TSVs only if the CI beats v9
"""
import argparse
import glob
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "scripts")
import gate_v7 as G  # noqa: E402
from ber import config, io  # noqa: E402
from ber.dataset import load_records, name_freq  # noqa: E402
from ber.decode import decode, tune  # noqa: E402
from ber.features import REC_COLS, string_features  # noqa: E402
from ber.metrics import entity_f05  # noqa: E402
from ber.reverse import ReverseLookup  # noqa: E402

SIMPLE = ["rev_rank", "rev_self", "rev_best", "rev_gap", "rev_is_best", "rev_second", "name_freq_s1", "name_freq_tgt",
          "addr_miss_a", "addr_miss_b", "num_first_eq", "num_first_lev", "num_first_contains", "dn_eq", "lsk_eq",
          "n_tset", "a_tset", "st_overlap", "legal_conflict", "city_eq"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1, num_threads=8, seed=42)
ROUNDS = 400


def pair_evidence(pairs: pd.DataFrame, split: str, chunk: int = 300_000) -> pd.DataFrame:
    rev = ReverseLookup(split)
    out = []
    cols = REC_COLS + ["country"]
    for i in range(0, len(pairs), chunk):
        p = pairs.iloc[i:i + chunk][["s1_id", "cand_id"]].reset_index(drop=True)
        s1 = load_records(split, [1], p["s1_id"], cols)
        tg = load_records(split, [2, 3], p["cand_id"], cols)
        sf = string_features(p, s1, tg)
        rv = rev.features(p)
        info = p.join(s1.set_index("entity_id")[["country", "name_core"]], on="s1_id")
        info = info.join(tg.set_index("entity_id")[["name_core"]].add_suffix("_b"), on="cand_id")
        f1 = name_freq(split, [1], info[["country", "name_core"]])
        ft = name_freq(split, [2, 3], info[["country", "name_core_b"]].set_axis(["country", "name_core"], axis=1))
        k1 = pd.MultiIndex.from_frame(info[["country", "name_core"]])
        kt = pd.MultiIndex.from_frame(info[["country", "name_core_b"]].set_axis(["country", "name_core"], axis=1))
        fr = pd.DataFrame({"name_freq_s1": np.log1p(f1.reindex(k1).fillna(0).values),
                           "name_freq_tgt": np.log1p(ft.reindex(kt).fillna(0).values)})
        out.append(pd.concat([p, sf.reset_index(drop=True), rv.reset_index(drop=True), fr], axis=1))
        print(f"  evidence {split}: {min(i + chunk, len(pairs)):,}/{len(pairs):,}", flush=True)
    return pd.concat(out, ignore_index=True)[["s1_id", "cand_id"] + SIMPLE]


def design_rich(df, cols, p_lr):
    X = pd.DataFrame({"lr": G.logit(p_lr), "p_graph": G.logit(df["p"].values)})
    for c in cols:
        X[c] = G.logit(df[c].fillna(0.5).values)
    X["ce_diff"] = (df[cols[0]] - df[cols[1]]).abs().values
    for c in SIMPLE:
        X[c] = df[c].values.astype(np.float32)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    A = set(pd.read_parquet(G.CE6 / "ce_train.parquet", columns=["s1_id"])["s1_id"])
    base = pd.read_parquet(G.BASES["live"][0])
    base = base[base["s1_id"].isin(A)].reset_index(drop=True)
    spec = G.VARIANTS["v8_xlmr+mdeberta"]
    cols = [c for c, _, _ in spec]
    df = G.attach(base, spec, "oof")
    has = df[cols].notna().any(axis=1).values
    ev = pair_evidence(df[has], "train")
    df = df.merge(ev, on=["s1_id", "cand_id"], how="left")
    ids = np.array(sorted(A))
    confirm = set(ids[pd.util.hash_array(ids.astype(object), hash_key="gatev7split20260") % 2 == 1])
    fitm = ~df["s1_id"].isin(confirm).values
    y = df["label"].values
    gt = io.load_ground_truth()
    truth = io.truth_dict(gt[gt["s1_id"].isin(A)], ids)
    conf_ids = [s for s in ids if s in confirm]
    t_fit = {s: truth[s] for s in ids if s not in confirm}

    def evaluate(p):
        d = df.assign(p=p)
        prm, _ = tune(d[fitm][["s1_id", "cand_id", "label", "p"]], t_fit)
        pred = decode(d[~fitm][["s1_id", "cand_id", "p"]], prm)
        return prm, np.array([entity_f05(pred.get(s, []), truth[s]) for s in conf_ids])

    rows = fitm & has
    lr_fit = LogisticRegression(C=1.0, max_iter=3000).fit(G.design(df[rows], cols), y[rows])
    p_lr = G.combine(df, cols, lr_fit)
    _, f_v9 = evaluate(p_lr)
    X = design_rich(df, cols, p_lr)
    p_gb = p_lr.copy()
    fit_rows = np.flatnonzero(rows)
    grp = pd.util.hash_array(df["s1_id"].values[fit_rows].astype(object)) % 2
    for k in (0, 1):  # out-of-fold on FIT so thresholds are tuned on honest scores
        tr, va = fit_rows[grp != k], fit_rows[grp == k]
        p_gb[va] = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), ROUNDS).predict(X.iloc[va])
    m_fit = lgb.train(PARAMS, lgb.Dataset(X.iloc[fit_rows], y[fit_rows]), ROUNDS)
    conf_rows = np.flatnonzero(~fitm & has)
    p_gb[conf_rows] = m_fit.predict(X.iloc[conf_rows])
    prm_gb, f_gb = evaluate(p_gb)
    d = f_gb - f_v9
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"v9 logistic {f_v9.mean():.4f} | v10 richer {f_gb.mean():.4f} | v10 - v9 = {d.mean():+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] | decode {prm_gb}", flush=True)
    if not a.write:
        return
    assert lo > 0, "v10 does not beat v9 on CONFIRM - not writing"
    # deployment: logistic + LightGBM refit on ALL sample-A CE rows, FIT-tuned thresholds
    lr_all = LogisticRegression(C=1.0, max_iter=3000).fit(G.design(df[has], cols), y[has])
    p_lr_all = G.combine(df, cols, lr_all)
    X_all = design_rich(df, cols, p_lr_all)
    gb_all = lgb.train(PARAMS, lgb.Dataset(X_all[has], y[has]), ROUNDS)
    fs = sorted(glob.glob(str(G.BASES["live"][1] / "chunk_*.parquet")))
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    assert len(fs) == (len(s1_ids) + 99_999) // 100_000, "incomplete v8 score chunks"
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te["p"] = te["p"].astype(np.float64)
    te = G.attach(te, spec, "test")
    mt = te[cols].notna().any(axis=1).values
    evt = pair_evidence(te[mt], "test")
    te = te.merge(evt, on=["s1_id", "cand_id"], how="left")
    p_lr_te = G.combine(te, cols, lr_all)
    p_final = p_lr_te.copy()
    p_final[mt] = gb_all.predict(design_rich(te, cols, p_lr_te)[mt])
    te["p"] = p_final
    matches = decode(te[["s1_id", "cand_id", "p"]], prm_gb)
    owners = pd.Series([c for v in matches.values() for c in v])
    assert not owners.duplicated().any(), "a target record assigned to more than one S1"
    io.write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    cands = te.groupby("s1_id")["cand_id"].agg(list).to_dict()
    io.write_id_lists(config.OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cands, "candidate_entity_ids")
    json.dump({"variant": "v10 richer combiner", "confirm_v9": float(f_v9.mean()), "confirm_v10": float(f_gb.mean()),
               "delta": [float(d.mean()), float(lo), float(hi)],
               "decode": {k: (bool(v) if k == "one_to_one" else float(v)) for k, v in prm_gb.__dict__.items()}},
              open(config.artifact("model", "v10_meta.json"), "w"), indent=1)
    n = np.array([len(matches.get(s, ())) for s in s1_ids])
    print(f"[write] v10: {mt.sum():,} test pairs re-scored | empty={np.mean(n == 0):.3f} mean matches={n.mean():.2f}")


if __name__ == "__main__":
    main()
