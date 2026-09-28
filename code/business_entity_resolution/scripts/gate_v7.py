"""Acceptance gate for v7: compare against FROZEN v6 (graph + v6 XLM-R), not graph-only.

Population: sample-A entities (the only entities with v6 cross-encoder OOF scores).
Split fixed BEFORE looking at v7 results: A is hashed into FIT (combiner + thresholds)
and CONFIRM (evaluation only) halves. Every variant uses the same protocol:
  logistic combiner on [logit p_graph, (has_ce_k, logit ce_k, logit p_graph*logit ce_k) per CE]
  fitted on FIT rows that have >=1 CE score; rows without any CE keep p_graph;
  decoder thresholds tuned on FIT entities; macro F0.5 reported on CONFIRM entities;
  paired bootstrap (by entity) of variant - v6 on CONFIRM, plus slices.
Missing CE scores are explicit (indicator + 0), never an inner join.

    python scripts/gate_v7.py                          # variants whose files exist
    python scripts/gate_v7.py --write v7_xlmr          # write matching_results.tsv for one variant
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ber import config, io
from ber.decode import DecodeParams, decode, tune
from ber.metrics import entity_f05

EPS = 1e-5
CE6 = config.ARTIFACT_DIR / "ce"
CE7 = config.ARTIFACT_DIR / "ce_v7"
V6DIR = config.ARTIFACT_DIR / "submissions" / "v6"
BASES = {  # base graph model: OOF on train, score chunks on test
    "v6": (V6DIR / "oof" / "stage3_oof.parquet", V6DIR / "scores"),
    "live": (config.ARTIFACT_DIR / "oof" / "stage3_oof.parquet", config.ARTIFACT_DIR / "scores" / "test_cascade_k20"),
}
VARIANTS = {  # name -> list of (column, oof file, test file); base "v6" unless the name starts with "v8"
    "v6": [("ce6_xlmr", CE6 / "ce_oof_xlmr.parquet", CE6 / "ce_test_xlmr.parquet")],
    "v7_xlmr": [("ce7_xlmr", CE7 / "ce_oof_xlmr.parquet", CE7 / "ce_test_xlmr.parquet")],
    "v7_xlmr+v6": [("ce6_xlmr", CE6 / "ce_oof_xlmr.parquet", CE6 / "ce_test_xlmr.parquet"),
                   ("ce7_xlmr", CE7 / "ce_oof_xlmr.parquet", CE7 / "ce_test_xlmr.parquet")],
    "v7_xlmr+mdeberta": [("ce7_xlmr", CE7 / "ce_oof_xlmr.parquet", CE7 / "ce_test_xlmr.parquet"),
                         ("ce7_mdeberta", CE7 / "ce_oof_mdeberta.parquet", CE7 / "ce_test_mdeberta.parquet")],
    "v7_all": [("ce6_xlmr", CE6 / "ce_oof_xlmr.parquet", CE6 / "ce_test_xlmr.parquet"),
               ("ce7_xlmr", CE7 / "ce_oof_xlmr.parquet", CE7 / "ce_test_xlmr.parquet"),
               ("ce7_mdeberta", CE7 / "ce_oof_mdeberta.parquet", CE7 / "ce_test_mdeberta.parquet")],
}
for _n in ("v7_xlmr", "v7_xlmr+v6", "v7_xlmr+mdeberta", "v7_all"):
    VARIANTS["v8" + _n[2:]] = VARIANTS[_n]
VARIANTS["v8_base"] = []
VARIANTS["v8_x+m"] = VARIANTS["v7_xlmr+mdeberta"]  # new XLM-R + mDeBERTa on the v8 base
_x2 = ("ce7_xlmr_s2", CE7 / "ce_oof_xlmr_s2.parquet", CE7 / "ce_test_xlmr_s2.parquet")
VARIANTS["v8_xlmr2+v6"] = VARIANTS["v7_xlmr+v6"] + [_x2]
VARIANTS["v8_all2"] = VARIANTS["v7_all"] + [_x2]
VARIANTS["v10"] = VARIANTS["v7_xlmr+mdeberta"] + [_x2]  # v9 + second XLM-R seed
VARIANTS["v8_v6ce"] = VARIANTS["v6"]  # new candidates/base + the v6 cross-encoder scores


def base_of(name):
    return "live" if name.startswith(("v8", "v10")) else "v6"


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def design(df, cols):
    a = logit(df["p"].values)
    X = [a]
    for c in cols:
        has = df[c].notna().values
        b = np.where(has, logit(df[c].fillna(0.5).values), 0.0)
        X += [has.astype(float), b, a * b]
    return np.column_stack(X)


def attach(df, spec, kind):
    for col, oof, test in spec:
        f = pd.read_parquet(oof if kind == "oof" else test).rename(columns={"ce": col})
        df = df.merge(f[["s1_id", "cand_id", col]], on=["s1_id", "cand_id"], how="left")
    return df


def combine(df, cols, lr):
    p = df["p"].values.astype(np.float64).copy()
    m = df[cols].notna().any(axis=1).values
    if m.any():
        p[m] = lr.predict_proba(design(df[m], cols))[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default="")
    args = ap.parse_args()
    A = set(pd.read_parquet(CE6 / "ce_train.parquet", columns=["s1_id"])["s1_id"])
    bases = {}
    for b, (oof_p, _) in BASES.items():
        if oof_p.exists():
            d = pd.read_parquet(oof_p)
            bases[b] = d[d["s1_id"].isin(A)].reset_index(drop=True)
    ids = np.array(sorted(A))
    confirm = set(ids[pd.util.hash_array(ids.astype(object), hash_key="gatev7split20260") % 2 == 1])
    fit_ids = [s for s in ids if s not in confirm]
    conf_ids = [s for s in ids if s in confirm]
    gt = io.load_ground_truth()
    truth = io.truth_dict(gt[gt["s1_id"].isin(A)], ids)
    t_fit = {s: truth[s] for s in fit_ids}
    s1c = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=["entity_id", "country"]).set_index(
        "entity_id")["country"]
    singles = {s for s in conf_ids if not truth[s]}
    results, preds = {}, {}
    for name, spec in VARIANTS.items():
        if not all(o.exists() for _, o, _ in spec) or base_of(name) not in bases:
            print(f"[{name}] skipped (missing files)")
            continue
        if name.startswith(("v8", "v10")) and bases["live"]["p"].equals(bases["v6"]["p"]) if len(bases["live"]) == len(bases["v6"]) else False:
            print(f"[{name}] skipped (live base is still v6)")
            continue
        cols = [c for c, _, _ in spec]
        df = attach(bases[base_of(name)].copy(), spec, "oof")
        fitm = ~df["s1_id"].isin(confirm).values
        rows = fitm & (df[cols].notna().any(axis=1).values if cols else np.zeros(len(df), bool))
        lr = None
        if cols:
            lr = LogisticRegression(C=1.0, max_iter=3000).fit(design(df[rows], cols), df["label"].values[rows])
            df["p"] = combine(df, cols, lr)
        prm, f_fit = tune(df[fitm][["s1_id", "cand_id", "label", "p"]], t_fit)
        pred = decode(df[~fitm][["s1_id", "cand_id", "p"]], prm)
        f = np.array([entity_f05(pred.get(s, []), truth[s]) for s in conf_ids])
        preds[name] = f
        results[name] = {"confirm_f05": float(f.mean()), "fit_f05": f_fit, "decode": prm, "lr": lr, "cols": cols,
                         "coverage_rows": int(df[cols].notna().any(axis=1).sum()) if cols else 0}
        print(f"[{name}] CONFIRM F0.5 {f.mean():.4f} (fit {f_fit:.4f}) | rows with CE {results[name]['coverage_rows']:,}")
    if "v6" in preds:
        rng = np.random.default_rng(0)
        cty = s1c.reindex(conf_ids).values
        for name, f in preds.items():
            if name == "v6":
                continue
            d = f - preds["v6"]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sl = {k: d[m].mean() for k, m in (("US", cty == "US"), ("India", cty == "India"),
                                                ("singletons", np.array([s in singles for s in conf_ids])))}
            print(f"  {name} - v6 = {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] | "
                  + " ".join(f"{k} {v:+.4f}" for k, v in sl.items()))
            results[name]["delta_vs_v6"] = (float(d.mean()), float(lo), float(hi))
            ref = "v8_xlmr+mdeberta"  # submitted v9 (LB 0.976)
            if ref in preds and name != ref:
                d8 = f - preds[ref]
                b8 = [d8[rng.integers(0, len(d8), len(d8))].mean() for _ in range(1000)]
                print(f"      vs submitted v9: {d8.mean():+.4f}  95% CI [{np.percentile(b8, 2.5):+.4f}, "
                      f"{np.percentile(b8, 97.5):+.4f}]")
                results[name]["delta_vs_v9"] = (float(d8.mean()), float(np.percentile(b8, 2.5)))
    if not args.write:
        return
    r = results[args.write]
    assert args.write == "v6" or r["delta_vs_v6"][1] > 0, "variant does not beat frozen v6 on CONFIRM - not writing"
    if args.write == "v10":
        assert r.get("delta_vs_v9", (0, -1))[1] > 0, "v10 does not beat submitted v9 on CONFIRM - not writing"
    # deployment: refit the combiner on ALL sample-A rows, keep FIT-tuned thresholds
    spec = VARIANTS[args.write]
    cols = r["cols"]
    df = attach(bases[base_of(args.write)].copy(), spec, "oof")
    lr = None
    if cols:
        rows = df[cols].notna().any(axis=1).values
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(design(df[rows], cols), df["label"].values[rows])
    fs = sorted(glob.glob(str(BASES[base_of(args.write)][1] / "chunk_*.parquet")))
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    assert len(fs) == (len(s1_ids) + 99_999) // 100_000, "incomplete v6 score chunks"
    te = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    te["p"] = te["p"].astype(np.float64)
    te = attach(te, spec, "test")
    if cols:
        te["p"] = combine(te, cols, lr)
    matches = decode(te[["s1_id", "cand_id", "p"]], r["decode"])
    owners = pd.Series([c for v in matches.values() for c in v])
    assert not owners.duplicated().any(), "a target record assigned to more than one S1"
    io.write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    # candidate file = exactly the pairs this base scored
    cands = te.groupby("s1_id")["cand_id"].agg(list).to_dict()
    io.write_id_lists(config.OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, cands, "candidate_entity_ids")
    n = np.array([len(matches.get(s, ())) for s in s1_ids])
    json.dump({"variant": args.write, "confirm_f05": r["confirm_f05"], "delta_vs_v6": r.get("delta_vs_v6"),
               "decode": {k: (bool(v) if k == "one_to_one" else float(v)) for k, v in r["decode"].__dict__.items()}},
              open(config.artifact("model", f"gate_{args.write}.json"), "w"), indent=1, default=float)
    print(f"[write] {args.write}: test pairs with CE {te[cols].notna().any(axis=1).sum():,} | empty={np.mean(n == 0):.3f} "
          f"mean matches={n.mean():.2f}")


if __name__ == "__main__":
    main()
