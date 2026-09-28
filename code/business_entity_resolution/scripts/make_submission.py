"""End-to-end test inference -> output/matching_results.tsv + output/candidate_pairs.tsv.

Stages (each cached under artifacts/, rerun with --force):
  train : fit the final LightGBM on the labelled training feature table
  block : run every retriever over ALL test S1 vs test S2/S3 (integer-coded pairs)
  score : per chunk of S1 -> union candidates -> features -> p(match)
  write : decode (tuned thresholds) and write both TSVs
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from ber import config, io
from ber.blocking import RETRIEVERS, retrieve
from ber.dataset import build_features, union_candidates
from ber.decode import DecodeParams, decode, tune

RETR = ["combo", "skel", "addr"]


# NOTE: lightgbm is imported inside the stages that need it. Loading its OpenMP
# runtime in the same process as sparse_dot_topn's deadlocks on macOS, so the
# block stage must run in a process that never imports lightgbm (see main()).


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  num_threads=config.N_JOBS, seed=config.SEED, verbose=-1)
S1_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, num_threads=config.N_JOBS,
                 seed=config.SEED, verbose=-1)
S1_ROUNDS = 300


def stage1_feature_list(columns) -> list:
    from ber.features import CHEAP_STR
    ctx = [c for c in columns if c.startswith(("score_", "rank_"))] + [
        "n_retrievers", "best_score", "cand_src", "rank_in_src", "gap_to_best_src", "n_cands_src",
        "name_freq_s1", "name_freq_tgt"]
    return list(dict.fromkeys(ctx)) + CHEAP_STR


def prune_mask(X: pd.DataFrame, p1: np.ndarray, thr: float, top: int) -> np.ndarray:
    r = pd.Series(p1, index=X.index).groupby(X["s1_id"].values).rank(ascending=False, method="first")
    return (p1 >= thr) & (r.values <= top)


def features_for_new(new: pd.DataFrame, split: str, reverse, rev=None) -> pd.DataFrame:
    """Full feature rows for 2-hop candidates (never retrieved: retrieval columns NaN).
    Pass an existing ReverseLookup as `rev` - building one loads ~1.5 GB."""
    from ber.dataset import add_full_strings
    P = new[["s1_id", "cand_id"]].copy()
    for r in RETR:
        P[f"score_{r}"] = np.nan
        P[f"rank_{r}"] = np.nan
    X = build_features(P, split, cheap_only=True)
    X = add_full_strings(X, split)
    if reverse:
        if rev is None:
            from ber.reverse import ReverseLookup
            rev = ReverseLookup(split)
        X = pd.concat([X, rev.features(X)], axis=1)
    return X


GS_COLS = ["gs_max", "gs_sum", "gs_cnt", "gs_sib_max"]


def with_graph(base: pd.DataFrame, new: pd.DataFrame, sup: pd.DataFrame) -> pd.DataFrame:
    T = pd.concat([base.assign(hop2=0.0), new.assign(hop2=1.0)], ignore_index=True)
    T = T.merge(sup, on=["s1_id", "cand_id"], how="left")
    T[GS_COLS] = T[GS_COLS].fillna(0.0).astype(np.float32)
    return T


def _chunks_by_s1(s1_codes: np.ndarray, chunk: int):
    """Yield (lo, hi) code ranges of `chunk` S1 entities each."""
    n = int(s1_codes.max()) + 1 if len(s1_codes) else 0
    for lo in range(0, n, chunk):
        yield lo, min(lo + chunk, n)


def stage_train(n_train: int, k: int, force: bool, thr: float, top: int, reverse: bool = False,
                graph: bool = False, extra: str = "") -> None:
    """Cascade training that mirrors test inference exactly:
    cheap features for ALL blocked pairs -> stage-1 OOF -> prune -> full features for
    survivors only -> stage-2 CV (+ decode tuning) -> final models on all data."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from ber.dataset import add_full_strings, add_labels
    meta_p = config.artifact("model", "cascade_meta.json")
    if meta_p.exists() and not force:
        print("[train] cached")
        return
    t0 = time.time()
    raw = pd.read_parquet(config.artifact("blocking", f"eval_train_{n_train}.parquet"),
                          columns=["s1_id", "cand_id", "score", "rank", "retriever"])
    raw = raw[raw["rank"] <= k]
    if "rev" in RETR:  # reverse-retrieval candidates as a 4th retriever
        from ber.reverse import reverse_pairs
        raw = pd.concat([raw, reverse_pairs("train", set(raw["s1_id"]))], ignore_index=True)
    gt = io.load_ground_truth()
    codes = pd.factorize(raw["s1_id"])[0]
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    parts = []
    for lo, hi in _chunks_by_s1(codes, 50_000):
        a, b = np.searchsorted(sc, [lo, hi])
        pairs = union_candidates(raw.iloc[order[a:b]])
        parts.append(add_labels(build_features(pairs, "train", cheap_only=True), gt))
        print(f"[train] cheap features: {sum(len(p) for p in parts):,} pairs ({time.time() - t0:.0f}s)", flush=True)
    del raw
    X = pd.concat(parts, ignore_index=True)
    del parts
    y = X["label"].values
    f1 = stage1_feature_list(X.columns)
    ids = X["s1_id"].unique()
    truth = io.truth_dict(gt[gt.s1_id.isin(set(ids))], ids)
    n_true = sum(len(v) for v in truth.values())
    print(f"[train] blocked: {len(X) / len(ids):.1f} cands/S1, pair recall {y.sum() / n_true:.4f}", flush=True)
    p1 = np.zeros(len(X))
    for tr, va in GroupKFold(5).split(X, y, X["s1_id"]):
        m = lgb.train(S1_PARAMS, lgb.Dataset(X.iloc[tr][f1], y[tr]), S1_ROUNDS)
        p1[va] = m.predict(X.iloc[va][f1])
    for cap in (6, 8, 10, 12):
        kp = prune_mask(X, p1, thr, cap)
        print(f"[train]   prune thr={thr} top={cap}: {kp.sum() / len(ids):.2f} cands/S1, "
              f"recall {y[kp].sum() / n_true:.4f}", flush=True)
    m1_final = lgb.train(S1_PARAMS, lgb.Dataset(X[f1], y), S1_ROUNDS)
    m1_final.save_model(str(config.artifact("model", "stage1.txt")))
    keep = prune_mask(X, p1, thr, top)
    S = X[keep].reset_index(drop=True)
    del X
    if extra:  # second, independent entity sample: pruned by the stage-1 model trained on sample A only
        rawB = pd.read_parquet(config.artifact("blocking", extra), columns=["s1_id", "cand_id", "score", "rank", "retriever"])
        rawB = rawB[(rawB["rank"] <= k) & ~rawB["s1_id"].isin(set(ids))]
        if "rev" in RETR:
            from ber.reverse import reverse_pairs
            rawB = pd.concat([rawB, reverse_pairs("train", set(rawB["s1_id"]))], ignore_index=True)
        codesB = pd.factorize(rawB["s1_id"])[0]
        oB = np.argsort(codesB, kind="stable")
        scB = codesB[oB]
        partsB, nB = [], 0
        for lo, hi in _chunks_by_s1(codesB, 50_000):
            a_, b_ = np.searchsorted(scB, [lo, hi])
            XB = add_labels(build_features(union_candidates(rawB.iloc[oB[a_:b_]]), "train", cheap_only=True), gt)
            for c in f1:
                if c not in XB.columns:
                    XB[c] = np.nan
            kb = prune_mask(XB, m1_final.predict(XB[f1]), thr, top)
            partsB.append(XB[kb])
            nB += len(XB)
        del rawB
        SB = pd.concat(partsB, ignore_index=True)
        idsB = SB["s1_id"].unique()
        ids = np.concatenate([ids, idsB])
        truth = io.truth_dict(gt[gt.s1_id.isin(set(ids))], ids)
        n_true = sum(len(v) for v in truth.values())
        S = pd.concat([S, SB], ignore_index=True)
        print(f"[train] extra sample: {len(idsB):,} entities, {nB:,} blocked -> {len(SB):,} survivors", flush=True)
    S = add_full_strings(S, "train")
    if reverse:
        from ber.reverse import ReverseLookup
        S = pd.concat([S, ReverseLookup("train").features(S)], axis=1)
    ys = S["label"].values
    f2 = [c for c in S.columns if c not in ("s1_id", "cand_id", "label")]
    S.to_parquet(config.artifact("features", "stage2_table.parquet"), index=False)  # for tuning experiments
    print(f"[train] stage-1 keeps {len(S) / len(ids):.2f} cands/S1, pair recall {ys.sum() / n_true:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    p2 = np.zeros(len(S))
    iters, fold_models, s1_fold = [], [], {}
    for fi, (tr, va) in enumerate(GroupKFold(5).split(S, ys, S["s1_id"])):
        m = lgb.train(LGB_PARAMS, lgb.Dataset(S.iloc[tr][f2], ys[tr]), 3000,
                      valid_sets=[lgb.Dataset(S.iloc[va][f2], ys[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        p2[va] = m.predict(S.iloc[va][f2], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        fold_models.append(m)
        s1_fold.update(dict.fromkeys(S["s1_id"].values[va], fi))
    prm, f05 = tune(S[["s1_id", "cand_id", "label"]].assign(p=p2), truth)
    print(f"[train] stage-2 OOF macro F0.5 = {f05:.4f} with {prm}", flush=True)
    S[["s1_id", "cand_id", "label"]].assign(p=p2).to_parquet(config.artifact("oof", "stage2_oof.parquet"), index=False)
    lgb.train(LGB_PARAMS, lgb.Dataset(S[f2], ys), int(np.mean(iters) * 1.1)).save_model(
        str(config.artifact("model", "stage2.txt")))
    graph_meta = {}
    if graph:
        from ber.dataset import add_labels
        from ber.graph import SiblingIndex, expand_and_support
        sib = SiblingIndex("train")
        new, sup = expand_and_support(S[["s1_id", "cand_id"]].assign(p=p2), sib)
        N = add_labels(features_for_new(new, "train", reverse), gt)
        fold_of = N["s1_id"].map(s1_fold).values
        p2n = np.zeros(len(N))
        for fi, m in enumerate(fold_models):  # out-of-fold, like p2
            mk = fold_of == fi
            if mk.any():
                p2n[mk] = m.predict(N.loc[mk, f2], num_iteration=m.best_iteration)
        T = with_graph(S.assign(p2=p2), N.assign(p2=p2n), sup)
        yt = T["label"].values
        f3 = f2 + ["p2", "hop2"] + GS_COLS
        print(f"[train] graph: +{len(N):,} 2-hop pairs ({len(N) / len(ids):.2f}/S1) with {N['label'].sum():,} true "
              f"-> pair recall {yt.sum() / n_true:.4f}", flush=True)
        p3 = np.zeros(len(T))
        it3 = []
        for tr, va in GroupKFold(5).split(T, yt, T["s1_id"]):
            m = lgb.train(LGB_PARAMS, lgb.Dataset(T.iloc[tr][f3], yt[tr]), 3000,
                          valid_sets=[lgb.Dataset(T.iloc[va][f3], yt[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            p3[va] = m.predict(T.iloc[va][f3], num_iteration=m.best_iteration)
            it3.append(m.best_iteration)
        prm3, f3s = tune(T[["s1_id", "cand_id", "label"]].assign(p=p3), truth)
        print(f"[train] stage-3 (graph) OOF macro F0.5 = {f3s:.4f} (stage-2 {f05:.4f}, {f3s - f05:+.4f}) {prm3}",
              flush=True)
        T[["s1_id", "cand_id", "label"]].assign(p=p3).to_parquet(config.artifact("oof", "stage3_oof.parquet"),
                                                                 index=False)
        if f3s > f05:
            lgb.train(LGB_PARAMS, lgb.Dataset(T[f3], yt), int(np.mean(it3) * 1.1)).save_model(
                str(config.artifact("model", "stage3.txt")))
            graph_meta = {"graph": True, "stage3_features": f3, "stage3_oof_f05": f3s}
            prm, f05 = prm3, f3s
        else:
            print("[train] graph stage did not beat stage-2 -> disabled", flush=True)
    json.dump({"stage1_features": f1, "stage2_features": f2, "prune": {"thr": thr, "top": top}, "k": k,
               "reverse": reverse, "retrievers": list(RETR), **graph_meta,
               "n_train": n_train,
               "decode": {k_: (bool(v) if k_ == "one_to_one" else float(v)) for k_, v in prm.__dict__.items()},
               "oof_f05": f05, "cands_per_s1": len(S) / len(ids), "pair_recall": float(ys.sum() / n_true)},
              open(meta_p, "w"), indent=1)


def stage_block(k: int, force: bool) -> None:
    """Saves integer-coded pairs sorted by S1 row (strings would not fit in memory)."""
    todo = [n for n in RETR if force or not config.artifact("blocking", f"test_{n}_k{k}.parquet").exists()]
    if any(n != "rev" for n in todo):  # text retrievers need the records (~5 GB); 'rev' does not
        cols = ["entity_id", "country"] + RETRIEVERS["combo"].fields
        s1 = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=cols)
        tg = {s: pd.read_parquet(config.artifact("norm", f"test_s{s}.parquet"), columns=cols) for s in (2, 3)}
    for name in RETR:
        out = config.artifact("blocking", f"test_{name}_k{k}.parquet")
        if out.exists() and not force:
            print(f"[block] {name} cached")
            continue
        t = time.time()
        if name == "rev":
            from ber.reverse import reverse_pairs
            p = reverse_pairs("test", as_ids=False)[["s1_pos", "cand_pos", "src", "score", "rank"]]
        else:
            r = RETRIEVERS[name]
            r.k = k
            p = retrieve(s1, tg, r)[["s1_pos", "cand_pos", "src", "score", "rank"]]
        p = p.sort_values("s1_pos", kind="stable")
        p.to_parquet(out, index=False)
        print(f"[block] {name}: {len(p):,} pairs in {time.time() - t:.0f}s", flush=True)
        del p


def stage_score(k: int, chunk: int, force: bool) -> None:
    """Scores are written one parquet per chunk (constant memory, resumable)."""
    import lightgbm as lgb
    out_dir = config.artifact("scores", f"test_cascade_k{k}", "x").parent
    meta = json.load(open(config.artifact("model", "cascade_meta.json")))
    m1 = lgb.Booster(model_file=str(config.artifact("model", "stage1.txt")))
    m2 = lgb.Booster(model_file=str(config.artifact("model", "stage2.txt")))
    pr = meta["prune"]

    def stage1_filter(Xc: pd.DataFrame) -> np.ndarray:
        p1 = m1.predict(Xc[meta["stage1_features"]], num_threads=config.N_JOBS)
        return prune_mask(Xc, p1, pr["thr"], pr["top"])

    m3 = sib = None
    if meta.get("graph"):
        from ber.graph import SiblingIndex, expand_and_support
        m3 = lgb.Booster(model_file=str(config.artifact("model", "stage3.txt")))
        sib = SiblingIndex("test")
    rev = None
    if meta.get("reverse"):
        from ber.reverse import ReverseLookup
        rev = ReverseLookup("test")
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].values
    tid = {s: pd.read_parquet(config.artifact("norm", f"test_s{s}.parquet"), columns=["entity_id"])["entity_id"].values
           for s in (2, 3)}
    blk = {n: pd.read_parquet(config.artifact("blocking", f"test_{n}_k{k}.parquet")) for n in RETR}
    n_chunks = (len(s1_ids) + chunk - 1) // chunk
    for i in range(n_chunks):
        path = out_dir / f"chunk_{i:03d}.parquet"
        if path.exists() and not force:
            continue
        t = time.time()
        lo, hi = i * chunk, min((i + 1) * chunk, len(s1_ids))
        parts = []
        for n, b in blk.items():
            a_, b_ = np.searchsorted(b["s1_pos"].values, [lo, hi])
            sub = b.iloc[a_:b_]
            cand = np.where(sub["src"].values == 2, tid[2][np.minimum(sub["cand_pos"].values, len(tid[2]) - 1)],
                            tid[3][np.minimum(sub["cand_pos"].values, len(tid[3]) - 1)])
            parts.append(pd.DataFrame({"s1_id": s1_ids[sub["s1_pos"].values], "cand_id": cand,
                                       "score": sub["score"].values, "rank": sub["rank"].values, "retriever": n}))
        pairs = union_candidates(pd.concat(parts, ignore_index=True))
        X = build_features(pairs, "test", stage1_filter=stage1_filter)
        if rev is not None:
            X = pd.concat([X, rev.features(X)], axis=1)
        p = m2.predict(X[meta["stage2_features"]], num_threads=config.N_JOBS)
        if m3 is not None:
            new, sup = expand_and_support(X[["s1_id", "cand_id"]].assign(p=p), sib)
            if len(new):
                N = features_for_new(new, "test", meta.get("reverse", False), rev=rev)
                N = N.assign(p2=m2.predict(N[meta["stage2_features"]], num_threads=config.N_JOBS))
            else:
                N = X.iloc[:0].assign(p2=np.zeros(0))
            X = with_graph(X.assign(p2=p), N, sup)
            p = m3.predict(X[meta["stage3_features"]], num_threads=config.N_JOBS)
        pd.DataFrame({"s1_id": X["s1_id"].values, "cand_id": X["cand_id"].values,
                      "p": p.astype(np.float32)}).to_parquet(path, index=False)
        print(f"[score] chunk {i + 1}/{n_chunks}: {len(pairs):,} blocked -> {len(p):,} scored in "
              f"{time.time() - t:.0f}s", flush=True)
        del pairs, X, parts


def stage_write() -> None:
    meta = json.load(open(config.artifact("model", "cascade_meta.json")))
    prm = DecodeParams(**meta["decode"])
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    files = sorted(config.artifact("scores", f"test_cascade_k{meta['k']}", "x").parent.glob("chunk_*.parquet"))
    # provenance/coverage guard: every chunk must exist and be newer than the models that scored it
    n_expected = (len(s1_ids) + 100_000 - 1) // 100_000
    assert len(files) == n_expected, f"expected {n_expected} score chunks, found {len(files)}"
    model_time = max(config.artifact("model", m).stat().st_mtime for m in ("stage1.txt", "stage2.txt", "cascade_meta.json"))
    stale = [f.name for f in files if f.stat().st_mtime < model_time]
    assert not stale, f"score chunks older than the current models (mixed versions): {stale[:5]}"
    keep, n_pairs, seen = [], 0, set()
    p_min = min(prm.t1, prm.t2)  # pairs below this can never be selected
    cpath = config.OUTPUT_DIR / "candidate_pairs.tsv"
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "w", encoding="utf-8", newline="") as fh:  # streamed: chunks hold disjoint S1s
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for f in files:
            sc = pd.read_parquet(f)
            n_pairs += len(sc)
            for s1, ids in sc.groupby("s1_id")["cand_id"].agg(lambda x: ",".join(dict.fromkeys(x))).items():
                fh.write(f"{s1}\t{ids}\n")
                seen.add(s1)
            keep.append(sc[sc["p"] >= p_min])
        for s1 in s1_ids:  # S1s for which blocking found nothing
            if s1 not in seen:
                fh.write(f"{s1}\t\n")
    allp = pd.concat(keep, ignore_index=True)
    assert not allp.duplicated(["s1_id", "cand_id"]).any(), "duplicate (s1, candidate) pairs across chunks"
    matches = decode(allp, prm)
    owners = pd.Series([c for v in matches.values() for c in v])
    assert not owners.duplicated().any(), "a target record was assigned to more than one S1"
    assert set(matches).issubset(set(s1_ids)), "matches reference unknown S1 ids"
    io.write_id_lists(config.OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    n_match = np.array([len(matches.get(s, ())) for s in s1_ids])
    print(f"[write] {len(s1_ids):,} S1 rows | empty={np.mean(n_match == 0):.3f} | "
          f"mean matches={n_match.mean():.2f} | mean candidates={n_pairs / len(s1_ids):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="train,block,score,write")
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--prune_thr", type=float, default=0.01)
    ap.add_argument("--prune_top", type=int, default=8)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--extra", default="", help="second blocking file (independent train sample)")
    ap.add_argument("--rev", action="store_true", help="add reverse-retrieval candidates as a 4th retriever")
    a = ap.parse_args()
    st = a.stages.split(",")
    if a.rev and "rev" not in RETR:
        RETR.append("rev")
    if len(st) > 1:  # run each stage in its own process (OpenMP isolation, fresh memory)
        import subprocess
        import sys
        for s in st:
            cmd = [sys.executable, "-u", __file__, "--stages", s, "--n_train", str(a.n_train),
                   "--k", str(a.k), "--chunk", str(a.chunk), "--prune_thr", str(a.prune_thr),
                   "--prune_top", str(a.prune_top)] + (["--force"] if a.force else []) + (
                   ["--reverse"] if a.reverse else []) + (["--graph"] if a.graph else []) + (["--extra", a.extra] if a.extra else []) + (["--rev"] if a.rev else [])
            subprocess.run(cmd, check=True)
        return
    if "train" in st:
        stage_train(a.n_train, a.k, a.force, a.prune_thr, a.prune_top, a.reverse, a.graph, a.extra)
    if "reverse" in st:  # no lightgbm in this process (OpenMP)
        from ber.reverse import build_reverse
        build_reverse("train")
        build_reverse("test")
    if "block" in st:
        stage_block(a.k, a.force)
    if "score" in st:
        stage_score(a.k, a.chunk, a.force)
    if "write" in st:
        stage_write()


# Guard is required: on macOS, multiprocessing workers re-import this module.
if __name__ == "__main__":
    main()
