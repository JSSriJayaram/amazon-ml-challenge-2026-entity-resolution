"""Collective (second-pass) features.

Copies of the same business within S2/S3 resemble each other. A weak candidate
(e.g. no address) that closely resembles another candidate the model already
trusts for the same S1 is probably another copy. For each candidate c of S1 s:

  nb_name  = max_{c' != c} p(c') * name_sim(c, c')
  nb_addr  = max_{c' != c} p(c') * addr_sim(c, c')
  nb_both  = max_{c' != c} p(c') * min(name_sim, addr_sim)
plus the S1-level context of the pass-1 probabilities (top p, rank, gap, #confident).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config

NB_COLS = ["p1st", "p1st_rank", "p1st_gap", "p1st_top", "p1st_nconf", "nb_name", "nb_addr", "nb_both", "nb_sum"]


def _group_block(df: pd.DataFrame) -> pd.DataFrame:
    """df rows: s1_id, p, cand_name, cand_addr (sorted by s1_id)."""
    out = np.full((len(df), 4), np.nan, dtype=np.float32)
    s1 = df["s1_id"].values
    p = df["p"].values
    nm = df["cand_name"].values
    ad = df["cand_addr"].values
    starts = np.r_[0, np.flatnonzero(s1[1:] != s1[:-1]) + 1, len(df)]
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(a, b):
            bn = ba = bb = sm = 0.0
            for j in range(a, b):
                if i == j or p[j] < 0.05:
                    continue
                sn = fuzz.token_set_ratio(nm[i], nm[j]) / 100 if nm[i] and nm[j] else 0.0
                sa = fuzz.token_set_ratio(ad[i], ad[j]) / 100 if ad[i] and ad[j] else np.nan
                bn = max(bn, p[j] * sn)
                if sa == sa:  # not NaN
                    ba = max(ba, p[j] * sa)
                    bb = max(bb, p[j] * min(sn, sa))
                else:  # one side has no address: name agreement alone
                    bb = max(bb, p[j] * sn * 0.8)
                sm += p[j] * sn
            out[i] = (bn, ba, bb, sm)
    return pd.DataFrame(out, index=df.index, columns=["nb_name", "nb_addr", "nb_both", "nb_sum"])


def collective_features(pairs: pd.DataFrame, cand_name: pd.Series, cand_addr: pd.Series,
                        n_jobs: int = None, chunk_s1: int = 20_000) -> pd.DataFrame:
    """pairs: s1_id, cand_id, p (pass-1 probability). cand_name/addr: indexed by cand_id."""
    n_jobs = n_jobs or config.N_JOBS
    df = pairs[["s1_id", "cand_id", "p"]].copy()
    df["cand_name"] = cand_name.reindex(df["cand_id"]).fillna("").values
    df["cand_addr"] = cand_addr.reindex(df["cand_id"]).fillna("").values
    df = df.sort_values(["s1_id", "p"], ascending=[True, False])
    g = df.groupby("s1_id")["p"]
    ctx = pd.DataFrame(index=df.index)
    ctx["p1st"] = df["p"].astype(np.float32)
    ctx["p1st_rank"] = g.rank(ascending=False, method="first").astype(np.float32)
    ctx["p1st_top"] = g.transform("max").astype(np.float32)
    ctx["p1st_gap"] = (ctx["p1st_top"] - df["p"]).astype(np.float32)
    ctx["p1st_nconf"] = g.transform(lambda x: (x >= 0.5).sum()).astype(np.float32)
    codes = pd.factorize(df["s1_id"])[0]
    cuts = np.searchsorted(codes, np.arange(0, codes.max() + chunk_s1 + 1, chunk_s1))
    parts = [df.iloc[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    with Pool(n_jobs) as pool:
        nb = pd.concat(pool.map(_group_block, parts))
    return pd.concat([ctx, nb], axis=1).reindex(pairs.index)
