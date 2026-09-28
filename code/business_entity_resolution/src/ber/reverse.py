"""Reverse competition: for every S2/S3 record, its top-K S1 entities over the WHOLE
S1 universe of the split. EDA showed 32-49% of confident false merges are records
that truly belong to ANOTHER S1 entity; a pair-wise model cannot see that, this can.

Stored as dense arrays per source: s1_pos[n_records, K] (-1 = none), score[n_records, K].
Computed identically for train and test (always against the full S1 universe), so
train/test features agree. Must run in a process without lightgbm (OpenMP clash)."""
import time

import numpy as np
import pandas as pd

from . import config
from .blocking import RETRIEVERS, retrieve

K = 5


def build_reverse(split: str, k: int = K) -> None:
    cols = ["entity_id", "country"] + RETRIEVERS["combo"].fields
    s1 = pd.read_parquet(config.artifact("norm", f"{split}_s1.parquet"), columns=cols)
    for src in (2, 3):
        out = config.artifact("reverse", f"{split}_s{src}.npz")
        if out.exists():
            print(f"[reverse] {split} s{src} cached")
            continue
        t = time.time()
        q = pd.read_parquet(config.artifact("norm", f"{split}_s{src}.parquet"), columns=cols)
        r = RETRIEVERS["combo"]
        r.k = k
        # IDF from the S1 universe only (passed twice -> identical weights for any query set)
        res = retrieve(q, {1: s1}, r, universe_s1=s1)[["s1_pos", "cand_pos", "score", "rank"]]
        pos = np.full((len(q), k), -1, dtype=np.int32)
        sc = np.zeros((len(q), k), dtype=np.float32)
        rr = res["rank"].values.astype(np.int64) - 1
        pos[res["s1_pos"].values, rr] = res["cand_pos"].values  # query row -> S1 row
        sc[res["s1_pos"].values, rr] = res["score"].values
        np.savez(out, s1_pos=pos, score=sc)
        print(f"[reverse] {split} s{src}: {len(q):,} records in {time.time() - t:.0f}s", flush=True)
        del q, res, pos, sc


class ReverseLookup:
    """Adds rev_* features to (s1_id, cand_id) pairs of one split."""

    def __init__(self, split: str):
        self.s1_index = pd.Index(pd.read_parquet(config.artifact("norm", f"{split}_s1.parquet"),
                                                 columns=["entity_id"])["entity_id"].values)
        self.idx, self.pos, self.score = {}, {}, {}
        for src in (2, 3):
            ids = pd.read_parquet(config.artifact("norm", f"{split}_s{src}.parquet"), columns=["entity_id"])["entity_id"]
            self.idx[src] = pd.Index(ids.values)
            z = np.load(config.artifact("reverse", f"{split}_s{src}.npz"))
            self.pos[src], self.score[src] = z["s1_pos"], z["score"]

    def features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        n = len(pairs)
        s1p = self.s1_index.get_indexer(pairs["s1_id"].values)
        src = pairs["cand_id"].str[1].astype(int).values
        rev_pos = np.full((n, K), -1, dtype=np.int32)
        rev_sc = np.zeros((n, K), dtype=np.float32)
        for s in (2, 3):
            m = src == s
            if m.any():
                cp = self.idx[s].get_indexer(pairs["cand_id"].values[m])
                ok = cp >= 0
                rows = np.flatnonzero(m)[ok]
                rev_pos[rows] = self.pos[s][cp[ok]]
                rev_sc[rows] = self.score[s][cp[ok]]
        hit = rev_pos == s1p[:, None]
        found = hit.any(1)
        rank = np.where(found, hit.argmax(1) + 1, K + 1).astype(np.float32)
        self_sc = np.where(found, (rev_sc * hit).sum(1), 0.0).astype(np.float32)
        best = rev_sc[:, 0]
        return pd.DataFrame({
            "rev_rank": rank, "rev_self": self_sc, "rev_best": best, "rev_gap": best - self_sc,
            "rev_is_best": (rank == 1).astype(np.float32),
            "rev_second": rev_sc[:, 1],  # how contested the record is
        }, index=pairs.index)


if __name__ == "__main__":
    import sys
    for sp in sys.argv[1:] or ["train", "test"]:
        build_reverse(sp)


def reverse_pairs(split: str, s1_filter=None, k: int = K, as_ids: bool = True) -> pd.DataFrame:
    """Candidate pairs from the reverse lists: (S1, record) whenever the S1 is among the
    record's top-k S1 matches over the whole S1 universe. Recovers true copies that are
    crowded out of the S1-side top-k by look-alike names (+0.6 pt pair recall, oracle
    macro F0.5 0.9914 -> 0.9936 on training). Retriever name: 'rev'."""
    s1_ids = pd.read_parquet(config.artifact("norm", f"{split}_s1.parquet"),
                             columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
    keep = None
    if s1_filter is not None:
        keep = np.zeros(len(s1_ids), dtype=bool)
        p = pd.Index(s1_ids).get_indexer(list(s1_filter))
        keep[p[p >= 0]] = True
    parts = []
    for src in (2, 3):
        z = np.load(config.artifact("reverse", f"{split}_s{src}.npz"))
        pos, sc = z["s1_pos"][:, :k], z["score"][:, :k]
        r, c = np.nonzero(pos >= 0)
        s1p = pos[r, c]
        if keep is not None:
            m = keep[s1p]
            r, c, s1p = r[m], c[m], s1p[m]
        if as_ids:
            ids = pd.read_parquet(config.artifact("norm", f"{split}_s{src}.parquet"),
                                  columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
            df = pd.DataFrame({"s1_id": s1_ids[s1p], "cand_id": ids[r]})
        else:
            df = pd.DataFrame({"s1_pos": s1p.astype(np.int32), "cand_pos": r.astype(np.int32), "src": np.int8(src)})
        df["score"] = sc[r, c].astype(np.float32)
        df["rank"] = (c + 1).astype(np.int16)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["retriever"] = "rev"
    return out
