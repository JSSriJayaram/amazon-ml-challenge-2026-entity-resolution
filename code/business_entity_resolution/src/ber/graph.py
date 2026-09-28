"""Graph view of S2 ∪ S3: copies of one business are near-duplicates of each other.

Sibling index: for every S2/S3 record, its nearest S2/S3 neighbours in the same
country (both sources). Used two ways (EDA: 27-39% of never-retrieved true copies have
a confidently matched sibling that is much closer to them than S1 is):
  * 2-hop expansion  - siblings of confident matches become extra candidates of the S1
  * graph support    - "how strongly do the S1's confident matches vouch for this record"
Stored as dense arrays per source: sib_key[n, K] (global key = src*1e9 + row, -1 none),
sib_score[n, K]. Must run in a process without lightgbm (OpenMP clash)."""
import time

import numpy as np
import pandas as pd

from . import config
from .blocking import RETRIEVERS, retrieve

K_PER_SRC = 4  # neighbours per target source (self is dropped) -> up to 8 siblings
OFF = 1_000_000_000


def build_siblings(split: str) -> None:
    cols = ["entity_id", "country"] + RETRIEVERS["combo"].fields
    frames = {s: pd.read_parquet(config.artifact("norm", f"{split}_s{s}.parquet"), columns=cols) for s in (2, 3)}
    empty = frames[2].iloc[:0]
    for src in (2, 3):
        out = config.artifact("graph", f"{split}_sib_s{src}.npz")
        if out.exists():
            print(f"[graph] {split} s{src} cached")
            continue
        t = time.time()
        r = RETRIEVERS["combo"]
        r.k = K_PER_SRC + 1
        # IDF from S2+S3 of the country (universe_s1 empty -> identical weights train/test)
        res = retrieve(frames[src], frames, r, universe_s1=empty)
        res = res[~((res["src"] == src) & (res["cand_pos"] == res["s1_pos"]))]  # drop self
        res = res.sort_values(["s1_pos", "score"], ascending=[True, False])
        res["slot"] = res.groupby("s1_pos").cumcount()
        res = res[res["slot"] < 2 * K_PER_SRC]
        n = len(frames[src])
        key = np.full((n, 2 * K_PER_SRC), -1, dtype=np.int64)
        sc = np.zeros((n, 2 * K_PER_SRC), dtype=np.float32)
        key[res["s1_pos"].values, res["slot"].values] = res["src"].values.astype(np.int64) * OFF + res["cand_pos"].values
        sc[res["s1_pos"].values, res["slot"].values] = res["score"].values
        np.savez(out, key=key, score=sc)
        print(f"[graph] {split} s{src}: {n:,} records in {time.time() - t:.0f}s", flush=True)


class SiblingIndex:
    def __init__(self, split: str):
        self.idx, self.ids, self.key, self.score = {}, {}, {}, {}
        for s in (2, 3):
            ids = pd.read_parquet(config.artifact("norm", f"{split}_s{s}.parquet"), columns=["entity_id"])["entity_id"].to_numpy(dtype=object)
            self.ids[s] = ids
            self.idx[s] = pd.Index(ids)
            z = np.load(config.artifact("graph", f"{split}_sib_s{s}.npz"))
            self.key[s], self.score[s] = z["key"], z["score"]

    def gkey(self, cand_ids: np.ndarray) -> np.ndarray:
        src = pd.Series(cand_ids).str[1].astype(int).values
        out = np.full(len(cand_ids), -1, dtype=np.int64)
        for s in (2, 3):
            m = src == s
            if m.any():
                pos = self.idx[s].get_indexer(cand_ids[m])
                out[m] = np.where(pos >= 0, s * OFF + pos, -1)
        return out

    def id_of(self, gkeys: np.ndarray) -> np.ndarray:
        src, pos = gkeys // OFF, gkeys % OFF
        return np.where(src == 2, self.ids[2][np.minimum(pos, len(self.ids[2]) - 1)],
                        self.ids[3][np.minimum(pos, len(self.ids[3]) - 1)])

    def neighbours(self, gkeys: np.ndarray):
        """(n, K) sibling keys and scores for the given global keys."""
        src, pos = gkeys // OFF, gkeys % OFF
        K = self.key[2].shape[1]
        k = np.full((len(gkeys), K), -1, dtype=np.int64)
        s = np.zeros((len(gkeys), K), dtype=np.float32)
        for q in (2, 3):
            m = src == q
            k[m], s[m] = self.key[q][pos[m]], self.score[q][pos[m]]
        return k, s


def expand_and_support(pairs: pd.DataFrame, sib: SiblingIndex, p_conf: float = 0.8, min_sib: float = 0.5):
    """pairs: s1_id, cand_id, p (stage-2). Returns (new_pairs, support) where
    new_pairs = 2-hop candidates (s1_id, cand_id) not already present, and
    support = DataFrame aligned to pairs ∪ new_pairs with graph features."""
    pairs = pairs[["s1_id", "cand_id", "p"]].reset_index(drop=True)
    conf = pairs[pairs["p"] >= p_conf]
    ck = sib.gkey(conf["cand_id"].values)
    nk, ns = sib.neighbours(ck)
    K = nk.shape[1]
    edges = pd.DataFrame({
        "s1_id": np.repeat(conf["s1_id"].values, K),
        "via_p": np.repeat(conf["p"].values, K),
        "via": np.repeat(ck, K),
        "nkey": nk.ravel(),
        "sib": ns.ravel(),
    })
    edges = edges[(edges["nkey"] >= 0) & (edges["sib"] >= min_sib)]
    edges["cand_id"] = sib.id_of(edges["nkey"].values)
    edges = edges[edges["nkey"] != edges["via"]]
    edges["w"] = edges["via_p"] * edges["sib"]
    sup = edges.groupby(["s1_id", "cand_id"]).agg(gs_max=("w", "max"), gs_sum=("w", "sum"),
                                                   gs_cnt=("w", "size"), gs_sib_max=("sib", "max")).reset_index()
    have = set(zip(pairs["s1_id"].values, pairs["cand_id"].values))
    new = sup[[(a, b) not in have for a, b in zip(sup["s1_id"].values, sup["cand_id"].values)]][["s1_id", "cand_id"]]
    return new.reset_index(drop=True), sup


if __name__ == "__main__":
    import sys
    for sp in sys.argv[1:] or ["train", "test"]:
        build_siblings(sp)
