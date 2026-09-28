"""Candidate generation. Each retriever turns a record into a sparse TF-IDF vector
(its "view"); S1 queries are matched to S2/S3 targets inside the same country with a
multi-threaded sparse top-K product. Retriever outputs are unioned downstream."""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config


def _split(s: str) -> List[str]:
    return s.split() if s else []


def _bigrams(prefix: str, toks: List[str]) -> List[str]:
    """Adjacent-word pairs: rare even when both words are common, so they keep
    specificity after frequent unigrams are capped by max_df."""
    return [f"{prefix}{a}_{b}" for a, b in zip(toks, toks[1:])]


def _addr_tokens(r, skel: bool = False) -> List[str]:
    st = _split(r.addr_skel if skel else r.addr_street_words)
    nums = _split(r.addr_nums)
    p = "s" if skel else "a"
    return ([f"{p}:{t}" for t in st] + _bigrams(p.upper() + ":", st)
            + [f"d:{t}" for t in nums]
            + [f"D:{n}_{w}" for n in nums[:2] for w in st[:2]]
            + ([f"c:{r.addr_city}"] if r.addr_city else []))


def view_combo(r) -> List[str]:
    """Name words/pairs + name skeletons + address tokens, namespaced."""
    nc = _split(r.name_core)
    return ([f"n:{t}" for t in nc] + _bigrams("N:", nc)
            + [f"k:{t}" for t in _split(r.name_skel) if len(t) > 1]
            + ([f"C:{r.addr_city}_{nc[0]}"] if r.addr_city and nc else [])
            + _addr_tokens(r))


def view_address(r) -> List[str]:
    """Address only: finds renamed businesses / DBA names at the same address."""
    return _addr_tokens(r) + _addr_tokens(r, skel=True)


def view_name_addr_skel(r) -> List[str]:
    """Skeleton-only view: survives transliteration and vowel typos in both fields."""
    prefix = [f"p:{r.name_nospace[:6]}"] if r.name_nospace else []
    sk = _split(r.name_skel)
    return [f"k:{t}" for t in sk] + _bigrams("K:", sk) + prefix + _addr_tokens(r, skel=True)


def char_ngrams(s: str, n: int = 3) -> List[str]:
    s = f" {s} "
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def view_name_char(r) -> List[str]:
    """Char 3-grams of the core name (typos, spacing) + street/number tokens to split chains."""
    return char_ngrams(r.name_nospace, 3) + _addr_tokens(r)


@dataclass
class Retriever:
    name: str
    view: Callable
    k: int = 20
    max_df: int = 5_000  # drop tokens more frequent than this in the country block (speed)
    fields: List[str] = field(default_factory=lambda: [
        "name_core", "name_skel", "name_nospace", "addr_street_words", "addr_nums", "addr_city", "addr_skel"])


RETRIEVERS: Dict[str, Retriever] = {
    "combo": Retriever("combo", view_combo, k=20),
    "skel": Retriever("skel", view_name_addr_skel, k=20),
    "char": Retriever("char", view_name_char, k=20),
    "addr": Retriever("addr", view_address, k=20),
}


N_HASH = 2 ** 23  # hashed vocabulary size; collisions are negligible at this size


def _hashed_counts(df: pd.DataFrame, r: Retriever, chunk: int = 250_000) -> sp.csr_matrix:
    """Term counts via feature hashing, built chunk by chunk so token lists for
    millions of records are never held in memory at once."""
    hv = HashingVectorizer(n_features=N_HASH, analyzer=r.view, alternate_sign=False,
                           norm=None, dtype=np.float32)
    parts = [hv.transform(df[r.fields].iloc[i:i + chunk].itertuples(index=False))
             for i in range(0, len(df), chunk)]
    return sp.vstack(parts, format="csr") if parts else sp.csr_matrix((0, N_HASH), dtype=np.float32)


def _doc_freq(frames: List[pd.DataFrame], r: Retriever) -> Tuple[np.ndarray, int]:
    df = np.zeros(N_HASH, dtype=np.int64)
    n = 0
    for f in frames:
        for i in range(0, len(f), 250_000):
            m = _hashed_counts(f.iloc[i:i + 250_000], r)
            df += np.bincount(m.indices, minlength=N_HASH)
            n += m.shape[0]
    return df, n


def _tfidf(m: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """Sublinear tf * idf (0 for capped tokens), then L2-normalize rows."""
    m = m.copy()
    m.data = (1.0 + np.log(m.data)) * idf[m.indices]
    m.eliminate_zeros()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags((1.0 / norms).astype(np.float32)) @ m


def retrieve(s1: pd.DataFrame, targets: Dict[int, pd.DataFrame], r: Retriever,
             universe_s1: pd.DataFrame = None, n_threads: int = None) -> pd.DataFrame:
    """Top-k targets per S1 query for each country present in `s1`.

    `universe_s1` (all S1 records of the split) is used for IDF so that a query
    subsample sees the same weights as the full run. Returns columns
    s1_id, cand_id, score, rank, retriever."""
    n_threads = n_threads or config.N_JOBS
    universe_s1 = s1 if universe_s1 is None else universe_s1
    out = []
    for country, q in s1.groupby("country", sort=False):
        t0 = time.time()
        tgts = {s: t[t["country"] == country] for s, t in targets.items()}
        dfreq, n = _doc_freq([universe_s1[universe_s1["country"] == country], *tgts.values()], r)
        idf = (np.log((1 + n) / (1 + dfreq)) + 1).astype(np.float32)
        idf[dfreq > r.max_df] = 0.0
        del dfreq
        Q = _tfidf(_hashed_counts(q, r), idf)
        for src, t in tgts.items():
            if len(t) == 0:
                continue
            T = _tfidf(_hashed_counts(t, r), idf).T.tocsr()
            M = sp_matmul_topn(Q, T, top_n=r.k, sort=True, n_threads=n_threads).tocoo()
            del T
            out.append(pd.DataFrame({
                "s1_id": q["entity_id"].values[M.row],
                "cand_id": t["entity_id"].values[M.col],
                "s1_pos": q.index.values[M.row].astype(np.int32),  # row in the S1 frame passed in
                "cand_pos": t.index.values[M.col].astype(np.int32),  # row in its source frame
                "src": np.int8(src),
                "score": M.data.astype(np.float32),
                "rank": _row_ranks(M),
            }))
        print(f"  [{r.name}] {country}: {len(q):,} queries in {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(out, ignore_index=True)
    res["retriever"] = r.name
    return res


def _row_ranks(M: sp.coo_matrix) -> np.ndarray:
    """1-based rank of each entry within its row, by descending score."""
    order = np.lexsort((-M.data, M.row))
    ranks = np.empty(len(order), dtype=np.int16)
    row_sorted = M.row[order]
    starts = np.r_[0, np.flatnonzero(np.diff(row_sorted)) + 1]
    pos = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    ranks[order] = pos + 1
    return ranks
