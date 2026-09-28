"""Union retriever outputs into one row per (s1_id, cand_id) and attach features.

Memory rule: never materialize a full 10M-record universe as pandas strings.
Records are loaded only for ids present in the candidate set, and universe-level
statistics (name frequencies) are computed in Polars and reduced to needed keys."""
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from . import config
from .features import REC_COLS, context_features, string_features


def norm_path(split: str, src: int):
    return config.artifact("norm", f"{split}_s{src}.parquet")


def load_records(split: str, sources: Iterable[int], ids: Iterable[str], columns: List[str]) -> pd.DataFrame:
    ids = list(set(ids))
    parts = [pq.read_table(norm_path(split, s), columns=columns, filters=[("entity_id", "in", ids)]).to_pandas()
             for s in sources]
    return pd.concat(parts, ignore_index=True)


def name_freq(split: str, sources: Iterable[int], keys: pd.DataFrame) -> pd.Series:
    """Count of records per (country, name_core) in the universe, only for `keys`."""
    scan = pl.concat([pl.scan_parquet(norm_path(split, s)).select("country", "name_core") for s in sources])
    need = pl.from_pandas(keys.drop_duplicates())
    f = scan.group_by("country", "name_core").len().join(need.lazy(), on=["country", "name_core"]).collect()
    return f.to_pandas().set_index(["country", "name_core"])["len"]


def union_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: s1_id, cand_id, score, rank, retriever -> wide score_<r>/rank_<r> columns."""
    wide = raw.pivot(index=["s1_id", "cand_id"], columns="retriever", values=["score", "rank"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return wide.reset_index()


def build_features(pairs: pd.DataFrame, split: str, stage1_filter=None, cheap_only: bool = False) -> pd.DataFrame:
    """Context + string features for candidate pairs.

    stage1_filter(X_cheap) -> boolean mask: if given, cheap features are computed for
    all pairs, the filter prunes them, and full string features are computed only for
    survivors (the returned rows are exactly the pairs the stage-2 model scores)."""
    cols = REC_COLS + ["country"]
    s1 = load_records(split, [1], pairs["s1_id"], cols)
    tgt = load_records(split, [2, 3], pairs["cand_id"], cols)
    info = pairs.join(s1.set_index("entity_id")[["country", "name_core"]], on="s1_id")
    info = info.join(tgt.set_index("entity_id")[["name_core"]].add_suffix("_b"), on="cand_id")
    f1 = name_freq(split, [1], info[["country", "name_core"]])
    ft = name_freq(split, [2, 3], info[["country", "name_core_b"]].set_axis(["country", "name_core"], axis=1))
    ctx = pd.concat([pairs[["s1_id", "cand_id"]], context_features(info, f1, ft)], axis=1)
    if cheap_only:
        return pd.concat([ctx, string_features(pairs, s1, tgt, cheap=True)], axis=1)
    if stage1_filter is None:
        return pd.concat([ctx, string_features(pairs, s1, tgt)], axis=1)
    cheap = pd.concat([ctx, string_features(pairs, s1, tgt, cheap=True)], axis=1)
    keep = stage1_filter(cheap)
    surv = cheap[keep].reset_index(drop=True)
    full = string_features(surv[["s1_id", "cand_id"]], s1, tgt)
    return pd.concat([surv.drop(columns=[c for c in full.columns if c in surv.columns]), full], axis=1)


def add_labels(X: pd.DataFrame, gt_long: pd.DataFrame) -> pd.DataFrame:
    g = gt_long[gt_long["s1_id"].isin(set(X["s1_id"]))]
    key = set(zip(g["s1_id"].values, g["cand_id"].values))
    X["label"] = np.fromiter(((a, b) in key for a, b in zip(X["s1_id"].values, X["cand_id"].values)),
                             dtype=np.int8, count=len(X))
    return X


def add_full_strings(X: pd.DataFrame, split: str) -> pd.DataFrame:
    """Given a cheap feature table (ctx + CHEAP_STR), add the full string features."""
    cols = REC_COLS + ["country"]
    s1 = load_records(split, [1], X["s1_id"], cols)
    tgt = load_records(split, [2, 3], X["cand_id"], cols)
    full = string_features(X[["s1_id", "cand_id"]], s1, tgt)
    return pd.concat([X.drop(columns=[c for c in full.columns if c in X.columns]), full], axis=1)
