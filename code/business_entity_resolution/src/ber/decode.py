"""Turn pair probabilities into per-S1 match lists, and tune the decision rule
directly on the competition metric (macro F0.5 over all S1 entities).

Rule: (1) each S2/S3 record keeps only its best S1 (ground truth never assigns a
record to two S1s); (2) keep the top candidate if p >= t1; (3) keep further
candidates if p >= t2 and p >= alpha * p_top."""
import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd


@dataclass
class DecodeParams:
    t1: float = 0.5
    t2: float = 0.5
    alpha: float = 0.0
    one_to_one: bool = True


def _prepare(pairs: pd.DataFrame, one_to_one: bool) -> pd.DataFrame:
    p = pairs[["s1_id", "cand_id", "p"]].copy()
    if one_to_one:
        # deterministic: highest p wins, exact ties broken by s1_id (reproducibility, not evidence)
        p = p.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True], kind="mergesort")
        p = p.drop_duplicates("cand_id")
    p = p.sort_values(["s1_id", "p"], ascending=[True, False])
    p["pos"] = p.groupby("s1_id").cumcount()
    p["p_top"] = p.groupby("s1_id")["p"].transform("max")
    return p


def _select(p: pd.DataFrame, prm: DecodeParams) -> np.ndarray:
    first = (p["pos"].values == 0) & (p["p"].values >= prm.t1)
    top_ok = p["p_top"].values >= prm.t1
    rest = ((p["pos"].values > 0) & top_ok & (p["p"].values >= prm.t2)
            & (p["p"].values >= prm.alpha * p["p_top"].values))
    return first | rest


def decode(pairs: pd.DataFrame, prm: DecodeParams, reassign: bool = False) -> Dict[str, List[str]]:
    if reassign:
        return decode_reassign(pairs, prm)
    p = _prepare(pairs, prm.one_to_one)
    sel = p[_select(p, prm)]
    return sel.groupby("s1_id")["cand_id"].agg(list).to_dict()


def decode_reassign(pairs: pd.DataFrame, prm: DecodeParams, max_iter: int = 6) -> Dict[str, List[str]]:
    """Eligibility-aware one-to-one: each S1 first selects its eligible candidates
    independently; only records selected by >1 S1 are disputed, the highest-p owner keeps
    them (ties -> s1_id), losing pairs are removed and the affected S1s re-decided
    (their top score / alpha gate may change) until no disputes remain. A record whose
    best owner would REJECT it is therefore not stranded."""
    P = pairs[["s1_id", "cand_id", "p"]].reset_index(drop=True)
    banned = np.zeros(len(P), dtype=bool)
    key = pd.MultiIndex.from_frame(P[["s1_id", "cand_id"]])
    for _ in range(max_iter):
        q = _prepare(P[~banned], one_to_one=False)
        sel = q[_select(q, prm)]
        dup = sel[sel.duplicated("cand_id", keep=False)]
        if dup.empty:
            break
        dup = dup.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True], kind="mergesort")
        losers = dup[dup.duplicated("cand_id", keep="first")]
        banned |= key.isin(pd.MultiIndex.from_frame(losers[["s1_id", "cand_id"]]))
    return sel.groupby("s1_id")["cand_id"].agg(list).to_dict()


def _macro_from_counts(tp: np.ndarray, npred: np.ndarray, ntrue: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    f = np.where(ntrue == 0, (npred == 0).astype(float), f)
    return float(f.mean())


def tune(pairs: pd.DataFrame, truth: Mapping[str, set], grid: dict = None, verbose: bool = False):
    """Grid-search DecodeParams on labelled pairs (needs column `label`).
    Every S1 in `truth` counts, including ones with no candidates."""
    grid = grid or {
        "t1": np.round(np.arange(0.20, 0.91, 0.05), 2),
        "t2": np.round(np.arange(0.20, 0.96, 0.05), 2),
        "alpha": [0.0, 0.3, 0.5, 0.7],
    }
    s1_ids = list(truth)
    idx = pd.Index(s1_ids)
    ntrue = np.array([len(truth[s]) for s in s1_ids])
    best = (-1.0, None)
    for oto in (True, False):
        p = _prepare(pairs.assign(p=pairs["p"]), oto)
        lab = pairs.set_index(["s1_id", "cand_id"])["label"].reindex(
            pd.MultiIndex.from_frame(p[["s1_id", "cand_id"]])).values
        code = idx.get_indexer(p["s1_id"])
        for t1, t2, a in itertools.product(grid["t1"], grid["t2"], grid["alpha"]):
            prm = DecodeParams(t1, t2, a, oto)
            m = _select(p, prm)
            npred = np.bincount(code[m], minlength=len(idx))
            tp = np.bincount(code[m], weights=lab[m], minlength=len(idx))
            score = _macro_from_counts(tp, npred, ntrue)
            if score > best[0]:
                best = (score, prm)
    if verbose:
        print(f"best macro F0.5 = {best[0]:.4f} with {best[1]}")
    return best[1], best[0]
