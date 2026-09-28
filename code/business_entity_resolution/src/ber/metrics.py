"""Competition metric (macro F0.5 per S1 entity) and blocking diagnostics."""
from typing import Dict, Iterable, Mapping

import numpy as np

BETA2 = 0.25  # beta = 0.5


def entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(preds: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> float:
    """Average over every S1 in `truth` (missing predictions count as empty)."""
    return float(np.mean([entity_f05(preds.get(s, ()), t) for s, t in truth.items()]))


def blocking_report(cands: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> Dict[str, float]:
    """Pair recall ceiling, mean candidates per S1, and best achievable macro F0.5
    if the matcher were perfect on this candidate set."""
    n_true = n_hit = n_cand = 0
    oracle = []
    for s, t in truth.items():
        c = set(cands.get(s, ()))
        t = set(t)
        n_true += len(t)
        n_hit += len(c & t)
        n_cand += len(c)
        oracle.append(entity_f05(c & t, t))
    return {
        "pair_recall": n_hit / max(n_true, 1),
        "mean_cands": n_cand / max(len(truth), 1),
        "oracle_f05": float(np.mean(oracle)),
    }
