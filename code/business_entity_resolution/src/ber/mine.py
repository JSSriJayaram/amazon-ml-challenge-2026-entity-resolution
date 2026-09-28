"""Layer-2 lexicon: learn token variants from TRAINING ground-truth pairs only.

For each true (S1, S2/S3) pair, tokens present on only one side are aligned
(single leftover on each side, or best Jaro-Winkler / same skeleton); a variant is
kept when it is seen often and maps to one canonical form most of the time.
Canonical = the S1 token (S1 is the cleaner reference source)."""
import json
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd
from rapidfuzz.distance import JaroWinkler

from . import config, io
from .normalize import skeleton


def _align(a: List[str], b: List[str]) -> List[tuple]:
    sa, sb = set(a), set(b)
    ra, rb = [t for t in a if t not in sb], [t for t in b if t not in sa]
    if not ra or not rb:
        return []
    if len(ra) == 1 and len(rb) == 1:
        return [(rb[0], ra[0])]
    out = []
    for v in rb:
        best = max(ra, key=lambda c: (skeleton(c) == skeleton(v), JaroWinkler.normalized_similarity(c, v)))
        if skeleton(best) == skeleton(v) or JaroWinkler.normalized_similarity(best, v) >= 0.75:
            out.append((v, best))
    return out


def mine(n_pairs: int = 400_000, min_count: int = 15, min_ratio: float = 0.6) -> Dict[str, Dict[str, str]]:
    gt = io.load_ground_truth().sample(n_pairs, random_state=config.SEED)
    cols = ["entity_id", "name_core", "addr_words"]
    s1 = pd.read_parquet(config.artifact("norm", "train_s1.parquet"), columns=cols,
                         filters=[("entity_id", "in", list(set(gt.s1_id)))]).set_index("entity_id")
    tg = pd.concat([pd.read_parquet(config.artifact("norm", f"train_s{s}.parquet"), columns=cols,
                                    filters=[("entity_id", "in", list(set(gt.cand_id)))]) for s in (2, 3)]
                   ).set_index("entity_id")
    a, b = s1.reindex(gt.s1_id.values), tg.reindex(gt.cand_id.values)
    result = {}
    for field in ("name_core", "addr_words"):
        pair_n, var_n = Counter(), Counter()
        for x, y in zip(a[field].values, b[field].values):
            if not isinstance(x, str) or not isinstance(y, str):
                continue
            ta, tb = x.split(), y.split()
            var_n.update(set(tb) - set(ta))
            pair_n.update(_align(ta, tb))
        best: Dict[str, tuple] = {}
        for (v, c), n in pair_n.items():
            if v != c and len(v) > 1 and not v.isdigit() and n >= min_count and n / var_n[v] >= min_ratio:
                if v not in best or n > best[v][1]:
                    best[v] = (c, n)
        # a canonical form must not itself be remapped (avoid chains/cycles)
        result[field] = {v: c for v, (c, _) in best.items() if c not in best}
        print(f"[mine] {field}: {len(result[field])} variants")
    return result


def main() -> None:
    res = mine()
    path = config.artifact("lexicon", "mined_aliases.json")
    json.dump(res, open(path, "w"), indent=0, ensure_ascii=False)
    for f, d in res.items():
        top = sorted(d.items(), key=lambda kv: -len(kv[0]))[:5]
        print(f, list(d.items())[:25])


if __name__ == "__main__":
    main()
