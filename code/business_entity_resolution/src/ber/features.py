"""Pair features. Unknown values are NaN (not 0): 'missing' is not 'different'.

Groups: name similarity, address similarity, retrieval/context, frequency."""
import re
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import config

def _load_noise_words(min_ratio_file="noise_words.json"):
    """Words the data generator adds/drops (learned from training pairs by EDA,
    kept when they appear on only one side of a true pair >= 60% of the time)."""
    import json
    p = config.ARTIFACT_DIR / "lexicon" / min_ratio_file
    if not p.exists():  # fall back to the copy shipped with the repository
        p = config.PKG_ROOT / "lexicon" / min_ratio_file
    return set(json.load(open(p))) if p.exists() else set()


NOISE_WORDS = _load_noise_words()
# Country-specific generator noise mined WITHOUT labels from confident test matches
# (France has no training data). 'trading' is noise in France but a real name part
# in India ('Shiv Trading'), hence per-country.
NOISE_BY_COUNTRY = {"France": NOISE_WORDS | {"et", "fils", "associes", "nee", "known", "trading", "business",
                                             "labs", "sys", "frs", "cb", "fs"}}


def _distinctive(name_core: str, country: str = None) -> str:
    noise = NOISE_BY_COUNTRY.get(country, NOISE_WORDS)
    return " ".join(t for t in name_core.split() if t not in noise and len(t) > 1)


def _num_best_sim(a: str, b: str) -> float:
    """Best digit-level similarity between any two house/unit numbers ('8209' vs '209')."""
    na, nb = [x for x in a.split() if len(x) >= 2], [x for x in b.split() if len(x) >= 2]
    if not na or not nb:
        return np.nan
    return max(Levenshtein.normalized_similarity(x, y) for x in na[:3] for y in nb[:3])


_LOOSE = str.maketrans({"b": "p", "v": "p", "w": "p", "f": "p", "d": "t", "g": "k", "q": "k",
                        "j": "s", "z": "s", "y": "", "h": ""})


def _loose_token(t: str) -> str:
    """Pronunciation key robust to Indic transliteration (voicing is not written in
    Tamil; aspiration/aytham appear as 'h'; soft c/g sound like s/j)."""
    if not t or t.isdigit():
        return t
    t = t.replace("ght", "t").replace("sh", "s").replace("ch", "s").replace("ph", "p").replace("x", "ks")
    t = re.sub(r"c(?=[eiy])", "s", t).replace("c", "k")
    t = re.sub(r"g(?=[ei])", "s", t)
    t = t.translate(_LOOSE)
    head = "a" if t[:1] in "aeiou" else t[:1]
    t = head + re.sub(r"[aeiou]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", t)


def loose_skeleton(name_core: str) -> str:
    """'bright infra' ~ 'piraitt inhpraa'; 'shiv trading' ~ 'civ ttireetting'."""
    return " ".join(k for k in (_loose_token(t) for t in name_core.split()) if k)


REC_COLS = ["entity_id", "name_core", "name_legal", "name_nospace", "name_skel", "name_sorted",
            "name_is_domain", "addr_clean", "addr_nums", "addr_first_num", "addr_street_words",
            "addr_city", "addr_skel", "addr_missing"]


def _jacc(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / len(sa | sb)


def _overlap(a: str, b: str) -> float:
    """|A∩B| / min(|A|,|B|): tolerant of one side being partial."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return np.nan
    return len(sa & sb) / min(len(sa), len(sb))


def _sim(fn, a: str, b: str) -> float:
    return fn(a, b) if a and b else np.nan


def _prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _pair_block(block: pd.DataFrame) -> pd.DataFrame:
    f: Dict[str, List[float]] = {k: [] for k in (
        "n_jw", "n_lev", "n_tset", "n_tsort", "n_partial", "n_jacc", "n_overlap",
        "ns_lev", "ns_jw", "ns_eq", "ns_contains", "ns_prefix",
        "sk_lev", "sk_tset", "sk_jacc", "sk_eq",
        "legal_eq", "legal_conflict", "legal_a", "legal_b",
        "a_tset", "a_lev", "a_partial", "st_jacc", "st_overlap", "st_tset", "ask_jacc",
        "num_first_eq", "num_first_lev", "num_jacc", "num_overlap", "num_conflict",
        "city_eq", "city_jw",
        "dn_tset", "dn_jacc", "dn_eq", "dn_lev", "dn_len_a", "dn_len_b", "num_first_contains", "num_best_sim",
        "lsk_tset", "lsk_eq", "lsk_lev", "lsk_jacc")}
    ctry = block["country"].values if "country" in block.columns else [None] * len(block)
    cols = [block[c].values for c in REC_COLS[1:] if c != "addr_missing"]
    colsb = [block[c + "_b"].values for c in REC_COLS[1:] if c != "addr_missing"]
    for i in range(len(block)):
        (nc, lg, ns, sk, nso, _dom, ac, nums, fn, st, city, ask) = (c[i] for c in cols)
        (nc2, lg2, ns2, sk2, nso2, _dom2, ac2, nums2, fn2, st2, city2, ask2) = (c[i] for c in colsb)
        f["n_jw"].append(_sim(JaroWinkler.normalized_similarity, nc, nc2))
        f["n_lev"].append(_sim(Levenshtein.normalized_similarity, nc, nc2))
        f["n_tset"].append(_sim(fuzz.token_set_ratio, nc, nc2) / 100)
        f["n_tsort"].append(_sim(fuzz.token_sort_ratio, nc, nc2) / 100)
        f["n_partial"].append(_sim(fuzz.partial_ratio, nc, nc2) / 100)
        f["n_jacc"].append(_jacc(nc, nc2))
        f["n_overlap"].append(_overlap(nc, nc2))
        f["ns_lev"].append(_sim(Levenshtein.normalized_similarity, ns, ns2))
        f["ns_jw"].append(_sim(JaroWinkler.normalized_similarity, ns, ns2))
        f["ns_eq"].append(float(bool(ns) and ns == ns2))
        f["ns_contains"].append(float(bool(ns) and bool(ns2) and (ns in ns2 or ns2 in ns)))
        f["ns_prefix"].append(_prefix_len(ns, ns2) / max(len(ns), len(ns2), 1))
        f["sk_lev"].append(_sim(Levenshtein.normalized_similarity, sk, sk2))
        f["sk_tset"].append(_sim(fuzz.token_set_ratio, sk, sk2) / 100)
        f["sk_jacc"].append(_jacc(sk, sk2))
        f["sk_eq"].append(float(bool(sk) and sk == sk2))
        f["legal_a"].append(float(bool(lg)))
        f["legal_b"].append(float(bool(lg2)))
        f["legal_eq"].append(float(lg == lg2) if lg and lg2 else np.nan)
        f["legal_conflict"].append(float(not set(lg.split()) & set(lg2.split())) if lg and lg2 else np.nan)
        f["a_tset"].append(_sim(fuzz.token_set_ratio, ac, ac2) / 100)
        f["a_lev"].append(_sim(Levenshtein.normalized_similarity, ac, ac2))
        f["a_partial"].append(_sim(fuzz.partial_ratio, ac, ac2) / 100)
        f["st_jacc"].append(_jacc(st, st2))
        f["st_overlap"].append(_overlap(st, st2))
        f["st_tset"].append(_sim(fuzz.token_set_ratio, st, st2) / 100)
        f["ask_jacc"].append(_jacc(ask, ask2))
        f["num_first_eq"].append(float(fn == fn2) if fn and fn2 else np.nan)
        f["num_first_lev"].append(_sim(Levenshtein.normalized_similarity, fn, fn2))
        f["num_jacc"].append(_jacc(nums, nums2))
        f["num_overlap"].append(_overlap(nums, nums2))
        f["num_conflict"].append(float(not set(nums.split()) & set(nums2.split())) if nums and nums2 else np.nan)
        f["city_eq"].append(float(city == city2) if city and city2 else np.nan)
        f["city_jw"].append(_sim(JaroWinkler.normalized_similarity, city, city2))
        dn, dn2 = _distinctive(nc, ctry[i]), _distinctive(nc2, ctry[i])
        f["dn_tset"].append(_sim(fuzz.token_set_ratio, dn, dn2) / 100)
        f["dn_jacc"].append(_jacc(dn, dn2))
        f["dn_eq"].append(float(bool(dn) and dn.replace(" ", "") == dn2.replace(" ", "")))
        f["dn_lev"].append(_sim(Levenshtein.normalized_similarity, dn.replace(" ", ""), dn2.replace(" ", "")))
        f["dn_len_a"].append(float(len(dn.split())))
        f["dn_len_b"].append(float(len(dn2.split())))
        f["num_first_contains"].append(float(fn in fn2 or fn2 in fn) if fn and fn2 else np.nan)
        f["num_best_sim"].append(_num_best_sim(nums, nums2))
        lk, lk2 = loose_skeleton(nc), loose_skeleton(nc2)
        f["lsk_tset"].append(_sim(fuzz.token_set_ratio, lk, lk2) / 100)
        f["lsk_eq"].append(float(bool(lk) and lk.replace(" ", "") == lk2.replace(" ", "")))
        f["lsk_lev"].append(_sim(Levenshtein.normalized_similarity, lk.replace(" ", ""), lk2.replace(" ", "")))
        f["lsk_jacc"].append(_jacc(lk, lk2))
    out = pd.DataFrame(f, index=block.index, dtype=np.float32)
    out["n_len_a"] = block["name_core"].str.split().str.len().astype(np.float32).values
    out["n_len_b"] = block["name_core_b"].str.split().str.len().astype(np.float32).values
    out["ns_len_diff"] = (block["name_nospace"].str.len() - block["name_nospace_b"].str.len()).abs().astype(np.float32).values
    out["dom_b"] = block["name_is_domain_b"].astype(np.float32).values
    out["addr_miss_a"] = block["addr_missing"].astype(np.float32).values
    out["addr_miss_b"] = block["addr_missing_b"].astype(np.float32).values
    return out


# Stage-1 (candidate filter) uses only these fast string comparisons; formulas are
# identical to the ones in _pair_block so train and test values agree exactly.
CHEAP_STR = ["ns_lev", "n_tset", "sk_tset", "a_tset", "st_overlap", "num_first_eq", "num_overlap", "city_eq"]


def _cheap_block(block: pd.DataFrame) -> pd.DataFrame:
    g = lambda c: block[c].values
    nc, ns, sk, ac, st, nums, fn, city = (g(c) for c in (
        "name_core", "name_nospace", "name_skel", "addr_clean", "addr_street_words", "addr_nums",
        "addr_first_num", "addr_city"))
    nc2, ns2, sk2, ac2, st2, nums2, fn2, city2 = (g(c + "_b") for c in (
        "name_core", "name_nospace", "name_skel", "addr_clean", "addr_street_words", "addr_nums",
        "addr_first_num", "addr_city"))
    n = len(block)
    f = {k: np.empty(n, dtype=np.float32) for k in CHEAP_STR}
    for i in range(n):
        f["ns_lev"][i] = _sim(Levenshtein.normalized_similarity, ns[i], ns2[i])
        f["n_tset"][i] = _sim(fuzz.token_set_ratio, nc[i], nc2[i]) / 100
        f["sk_tset"][i] = _sim(fuzz.token_set_ratio, sk[i], sk2[i]) / 100
        f["a_tset"][i] = _sim(fuzz.token_set_ratio, ac[i], ac2[i]) / 100
        f["st_overlap"][i] = _overlap(st[i], st2[i])
        f["num_first_eq"][i] = float(fn[i] == fn2[i]) if fn[i] and fn2[i] else np.nan
        f["num_overlap"][i] = _overlap(nums[i], nums2[i])
        f["city_eq"][i] = float(city[i] == city2[i]) if city[i] and city2[i] else np.nan
    return pd.DataFrame(f, index=block.index)


def string_features(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame,
                    n_jobs: int = None, chunk: int = 200_000, cheap: bool = False) -> pd.DataFrame:
    """pairs: s1_id, cand_id. s1/tgt: normalized record tables (tgt = S2 ∪ S3).
    cheap=True computes only CHEAP_STR (stage-1 filter)."""
    n_jobs = n_jobs or config.N_JOBS
    a = s1[REC_COLS + (["country"] if "country" in s1.columns else [])].set_index("entity_id")
    b = tgt[REC_COLS].set_index("entity_id").add_suffix("_b")
    block = pairs[["s1_id", "cand_id"]].join(a, on="s1_id").join(b, on="cand_id")
    parts = [block.iloc[i:i + chunk] for i in range(0, len(block), chunk)]
    with Pool(n_jobs) as pool:
        res = pool.map(_cheap_block if cheap else _pair_block, parts)
    return pd.concat(res)


def context_features(pairs: pd.DataFrame, freq_s1: pd.Series, freq_tgt: pd.Series,
                     reverse: bool = False) -> pd.DataFrame:
    """Retrieval/context features. `pairs` holds one row per (s1_id, cand_id) with
    per-retriever score_/rank_ columns (NaN when that retriever missed the pair).

    reverse=True adds competition-between-S1s features; only valid when `pairs`
    covers ALL S1 of the universe (not a query subsample), else train/test differ.
    freq_*: record counts per (country, name_core) over the whole universe."""
    out = pd.DataFrame(index=pairs.index)
    score_cols = [c for c in pairs.columns if c.startswith("score_")]
    out["n_retrievers"] = pairs[score_cols].notna().sum(1).astype(np.float32)
    best = pairs[score_cols].max(1)
    out["best_score"] = best.astype(np.float32)
    out["cand_src"] = pairs["cand_id"].str[1].astype(np.float32)
    g = best.groupby([pairs["s1_id"], out["cand_src"]])
    out["rank_in_src"] = g.rank(ascending=False, method="min").astype(np.float32)
    out["gap_to_best_src"] = (g.transform("max") - best).astype(np.float32)
    out["n_cands_src"] = g.transform("size").astype(np.float32)
    if reverse:  # how does this S1 rank among all S1s that retrieved this candidate?
        gc = best.groupby(pairs["cand_id"])
        out["rev_rank"] = gc.rank(ascending=False, method="min").astype(np.float32)
        out["rev_gap"] = (gc.transform("max") - best).astype(np.float32)
        out["rev_n"] = gc.transform("size").astype(np.float32)
        out["mutual_best"] = ((out["rank_in_src"] == 1) & (out["rev_rank"] == 1)).astype(np.float32)
    # name frequency in the universe: common names need address evidence
    f1, ft = freq_s1, freq_tgt
    k1 = pd.MultiIndex.from_frame(pairs[["country", "name_core"]])
    kt = pd.MultiIndex.from_frame(pairs[["country", "name_core_b"]].set_axis(["country", "name_core"], axis=1))
    out["name_freq_s1"] = np.log1p(f1.reindex(k1).fillna(0).values).astype(np.float32)
    out["name_freq_tgt"] = np.log1p(ft.reindex(kt).fillna(0).values).astype(np.float32)
    for c in pairs.columns:
        if c.startswith(("score_", "rank_")):
            out[c] = pairs[c].astype(np.float32)
    return out
