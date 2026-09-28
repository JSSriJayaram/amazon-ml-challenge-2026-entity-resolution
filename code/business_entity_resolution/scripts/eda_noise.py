"""Reverse-engineer the data generator: which noise operations turn a Source-1
record into its Source-2/3 copies, and how often (per source). Uses TRAIN only."""
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode

from ber import config, io

N_PAIRS = 30_000
TOK = re.compile(r"[a-z0-9]+")


def asc(s):
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def toks(s):
    return TOK.findall(asc(s))


def script(s):
    if s.isascii():
        return "ascii"
    if all(ord(c) < 0x250 for c in s):
        return "latin-accents"
    return "non-latin"


LEGAL = {"inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company", "llp", "lp",
         "pllc", "pc", "plc", "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "incorporated", "cie"}


def name_ops(a, b):
    ops = []
    if b == a:
        return ["identical"]
    ta, tb = toks(a), toks(b)
    ca, cb = [t for t in ta if t not in LEGAL], [t for t in tb if t not in LEGAL]
    sc = script(b)
    if sc != "ascii" and script(a) == "ascii":
        ops.append(f"script:{sc}")
    if b.lower() == a.lower() and b != a:
        ops.append("case-only")
    if re.search(r"\.(com|in|net|org|fr|co)\b", b.lower()):
        ops.append("domain-form")
    if re.search(r"\b(dba|d/b/a|formerly|aka|t/a)\b", b.lower()):
        ops.append("dba/formerly")
    if re.search(r"[\[\(].*[\]\)]", b) and not re.search(r"[\[\(].*[\]\)]", a):
        ops.append("brackets-added")
    if set(ta) & LEGAL != set(tb) & LEGAL:
        ops.append("legal-changed" if (set(ta) & LEGAL and set(tb) & LEGAL) else
                   ("legal-dropped" if set(ta) & LEGAL else "legal-added"))
    if ca and cb and sorted(ca) == sorted(cb) and ca != cb:
        ops.append("reordered")
    if len(set(cb)) < len(cb):
        ops.append("repeated-word")
    extra, miss = set(cb) - set(ca), set(ca) - set(cb)
    if extra and not miss:
        ops.append("tokens-added")
    if miss and not extra:
        ops.append("tokens-dropped")
    if extra and miss:
        if len(extra) == len(miss) and all(min(Levenshtein.distance(e, m) for m in miss) <= 2 for e in extra):
            ops.append("typo")
        elif fuzz.token_set_ratio(" ".join(ca), " ".join(cb)) < 50:
            ops.append("renamed")
        else:
            ops.append("tokens-replaced")
    if re.search(r"[a-z][0-9]|[0-9][a-z]", asc(b)) and not re.search(r"[a-z][0-9]|[0-9][a-z]", asc(a)):
        ops.append("ocr-digit")
    if len(cb) == 1 and len(cb[0]) <= 4 and len(ca) >= 2 and cb[0] == "".join(t[0] for t in ca)[:len(cb[0])]:
        ops.append("acronym")
    return ops or ["other"]


def comps(s):
    return [c.strip() for c in asc(s).split(",") if c.strip()]


def addr_ops(a, b):
    if not b.strip():
        return ["empty"]
    if b == a:
        return ["identical"]
    ops = []
    if b.lower() == a.lower():
        ops.append("case-only")
    ca, cb = comps(a), comps(b)
    if sorted(ca) == sorted(cb) and ca != cb:
        ops.append("components-reordered")
    if len(cb) < len(ca):
        ops.append("components-dropped")
    if len(cb) > len(ca):
        ops.append("components-added")
    na, nb = re.findall(r"\d+", asc(a)), re.findall(r"\d+", asc(b))
    if na and nb:
        if na[0] == nb[0]:
            pass
        elif na[0].lstrip("0") == nb[0].lstrip("0"):
            ops.append("num-zero-padded")
        elif set(na) & set(nb):
            ops.append("num-extra-or-reordered")
        else:
            ops.append("num-changed")
    elif na and not nb:
        ops.append("num-dropped")
    elif nb and not na:
        ops.append("num-added")
    if re.search(r"\b(pmb|suite|ste|unit|apt|flat|floor|fl)\b", asc(b)) and not re.search(
            r"\b(pmb|suite|ste|unit|apt|flat|floor|fl)\b", asc(a)):
        ops.append("unit/pmb-added")
    if re.search(r"<null>|\bnull\b", b.lower()):
        ops.append("null-token")
    if re.search(r"\b(near|opp|behind|beside)\b", asc(b)) and not re.search(r"\b(near|opp|behind|beside)\b", asc(a)):
        ops.append("landmark-added")
    if re.search(r"(#|\bno\.?|h\.?\s?no)\s?\d", asc(b)) and not re.search(r"(#|\bno\.?|h\.?\s?no)\s?\d", asc(a)):
        ops.append("num-prefix(#/No)")
    if script(b) == "non-latin":
        ops.append("script:non-latin")
    return ops or ["other(abbrev/typo)"]


def main():
    gt = io.load_ground_truth().sample(N_PAIRS, random_state=0)
    s1 = io.load_source("train", 1).set_index("entity_id")
    tg = pd.concat([io.load_source("train", s) for s in (2, 3)]).set_index("entity_id")
    a, b = s1.reindex(gt.s1_id.values), tg.reindex(gt.cand_id.values)
    src = gt.cand_id.str[:2].values
    ctry = a.country.values
    rows = []
    for i in range(len(gt)):
        for op in name_ops(a.business_name.values[i], b.business_name.values[i]):
            rows.append(("name", op, src[i], ctry[i]))
        for op in addr_ops(a.business_address.values[i], b.business_address.values[i]):
            rows.append(("addr", op, src[i], ctry[i]))
    r = pd.DataFrame(rows, columns=["field", "op", "src", "country"])
    n = pd.Series({"S2": (src == "S2").sum(), "S3": (src == "S3").sum()})
    for f in ("name", "addr"):
        t = r[r.field == f].groupby(["op", "src"]).size().unstack(fill_value=0)
        t = (t / n).round(3)
        t["India"] = (r[(r.field == f) & (r.country == "India")].op.value_counts() / (ctry == "India").sum()).round(3)
        t["US"] = (r[(r.field == f) & (r.country == "US")].op.value_counts() / (ctry == "US").sum()).round(3)
        print(f"\n=== {f.upper()} operations (share of true pairs) ===")
        print(t.fillna(0).sort_values("S2", ascending=False).to_string())
    # which tokens does the generator ADD to names?
    add = Counter()
    for x, y in zip(a.business_name.values, b.business_name.values):
        add.update(set(toks(y)) - set(toks(x)))
    print("\nmost common ADDED name tokens:", [w for w, _ in add.most_common(60)])
    drop = Counter()
    for x, y in zip(a.business_name.values, b.business_name.values):
        drop.update(set(toks(x)) - set(toks(y)))
    print("\nmost common DROPPED name tokens:", [w for w, _ in drop.most_common(40)])


if __name__ == "__main__":
    main()
