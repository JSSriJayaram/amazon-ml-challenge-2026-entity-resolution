"""Multi-view normalization of names and addresses.

Every record keeps several views because different comparisons need different
amounts of cleaning (see docs): raw -> clean -> canonical -> core -> skeleton.
"""
import re
import unicodedata
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from unidecode import unidecode

from . import config
from .lexicons import (ABBREV_NAME, CITY_ALIASES, LEGAL, LEGAL_FORMS, LEGAL_PHRASES,
                       NULL_COMPONENTS, REGION_PHRASES, STREET)

# Layer-2 variants mined from training pairs (ber.mine); empty until mined.
def _load_mined():
    import json
    p = config.ARTIFACT_DIR / "lexicon" / "mined_aliases.json"
    if not p.exists():  # fall back to the copy shipped with the repository
        p = config.PKG_ROOT / "lexicon" / "mined_aliases.json"
    if not p.exists():
        return {}, {}
    d = json.load(open(p))
    return d.get("name_core", {}), d.get("addr_words", {})


MINED_NAME, MINED_ADDR = _load_mined()

# ---------------------------------------------------------------- shared cleaning
_DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|net|org|in|co|fr|biz|info|us|io)\b")
_DOTTED_INITIAL = re.compile(r"(?<=\b[a-z])\.(?=\s?[a-z]\b)")
_APOS = re.compile(r"(?<=[a-z])['`’](?=[a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9/\- ]+")
_SPACES = re.compile(r"\s+")
_SINGLE_LETTER_RUN = re.compile(r"\b(?:[a-z] ){1,}[a-z]\b")
_NUM = re.compile(r"\d+")
_OCR_LN = re.compile(r"\bln(?=[bcdfghjkmnpqrstvwxz])")
_OCR_MAP = str.maketrans({"1": "l", "0": "o", "5": "s", "3": "e"})


def to_ascii(s: str) -> str:
    """Unicode NFKC -> transliterate any script to ASCII -> lowercase."""
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def basic_clean(s: str) -> str:
    s = s.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    s = _APOS.sub("", s)
    s = _DOTTED_INITIAL.sub("", s)
    s = _NON_ALNUM.sub(" ", s)
    s = s.replace(" - ", " ").replace(" / ", " ")
    return _SPACES.sub(" ", s).strip(" -/")


def join_single_letters(s: str) -> str:
    """'m g road' -> 'mg road' (spaced initials)."""
    return _SINGLE_LETTER_RUN.sub(lambda m: m.group(0).replace(" ", ""), s)


def skeleton(tok: str) -> str:
    """Consonant skeleton that is robust to transliteration/vowel noise:
    'praaivett' and 'private' -> 'prvt'; 'lkssmii' and 'laxmi' -> 'lksm'."""
    if not tok or tok.isdigit():
        return tok
    t = (tok.replace("ph", "f").replace("x", "ks").replace("q", "k").replace("ck", "k")
         .replace("c", "k").replace("w", "v").replace("z", "j").replace("y", "i"))
    t = re.sub(r"([bdgjkprstv])h", r"\1", t)
    t = t[0] + re.sub(r"[aeiou]", "", t[1:])
    return re.sub(r"(.)\1+", r"\1", t)


def _collapse_repeats(toks: List[str]) -> List[str]:
    out = []
    for t in toks:
        if not out or out[-1] != t:
            out.append(t)
    return out


# ---------------------------------------------------------------- names
def _fix_ocr_token(t: str) -> str:
    if any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and not re.fullmatch(r"\d+(st|nd|rd|th)", t):
        return t.translate(_OCR_MAP)
    return t


# Legal words seen through transliteration ('praaivett', 'limittedd') share a skeleton.
_LEGAL_SKEL = {"prvt": "pvt", "lmtd": "ltd", "lmt": "ltd", "inkrprtd": "inc", "krprtn": "corp",
               "kmpn": "co", "prnrsp": "llp", "prnrship": "llp"}


def normalize_name(raw: str, country: str) -> Dict[str, str]:
    s = to_ascii(raw)
    is_domain = bool(_DOMAIN.search(s))
    s = _DOMAIN.sub(r" \1 ", s)
    s = _OCR_LN.sub("in", s)
    s = join_single_letters(basic_clean(s))
    s = s.replace("-", " ").replace("/", " ")
    for phrase, canon in LEGAL_PHRASES:
        s = re.sub(rf"\b{phrase}\b", canon, s)
    toks = [_fix_ocr_token(t) for t in s.split()]
    toks = [ABBREV_NAME.get(t, t) for t in toks]
    toks = [MINED_NAME.get(t, t) for t in toks]
    canon, legal, core = [], [], []
    ctry = country.lower().strip()
    for t in toks:
        if t not in LEGAL and len(t) > 4:
            t = _LEGAL_SKEL.get(skeleton(t), t)
        for u in LEGAL.get(t, t).split():
            canon.append(u)
            if u in LEGAL_FORMS:
                legal.append(u)
            elif u != ctry and u != "and":
                core.append(u)
    core = _collapse_repeats(core)
    return {
        "name_clean": " ".join(canon),
        "name_core": " ".join(core),
        "name_legal": " ".join(sorted(set(legal))),
        "name_sorted": " ".join(sorted(set(core))),
        "name_nospace": "".join(core),
        "name_skel": " ".join(skeleton(t) for t in core),
        "name_is_domain": is_domain,
    }


# ---------------------------------------------------------------- addresses
_REGION_RE = {
    c: re.compile(r"\b(" + "|".join(sorted(map(re.escape, d), key=len, reverse=True)) + r")\b")
    for c, d in REGION_PHRASES.items()
}
_STREET_CANON = set(STREET.values()) - {"n", "s", "e", "w", "near", "opp"}
_LANDMARK = {"near", "opp", "behind", "beside", "next", "pres", "face", "via"}
_PMB = re.compile(r"\b(?:pmb|po box|p o box|box)\s*\d+\b")
_NDEG = re.compile(r"\bn\s?deg\s?\d*\b")  # 'N°44' after transliteration
# house-number suffixes: '7 bis', '7 ter', '9 e rue' -> '7', '9 rue'
_NUM_SUFFIX = re.compile(r"\b(\d+)\s+(?:bis|ter|quater|[a-qs-z])\b(?=\s|$)")


# Context-dependent abbreviations: 'st'/'ste' = saint and 'dr' = doctor when they
# introduce a name ('St Joseph Rd', '12 Dr Ambedkar Marg'); street/drive when they
# close one ('Nethaji St', 'Rita Blanca Dr').
_SAINT = {"st": "saint", "ste": "saint", "saint": "saint", "sainte": "saint", "snt": "saint"}
_TITLE = {"dr": "doctor"}


def _contextual(toks: List[str]) -> List[str]:
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        prev = toks[i - 1] if i > 0 else ""
        opens_name = (nxt.isalpha() and nxt not in STREET and (not prev or prev.isdigit() or prev in STREET))
        if t in ("saint", "sainte"):
            out.append("saint")
        elif t in _SAINT and opens_name:
            out.append("saint")
        elif t in _TITLE and opens_name:
            out.append(_TITLE[t])
        else:
            out.append(t)
    return out


def normalize_address(raw: str, country: str) -> Dict[str, str]:
    s = to_ascii(raw)
    s = _PMB.sub(" ", s)
    s = _NDEG.sub(" ", s)
    region = REGION_PHRASES.get(country)
    if region:
        s = _REGION_RE[country].sub(lambda m: region[m.group(1)], s)
    comps = []
    for c in s.split(","):
        c = _NUM_SUFFIX.sub(r"\1", basic_clean(c))
        c = join_single_letters(c)
        if not c or c in NULL_COMPONENTS:
            continue
        toks = [CITY_ALIASES.get(t, STREET.get(t, t)) for t in _contextual(c.split())]
        toks = [MINED_ADDR.get(t, t) for t in toks]
        toks = [t for t in toks if t not in NULL_COMPONENTS]
        if toks:
            comps.append(" ".join(toks))
    region_codes = set(region.values()) if region else set()
    nums = [str(int(n)) for n in _NUM.findall(" ".join(comps))]
    words = [t for c in comps for t in re.split(r"[\s/-]+", c) if t and not t.isdigit()]
    # city guess: last component (digits dropped) that is not a region/country code,
    # a landmark phrase, or a street line
    ctry = country.lower().strip()
    city = ""
    for c in reversed(comps):
        toks = [t for t in c.split() if not t.isdigit()]
        if (toks and " ".join(toks) not in region_codes and toks != [ctry]
                and toks[0] not in _LANDMARK and not _STREET_CANON.intersection(toks)
                and not any(ch.isdigit() for ch in "".join(toks))):
            city = " ".join(toks)
            break
    street_words = [w for w in words if w not in STREET.values() and w not in region_codes and not re.fullmatch(r"\d+(st|nd|rd|th)", w)]
    return {
        "addr_clean": " ".join(comps),
        "addr_nums": " ".join(nums),
        "addr_first_num": nums[0] if nums else "",
        "addr_words": " ".join(words),
        "addr_street_words": " ".join(street_words),
        "addr_city": city,
        "addr_skel": " ".join(skeleton(w) for w in street_words),
    }


# ---------------------------------------------------------------- frames
def _normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    names = [normalize_name(n, c) for n, c in zip(df["business_name"].values, df["country"].values)]
    addrs = [normalize_address(a, c) for a, c in zip(df["business_address"].values, df["country"].values)]
    out = pd.concat([pd.DataFrame(names, index=df.index), pd.DataFrame(addrs, index=df.index)], axis=1)
    out.insert(0, "entity_id", df["entity_id"].values)
    out["country"] = df["country"].values
    out["name_missing"] = df["business_name"].eq("").values
    out["addr_missing"] = out["addr_clean"].eq("").values
    return out


def normalize_frame(df: pd.DataFrame, n_jobs: Optional[int] = None, chunk: int = 50_000) -> pd.DataFrame:
    n_jobs = n_jobs or config.N_JOBS
    parts = [df.iloc[i:i + chunk] for i in range(0, len(df), chunk)]
    if n_jobs == 1 or len(parts) == 1:
        res = [_normalize_chunk(p) for p in parts]
    else:
        with Pool(n_jobs) as pool:
            res = pool.map(_normalize_chunk, parts)
    out = pd.concat(res, ignore_index=True)
    out["source"] = np.int8(df["source"].iloc[0]) if "source" in df else np.int8(0)
    return out
