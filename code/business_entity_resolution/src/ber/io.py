"""Loading source/ground-truth TSVs and writing submission files."""
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from . import config

# dtype=str keeps IDs/PINs intact; keep_default_na=False stops "" and "NA ..." becoming NaN;
# QUOTE_NONE stops a stray '"' in an address from swallowing following lines.
READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def load_source(split: str, source: int) -> pd.DataFrame:
    df = pd.read_csv(config.source_path(split, source), **READ_KW)
    for c in ("business_name", "business_address", "country"):
        df[c] = df[c].str.strip()
    df["source"] = source
    return df


def load_ground_truth() -> pd.DataFrame:
    """Long format: one row per (s1_id, cand_id) true pair."""
    gt = pd.read_csv(config.ground_truth_path(), **READ_KW)
    gt["cand_id"] = gt["matched_entity_ids"].str.split(",")
    long = gt[["source1_entity_id", "cand_id"]].explode("cand_id")
    long = long[long["cand_id"].notna() & (long["cand_id"] != "")]
    return long.rename(columns={"source1_entity_id": "s1_id"}).reset_index(drop=True)


def truth_dict(gt_long: pd.DataFrame, s1_ids: Iterable[str]) -> Dict[str, set]:
    """s1_id -> set of true matches, including empty sets for singletons."""
    d = {s: set() for s in s1_ids}
    for s1, c in zip(gt_long["s1_id"].values, gt_long["cand_id"].values):
        if s1 in d:
            d[s1].add(c)
    return d


def write_id_lists(path: Path, s1_ids: List[str], lists: Dict[str, List[str]], header: str) -> None:
    """One row per S1 id (in given order), de-duplicated S2/S3 ids, empty when none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for s in s1_ids:
            ids = [i for i in dict.fromkeys(lists.get(s, ())) if i.startswith(("S2-", "S3-"))]
            f.write(f"{s}\t{','.join(ids)}\n")
