"""France-only threshold variants of v10 (logit offset on France pairs only; US/India untouched).
Checks first that decoding the saved v10 scores reproduces the submitted v10 matches exactly."""
import shutil

import numpy as np
import pandas as pd

from ber import config, io
from ber.decode import DecodeParams, decode

PRM = DecodeParams(t1=0.8, t2=0.75, alpha=0.0, one_to_one=True)


def lg(x):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def main():
    te = pd.read_parquet(config.artifact("submissions", "v10", "test_p.parquet"))
    s1_ids = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id"])["entity_id"].tolist()
    fr = (te["country"] == "France").values
    ref = pd.read_csv(config.artifact("submissions", "v10", "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False)
    ref = dict(zip(ref.iloc[:, 0], ref.iloc[:, 1].map(lambda s: sorted(x for x in s.split(",") if x))))
    for name, d in (("v10_check", 0.0), ("v11_fr_strict", 0.75), ("v11_fr_loose", -0.75)):
        p = te["p"].values.copy()
        p[fr] = 1 / (1 + np.exp(-(lg(p[fr]) - d)))
        m = decode(te[["s1_id", "cand_id"]].assign(p=p), PRM)
        if name == "v10_check":
            diff = sum(sorted(m.get(s, [])) != ref[s] for s in s1_ids)
            print(f"[check] decoding saved scores vs submitted v10: {diff} S1 rows differ", flush=True)
            assert diff == 0
            continue
        owners = pd.Series([c for v in m.values() for c in v])
        assert not owners.duplicated().any()
        out = config.artifact("submissions", name)
        out.mkdir(parents=True, exist_ok=True)
        io.write_id_lists(out / "matching_results.tsv", s1_ids, m, "matched_entity_ids")
        shutil.copy(config.artifact("submissions", "v10", "candidate_pairs.tsv"), out / "candidate_pairs.tsv")
        cty = pd.read_parquet(config.artifact("norm", "test_s1.parquet"), columns=["entity_id", "country"])
        n = cty["entity_id"].map(lambda s: len(m.get(s, ()))).values
        for c in ("France", "US", "India"):
            k = (cty["country"] == c).values
            print(f"[{name}] {c}: empty {np.mean(n[k] == 0):.4f} mean {n[k].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
