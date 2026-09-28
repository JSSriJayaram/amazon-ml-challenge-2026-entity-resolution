"""Recall@K of each retriever and their union on a sample of train S1 queries,
searched against the FULL train S2/S3 pools (same distractor density as test)."""
import argparse
import time

import pandas as pd

from ber import config, io
from ber.blocking import RETRIEVERS, retrieve
from ber.metrics import blocking_report

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50_000)
ap.add_argument("--retrievers", default="combo,skel,char")
ap.add_argument("--seed", type=int, default=config.SEED)
ap.add_argument("--tag", default="", help="suffix for the output file (e.g. B for a second sample)")
args = ap.parse_args()

t = time.time()
s1 = pd.read_parquet(config.artifact("norm", "train_s1.parquet"))
tg = {s: pd.read_parquet(config.artifact("norm", f"train_s{s}.parquet")) for s in (2, 3)}
q = s1.sample(args.n, random_state=args.seed)
gt = io.load_ground_truth()
truth = io.truth_dict(gt[gt.s1_id.isin(set(q.entity_id))], q.entity_id)
print(f"loaded in {time.time() - t:.0f}s")

pairs = []
for name in args.retrievers.split(","):
    t = time.time()
    p = retrieve(q, tg, RETRIEVERS[name], universe_s1=s1)
    pairs.append(p)
    print(f"{name}: {time.time() - t:.0f}s, {len(p):,} pairs")

allp = pd.concat(pairs, ignore_index=True)
allp.to_parquet(config.artifact("blocking", f"eval_train_{args.n}{args.tag}.parquet"), index=False)
rows = []
for name in args.retrievers.split(",") + ["UNION"]:
    p = allp if name == "UNION" else allp[allp.retriever == name]
    for k in (1, 3, 5, 10, 20):
        pk = p[p["rank"] <= k]
        c = pk.groupby("s1_id").cand_id.agg(set).to_dict()
        rows.append({"retriever": name, "k_per_source": k, **blocking_report(c, truth)})
print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))
