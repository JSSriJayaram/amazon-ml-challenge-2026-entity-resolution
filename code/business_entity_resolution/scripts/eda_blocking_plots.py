"""Blocking EDA: what does each retriever contribute, how do they overlap, where do
true matches rank, and how well do retrieval scores separate matches?"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from ber import config, io

C = {"combo": "#2a78d6", "skel": "#eb6834", "addr": "#1baf7a", "UNION": "#0b0b0b"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})

def main():
    raw = pd.read_parquet(config.artifact("blocking", "eval_train_200000.parquet"),
                          columns=["s1_id", "cand_id", "score", "rank", "retriever"])
    keep = pd.Series(raw.s1_id.unique()).sample(50_000, random_state=0)
    raw = raw[raw.s1_id.isin(set(keep))]
    gt = io.load_ground_truth(); gt = gt[gt.s1_id.isin(set(keep))]
    key = set(zip(gt.s1_id, gt.cand_id)); n_true = len(gt)
    raw["y"] = [(a, b) in key for a, b in zip(raw.s1_id, raw.cand_id)]
    R = ["combo", "skel", "addr"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

    # 1 recall@k
    a = ax[0, 0]; ks = np.arange(1, 21)
    for r in R + ["UNION"]:
        d = raw if r == "UNION" else raw[raw.retriever == r]
        rec = [d[(d["rank"] <= k) & d.y][["s1_id", "cand_id"]].drop_duplicates().shape[0] / n_true for k in ks]
        a.plot(ks, rec, color=C[r], lw=2, label=f"{r}  (top-20: {rec[-1]:.1%})")
    a.set(title="Recall of true pairs vs. top-k per source", xlabel="k (per retriever, per source)",
          ylabel="share of true pairs retrieved", xlim=(1, 20), ylim=(0.4, 1.0))
    a.legend(frameon=False, loc="lower right")

    # 2 overlap of retrievers on found true pairs (k=20)
    a = ax[0, 1]
    pos = raw[raw.y].groupby(["s1_id", "cand_id"]).retriever.agg(lambda s: "+".join(sorted(set(s))))
    cnt = pos.value_counts()
    missed = n_true - len(pos)
    labels = list(cnt.index) + ["none (missed)"]; vals = list(cnt.values) + [missed]
    order = np.argsort(vals)
    a.barh([labels[i] for i in order], [vals[i] / n_true for i in order], color="#2a78d6", height=0.6)
    for i, j in enumerate(order):
        a.text(vals[j] / n_true, i, f" {vals[j] / n_true:.1%}", va="center", fontsize=9, color=INK)
    a.set(title="Which retrievers find each true pair (top-20)", xlabel="share of true pairs", xlim=(0, 1.0))
    a.grid(axis="y", visible=False)

    # 3 best rank of true pairs
    a = ax[1, 0]
    br = raw[raw.y].groupby(["s1_id", "cand_id"])["rank"].min()
    vc = br.value_counts().sort_index()
    a.bar(vc.index, vc.values / n_true, color="#2a78d6", width=0.7)
    a.set(title="Best rank of each true pair across retrievers", xlabel="best rank (1 = top)",
          ylabel="share of true pairs", xticks=range(1, 21))
    a.grid(axis="x", visible=False)
    a.text(10, (vc.values / n_true).max() * 0.8, f"rank 1: {vc.get(1, 0) / n_true:.1%}\nranks 11-20: "
           f"{vc[vc.index > 10].sum() / n_true:.1%}", color=INK)

    # 4 score separation (combo)
    a = ax[1, 1]
    d = raw[raw.retriever == "combo"]
    bins = np.linspace(0, 1, 51)
    a.hist(d.score[~d.y], bins=bins, density=True, histtype="step", lw=2, color="#e34948", label="non-match")
    a.hist(d.score[d.y], bins=bins, density=True, histtype="step", lw=2, color="#2a78d6", label="true match")
    a.set(title="combo retriever: score of matches vs non-matches", xlabel="cosine score", ylabel="density")
    a.legend(frameon=False)
    fig.suptitle("Blocking analysis - 50,000 training entities, 3 retrievers, top-20 per source", color=INK, fontsize=13)
    fig.tight_layout()
    out = config.artifact("reports", "blocking_eda.png")
    fig.savefig(out, dpi=130)
    print(out)
    print("overlap:", (cnt / n_true).round(4).to_dict(), "missed", round(missed / n_true, 4))

if __name__ == "__main__":
    main()
