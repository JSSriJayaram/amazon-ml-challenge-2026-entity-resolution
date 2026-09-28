"""Charts for the external review package (numbers from the experiment logs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "../../review_package/plots/"
BLUE, ORANGE, AQUA, RED, INK, MUTED, GRID, BG = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "figure.facecolor": BG, "axes.facecolor": BG})


def hbar(ax, labels, vals, fmt, color=BLUE, xlim=None):
    ax.barh(labels[::-1], vals[::-1], color=color, height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)


# 1 version progression
v = ["v1", "v2", "v3", "v4", "v5*", "v6"]
val = [0.9607, 0.9657, 0.9707, 0.9729, 0.9744, 0.9797]
lb = [0.949, 0.954, 0.960, 0.961, None, 0.971]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(v, val, color=BLUE, lw=2, marker="o", ms=8, label="validation (US+India, 5-fold OOF)")
lx = [x for x, y in zip(v, lb) if y is not None]; ly = [y for y in lb if y is not None]
ax.plot(lx, ly, color=ORANGE, lw=2, marker="o", ms=8, label="public leaderboard (incl. France)")
for x, y in zip(v, val):
    ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK)
for x, y in zip(lx, ly):
    ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.axhline(0.985, color=MUTED, lw=1, ls="--")
ax.text(0, 0.9855, "reported top teams: 0.985", color=MUTED, fontsize=9, va="bottom")
ax.set(title="Macro F0.5 by version (*v5 not submitted; v6 val = CE-covered entities)", ylabel="macro F0.5", ylim=(0.94, 0.99))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "1_version_progress.png", dpi=130)

# 2 model comparison (50k entities, identical features/folds)
m = ["XGBoost", "LightGBM", "LGBM+XGB avg", "MLP", "RandomForest", "ExtraTrees", "LogisticReg", "Rule (no learning)"]
f = [0.9610, 0.9607, 0.9611, 0.9501, 0.9426, 0.9421, 0.9196, 0.8314]
fig, ax = plt.subplots(figsize=(8, 4.5))
hbar(ax, m, f, "{:.4f}", xlim=(0.8, 0.98))
ax.set(title="Model comparison - same candidates, features, folds (50k entities)", xlabel="out-of-fold macro F0.5")
fig.tight_layout(); fig.savefig(OUT + "2_model_comparison.png", dpi=130)

# 3 error decomposition
e = ["missed: never retrieved", "missed: rejected by model", "false merge (non-singleton)", "singleton wrongly matched", "mixed"]
s = [0.34, 0.39, 0.14, 0.06, 0.06]
fig, ax = plt.subplots(figsize=(8, 3.8))
hbar(ax, e, s, "{:.0%}", color=RED, xlim=(0, 0.5))
ax.set(title="Where the lost score comes from (validation, loss = 1 - F0.5)", xlabel="share of lost score")
fig.tight_layout(); fig.savefig(OUT + "3_error_decomposition.png", dpi=130)

# 4 noise catalogue
nm = ["legal suffix dropped", "generic word swapped", "native script", "legal form changed", "generic words added",
      "renamed / DBA", "typo", "accents added", "brackets added", "case only", "domain form", "repeated word"]
nv = [0.23, 0.19, 0.075, 0.087, 0.10, 0.078, 0.064, 0.066, 0.063, 0.061, 0.051, 0.041]
am = ["components dropped", "components added", "extra numbers", "number prefix (#/No)", "house number changed",
      "house number dropped", "address empty", "zero padding", "<NULL> token", "reordered"]
av = [0.22, 0.05, 0.097, 0.085, 0.048, 0.05, 0.042, 0.037, 0.025, 0.03]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
hbar(ax[0], nm, nv, "{:.1%}", xlim=(0, 0.3)); ax[0].set(title="Name noise (share of true pairs)")
hbar(ax[1], am, av, "{:.1%}", color=AQUA, xlim=(0, 0.3)); ax[1].set(title="Address noise (share of true pairs)")
fig.suptitle("Data generator noise catalogue - 30,000 true training pairs", color=INK)
fig.tight_layout(); fig.savefig(OUT + "4_noise_catalogue.png", dpi=130)

# 5 cascade trade-off
c = pd.read_csv("../../artifacts/reports/cascade_tradeoff_strfeat.csv")
base = c.iloc[0]; r = c.iloc[1:]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(r.cands_per_s1, r.final_f05, s=60, color=BLUE, zorder=3, label="stage-1 filter settings")
ax.scatter([base.cands_per_s1], [base.final_f05], s=80, color=ORANGE, zorder=3, label="no filter")
ax.annotate("chosen: p>=0.01, top-8\n5.5 cands/S1, F0.5 0.9606", (5.46, 0.9606), xytext=(12, -40),
            textcoords="offset points", fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate(f"{base.cands_per_s1:.1f} cands/S1, F0.5 {base.final_f05:.4f}", (base.cands_per_s1, base.final_f05),
            xytext=(-150, -30), textcoords="offset points", fontsize=9, color=INK)
ax.set(title="Candidate-set size vs final score (stage-1 cascade)", xlabel="candidates per S1 entity",
       ylabel="macro F0.5", ylim=(0.954, 0.962))
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "5_cascade_tradeoff.png", dpi=130)

# 6 unseen-country simulation
u = ["India seen in training", "India unseen (US only)", "+ thresholds re-tuned on India", "+ self-training"]
uv = [0.948, 0.878, 0.885, 0.879]
fig, ax = plt.subplots(figsize=(8, 3.8))
hbar(ax, u, uv, "{:.3f}", color=ORANGE, xlim=(0.8, 1.0))
ax.set(title="France proxy: performance on a country absent from training", xlabel="India macro F0.5")
fig.tight_layout(); fig.savefig(OUT + "6_unseen_country.png", dpi=130)
print("ok")
