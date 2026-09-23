r"""
figures_poster.py — figures for the UNT Research Day poster.

The manuscript figures (figures_v2.py) are drawn at 8 pt for a 3.5 in IEEE
column. Dropped onto a 24x36 in board they would be unreadable, so everything
here is redrawn at the poster's real column width with ~20 pt type, heavier
lines and larger markers.

Run from the repo root:  python poster/figures_poster.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "poster/figures"
os.makedirs(OUT, exist_ok=True)

COL = 7.26  # width of one poster column, in inches

df = pd.read_csv("results/experiment_results_merged.csv")
final = df[df["round"] == 50]
main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]

ORDER = ["fedavg", "fedprox", "krum", "uniform_mean", "trimmed_mean", "csagg", "coord_median"]
SHORT = ["FedAvg", "FedProx", "Krum", "Unweighted mean",
         "Trimmed mean", "CS-Agg", "Coord. median"]
COLOR = {"fedavg": "#e15759", "uniform_mean": "#6b6b6b", "fedprox": "#f28e2b", "csagg": "#4e79a7",
         "krum": "#9c755f", "trimmed_mean": "#af7aa1", "coord_median": "#59a14f"}

# UNT palette, taken from the COI template itself.
UNT_GREEN = "#007439"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20,
    "axes.titlesize": 22, "axes.labelsize": 21,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    "axes.linewidth": 1.6, "lines.linewidth": 3.2, "lines.markersize": 10,
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 1.1,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "#cccccc",
})


def save(name):
    plt.savefig(f"{OUT}/{name}", dpi=300)
    plt.close()
    print("  saved", name)


# ─────────────────────────────────────────────────────────────────────────
# A | Where does the gain over FedAvg come from?
# ─────────────────────────────────────────────────────────────────────────
print("A: decomposition at alpha=0.1 ...")
a01 = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]
mu = a01.groupby("method")["auc"].mean()
uc = mu["uniform_mean"]

fig, ax = plt.subplots(figsize=(COL, 8.2), layout="constrained")
x = np.arange(len(ORDER))
bars = ax.bar(x, [mu[m] for m in ORDER], 0.70,
              color=[COLOR[m] for m in ORDER], zorder=3)
bars[ORDER.index("uniform_mean")].set_edgecolor("black")
bars[ORDER.index("uniform_mean")].set_linewidth(3.5)

ax.axhline(uc, color="black", linestyle="--", linewidth=2.8, zorder=4)
for xi, m in zip(x, ORDER):
    ax.text(xi, mu[m] + 0.006, f"{mu[m]:.3f}", ha="center", va="bottom", fontsize=16)

ax.text(-0.45, 0.935,
        "Refusing to give the largest site the\n"
        "loudest vote captures 79% of the gain\n"
        "over FedAvg, with no robust statistic.",
        fontsize=17.5, fontweight="bold", va="top", ha="left", linespacing=1.35)
ax.text(-0.60, uc - 0.008, "unweighted-mean control (no robust statistic)",
        ha="left", va="top", fontsize=16, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(SHORT, fontsize=15.5, rotation=30, ha="right")
ax.set_ylabel("AUC-ROC  (higher is better)")
ax.set_ylim(0.50, 0.95)
ax.set_xlim(-0.72, 6.72)
ax.set_axisbelow(True)
save("posterA_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────
# B | Both datasets
# ─────────────────────────────────────────────────────────────────────────
print("B: seven-strategy comparison ...")
fig, axes = plt.subplots(2, 1, figsize=(COL, 8.0), sharex=True, layout="constrained")
for ax, ds, title in zip(axes, ["brfss", "breast_cancer"],
                         ["BRFSS survey data: the rules separate",
                          "Breast Cancer clinical benchmark: they do not"]):
    sub = main[main["dataset"] == ds]
    m = sub.groupby("method")["auc"].mean().reindex(ORDER)
    e = sub.groupby("method")["auc"].sem().reindex(ORDER)
    ax.bar(np.arange(len(ORDER)), m, 0.70, yerr=e, capsize=5,
           color=[COLOR[k] for k in ORDER], error_kw={"elinewidth": 2.0}, zorder=3)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(title, fontsize=18)
    ax.set_axisbelow(True)
    ax.set_ylim(0.50, 0.88) if ds == "brfss" else ax.set_ylim(0.88, 1.06)
axes[1].text(3.0, 1.032, "every rule but Krum sits at the ceiling",
             fontsize=16.5, style="italic", ha="center", va="center")
axes[1].set_xticks(np.arange(len(ORDER)))
axes[1].set_xticklabels(SHORT, fontsize=15.5, rotation=30, ha="right")
save("posterB_both_datasets.png")


# ─────────────────────────────────────────────────────────────────────────
# C | Krum's instability: every run at alpha=0.1
# ─────────────────────────────────────────────────────────────────────────
print("C: run-level spread at alpha=0.1 ...")
fig, ax = plt.subplots(figsize=(COL, 6.9), layout="constrained")
for i, m in enumerate(ORDER):
    ax.bar(i, mu[m], 0.62, color=COLOR[m], alpha=0.45, zorder=2)
rng = np.random.default_rng(0)
for _, r in a01.iterrows():
    i = ORDER.index(r["method"])
    ax.scatter(i + rng.uniform(-0.2, 0.2), r["auc"], color=COLOR[r["method"]],
               s=90, alpha=0.9, zorder=3, edgecolors="white", linewidth=1.4)
ax.axhline(0.5, color="#444444", linestyle="--", linewidth=2.4, zorder=1)
ax.text(6.5, 0.515, "random guessing", ha="right", va="bottom", fontsize=17, color="#444444")
ax.annotate("Krum's worst run:\n0.328, worse than\na coin flip",
            xy=(2, 0.3275), xytext=(2.75, 0.345), va="bottom",
            fontsize=16, fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=2.4, color="#1a1a1a"))
ax.set_xticks(np.arange(len(ORDER)))
ax.set_xticklabels(SHORT, fontsize=15.5, rotation=30, ha="right")
ax.set_ylabel("AUC-ROC (each of 12 runs)")
ax.set_ylim(0.29, 0.82)
ax.set_axisbelow(True)
save("posterC_runlevel.png")


# ─────────────────────────────────────────────────────────────────────────
# D | What federated learning is, as a picture.
# ─────────────────────────────────────────────────────────────────────────
print("D: federated learning schematic ...")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(COL, 3.1), layout="constrained")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.0)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, label, face, edge, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                facecolor=face, edgecolor=edge, linewidth=2.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", linespacing=1.2)


SITES = [(4.30, 1.55, "Site 1"), (2.57, 0.90, "Site 2"), (1.49, 0.64, "Site 3")]
for top, h, label in SITES:
    box(0.10, top - h, 2.45, h, label, "#dce7f1", "#4e79a7")

box(4.45, 1.95, 2.30, 1.20, "Merge rule", "#e8e8e8", "#6b6b6b")
box(7.80, 1.95, 2.10, 1.20, "One shared\nmodel", "#dcebd6", UNT_GREEN)

for top, h, _ in SITES:
    ax.add_patch(FancyArrowPatch((2.68, top - h / 2), (4.33, 2.55), arrowstyle="-|>",
                                 mutation_scale=20, linewidth=1.9, color="#4e79a7",
                                 connectionstyle="arc3,rad=0.05"))
ax.add_patch(FancyArrowPatch((6.88, 2.55), (7.68, 2.55), arrowstyle="-|>",
                             mutation_scale=20, linewidth=1.9, color=UNT_GREEN))

ax.text(3.60, 4.30, "model updates only,\nnever patient records", ha="center",
        va="bottom", fontsize=13, style="italic", color="#33536f", linespacing=1.2)
ax.text(1.33, 0.60, "sites hold very unequal\namounts of data", ha="center",
        va="top", fontsize=13, style="italic", color="#33536f", linespacing=1.2)
ax.text(5.60, 1.78, "how much should\neach site count?", ha="center", va="top",
        fontsize=13.5, style="italic", color="#1a1a1a", linespacing=1.2)
save("posterD_federated.png")


# ─────────────────────────────────────────────────────────────────────────
# F | The partition itself.
# Client sizes depend only on the per-class counts, the seed and alpha, so
# src/data.py's partition_data is reproduced here exactly without touching the
# 1.2 GB XPT. Verified against the real pipeline: at alpha=0.1, seed 42, the
# largest client holds 106,481 rows (34.8% of train) and two clients get none.
# ─────────────────────────────────────────────────────────────────────────
print("F: Dirichlet partition ...")
N_TRAIN, N_POS = 306167, 63491
N_NEG = N_TRAIN - N_POS


def client_sizes(alpha, seed=42, num_clients=20):
    np.random.seed(seed)
    sizes = np.zeros(num_clients, dtype=int)
    for n in (N_NEG, N_POS):                 # class 0, then class 1
        idx = np.arange(n)
        np.random.shuffle(idx)               # consumes RNG exactly as the real call
        p = np.random.dirichlet([alpha] * num_clients)
        p = (p * n).astype(int)
        p[-1] = n - p[:-1].sum()
        sizes += p
    return sizes


s01 = np.sort(client_sizes(0.1))[::-1]
s10 = np.sort(client_sizes(1.0))[::-1]
assert s01.max() == 106481 and (s01 == 0).sum() == 2, "partition drifted from the verified run"

fig, axes = plt.subplots(2, 1, figsize=(COL, 4.7), sharex=True, sharey=True,
                         layout="constrained")
for ax, s, lab in zip(axes, [s01, s10],
                      ["alpha = 0.1  (the harshest split we test)",
                       "alpha = 1.0  (mild split, for comparison)"]):
    ax.bar(np.arange(20), s / 1000, 0.75, color=UNT_GREEN, zorder=3)
    ax.set_title(lab, fontsize=17)
    ax.set_ylabel("training rows\n(thousands)", fontsize=15)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=15)

axes[0].annotate("one site holds 106,481 rows:\n34.8% of all training data",
                 xy=(0.42, 100), xytext=(3.1, 82), fontsize=15, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", lw=2.2, color="#1a1a1a"))
axes[0].annotate("two sites get\nnothing at all", xy=(19, 3), xytext=(13.8, 42),
                 fontsize=14, arrowprops=dict(arrowstyle="->", lw=2.0, color="#1a1a1a"))
axes[1].set_xlabel("the 20 sites, largest to smallest", fontsize=16)
axes[1].set_xticks([0, 4, 9, 14, 19])
axes[1].set_xticklabels(["1", "5", "10", "15", "20"])
save("posterF_partition.png")


# ─────────────────────────────────────────────────────────────────────────
# G | Heterogeneity vs label noise, on FedAvg.
# Exactly the contrast the paper reports: alpha 1.0->0.1 costs 0.2348 AUC,
# while 0%->30% label noise at alpha=0.5 costs 0.0016. Ratio 149x.
# ─────────────────────────────────────────────────────────────────────────
print("G: heterogeneity vs label noise ...")
fa = main[(main["dataset"] == "brfss") & (main["method"] == "fedavg")]
het = fa[fa["alpha"] == 1.0]["auc"].mean() - fa[fa["alpha"] == 0.1]["auc"].mean()
noi = (fa[(fa["alpha"] == 0.5) & (fa["noise_rate"] == 0.0)]["auc"].mean()
       - fa[(fa["alpha"] == 0.5) & (fa["noise_rate"] == 0.3)]["auc"].mean())
assert abs(het - 0.2348) < 5e-4 and abs(noi - 0.0016) < 5e-4, "drifted from the paper"

fig, ax = plt.subplots(figsize=(COL, 4.9), layout="constrained")
ax.bar([0, 1], [het, noi], 0.55, color=[COLOR["fedavg"], "#9aa0a6"], zorder=3)
for xi, v in [(0, het), (1, noi)]:
    ax.text(xi, v + 0.007, f"{v:.4f}", ha="center", va="bottom",
            fontsize=19, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Lopsided data\n(alpha 1.0 to 0.1)", "Broken labels\n(0% to 30%)"],
                   fontsize=17)
ax.set_ylabel("AUC lost by FedAvg", fontsize=18)
ax.set_ylim(0, 0.30)
ax.set_xlim(-0.6, 2.05)
ax.set_axisbelow(True)
ax.annotate("", xy=(1.38, 0.0016), xytext=(1.38, het),
            arrowprops=dict(arrowstyle="<->", lw=2.4, color="#1a1a1a"))
ax.text(1.48, het / 2, "149x\nlarger", fontsize=20, fontweight="bold",
        va="center", ha="left", linespacing=1.2)
save("posterG_stressors.png")


# ─────────────────────────────────────────────────────────────────────────
# E | QR to the public GitHub repo.
# ─────────────────────────────────────────────────────────────────────────
print("E: repo QR code ...")
import qrcode

qr = qrcode.QRCode(box_size=20, border=1,
                   error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data("https://github.com/kunalchevuri/federated-learning-health-robustness")
qr.make(fit=True)
qr.make_image(fill_color="#1a1a1a", back_color="white").save(f"{OUT}/posterE_qr.png")
print("  saved posterE_qr.png")

print("\nposter figures written to", OUT)
