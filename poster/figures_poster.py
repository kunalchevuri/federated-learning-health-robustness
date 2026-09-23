r"""
figures_poster.py — figures re-rendered for the UNT Research Day poster.

The manuscript figures (figures_v2.py) are drawn at 8 pt for a 3.5 in IEEE
column. Dropped into the poster's 7.4 in middle column they would render at
roughly 1:1, leaving 8 pt axis labels on a board read from six feet away.
These are redrawn at the poster's real slot width with ~20 pt type, heavier
lines and larger markers. Run from the repo root:  python poster/figures_poster.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "poster/figures"
os.makedirs(OUT, exist_ok=True)

COL = 7.4  # width of the template's middle column, in inches

df = pd.read_csv("results/experiment_results_merged.csv")
final = df[df["round"] == 50]
main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]

ORDER = ["fedavg", "fedprox", "krum", "uniform_mean", "trimmed_mean", "csagg", "coord_median"]
LABEL = {"fedavg": "FedAvg", "uniform_mean": "Unweighted\nmean (control)", "fedprox": "FedProx",
         "csagg": "CS-Agg", "krum": "Krum", "trimmed_mean": "Trimmed\nmean",
         "coord_median": "Coord.\nmedian"}
SHORT = ["FedAvg", "FedProx", "Krum", "Unweighted mean",
         "Trimmed mean", "CS-Agg", "Coord. median"]
COLOR = {"fedavg": "#e15759", "uniform_mean": "#6b6b6b", "fedprox": "#f28e2b", "csagg": "#4e79a7",
         "krum": "#9c755f", "trimmed_mean": "#af7aa1", "coord_median": "#59a14f"}

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
# A | The headline argument: where does the advantage over FedAvg come from?
# ─────────────────────────────────────────────────────────────────────────
print("A: decomposition at alpha=0.1 ...")
a01 = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]
mu = a01.groupby("method")["auc"].mean()
fa, uc = mu["fedavg"], mu["uniform_mean"]

fig, ax = plt.subplots(figsize=(COL, 8.4), layout="constrained")
fig.suptitle("Where does the gain over FedAvg come from?",
             fontsize=19, fontweight="bold")
x = np.arange(len(ORDER))
bars = ax.bar(x, [mu[m] for m in ORDER], 0.70,
              color=[COLOR[m] for m in ORDER], zorder=3)
bars[ORDER.index("uniform_mean")].set_edgecolor("black")
bars[ORDER.index("uniform_mean")].set_linewidth(3.5)

ax.axhline(uc, color="black", linestyle="--", linewidth=2.8, zorder=4)
for xi, m in zip(x, ORDER):
    ax.text(xi, mu[m] + 0.006, f"{mu[m]:.3f}", ha="center", va="bottom", fontsize=16)

# The whole argument, stated once, in the empty space above the three low bars.
ax.text(-0.45, 0.935,
        "Refusing to give the largest site the\n"
        "loudest vote captures 79% of the gain\n"
        "over FedAvg, with no robust statistic.\n"
        "Only coord. median clearly beats it.",
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
# B | All seven strategies, both datasets
# ─────────────────────────────────────────────────────────────────────────
print("B: seven-strategy comparison ...")
fig, axes = plt.subplots(2, 1, figsize=(COL, 8.2), sharex=True, layout="constrained")
fig.suptitle("The clinical benchmark hides the difference",
             fontsize=19, fontweight="bold")
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
fig, ax = plt.subplots(figsize=(COL, 7.1), layout="constrained")
fig.suptitle("Krum is unreliable, not merely weaker",
             fontsize=19, fontweight="bold")
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

print("\nposter figures written to", OUT)


# ─────────────────────────────────────────────────────────────────────────
# D | What federated learning is, as a picture.
# This replaces the INTRODUCTION paragraph that described the mechanism, so a
# viewer with no background gets the setup without reading prose. The three
# site boxes are deliberately unequal in size: that inequality is the paper.
# ─────────────────────────────────────────────────────────────────────────
print("D: federated learning schematic ...")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.6, 2.9), layout="constrained")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.0)
ax.axis("off")
ax.grid(False)


def box(x, y, w, h, label, face, edge, fs=15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                facecolor=face, edgecolor=edge, linewidth=2.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", linespacing=1.2)


# Three sites, drawn to scale with how much data each one holds.
# top, height  -> Site 1 is the outsized client the paper is about
SITES = [(4.30, 1.55, "Site 1"), (2.57, 0.90, "Site 2"), (1.49, 0.64, "Site 3")]
for top, h, label in SITES:
    box(0.10, top - h, 2.45, h, label, "#dce7f1", "#4e79a7")

box(4.45, 1.95, 2.30, 1.20, "Merge rule", "#e8e8e8", "#6b6b6b")
box(7.80, 1.95, 2.10, 1.20, "One shared\nmodel", "#dcebd6", "#59a14f")

for top, h, _ in SITES:
    ax.add_patch(FancyArrowPatch((2.68, top - h / 2), (4.33, 2.55), arrowstyle="-|>",
                                 mutation_scale=20, linewidth=1.9, color="#4e79a7",
                                 connectionstyle="arc3,rad=0.05"))
ax.add_patch(FancyArrowPatch((6.88, 2.55), (7.68, 2.55), arrowstyle="-|>",
                             mutation_scale=20, linewidth=1.9, color="#59a14f"))

ax.text(3.60, 4.30, "model updates only,\nnever patient records", ha="center",
        va="bottom", fontsize=13, style="italic", color="#33536f", linespacing=1.2)
ax.text(1.33, 0.60, "sites hold very unequal\namounts of data", ha="center",
        va="top", fontsize=13, style="italic", color="#33536f", linespacing=1.2)
ax.text(5.60, 1.78, "how much should\neach site count?", ha="center", va="top",
        fontsize=13.5, style="italic", color="#1a1a1a", linespacing=1.2)
save("posterD_federated.png")


# ─────────────────────────────────────────────────────────────────────────
# E | QR to the OSF release. Posters get photographed; a link people have to
# retype by hand does not survive that.
# ─────────────────────────────────────────────────────────────────────────
print("E: OSF QR code ...")
import qrcode

qr = qrcode.QRCode(box_size=20, border=1,
                   error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data("https://osf.io/d5u2q")
qr.make(fit=True)
qr.make_image(fill_color="#1a1a1a", back_color="white").save(f"{OUT}/posterE_qr.png")
print("  saved posterE_qr.png")
