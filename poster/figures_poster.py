r"""
figures_poster.py — figures for the UNT Research Day poster.

Styling follows ChenLiu-1996/figures4papers (its `scientific-figure-making`
reference): Arial/Helvetica stack, top and right spines removed, heavy
`axes.linewidth`, frameless legends, no grid, values annotated in place above
or beside the bars, y-limits tightened to the region that carries the
comparison, and `dpi=300` (600 for the dense partition panel).

Two deliberate departures from that reference, both requested:

  * Series colours are unchanged -- Dr. Aledhari asked to keep them as they
    are. The figures4papers blue/green/red semantic palette is therefore not
    applied.
  * Every figure is drawn on #E2F0D9, the COI template's own slide fill, so
    the panels sit on the board instead of floating as white rectangles.

Method comparisons are horizontal, which removes the rotated x-tick labels and
lets all three comparison figures share one method order (worst at the bottom,
best at the top) -- the multi-panel consistency that reference asks for.

The federated-learning schematic is no longer drawn here; it is authored as
HTML + SVG in diagram_federated.html and rendered by render_diagram.py.

Run from the repo root:  python poster/figures_poster.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

OUT = "poster/figures"
os.makedirs(OUT, exist_ok=True)

COL = 7.26     # width of one poster column, in inches
PAPER = "#E2F0D9"   # the COI template's slide fill
INK = "#1A1A1A"

df = pd.read_csv("results/experiment_results_merged.csv")
final = df[df["round"] == 50]
main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]

# worst to best by mean AUC on BRFSS; every comparison figure uses this order,
# drawn bottom-to-top so the best rule sits at the top of the panel.
ORDER = ["fedavg", "fedprox", "krum", "uniform_mean", "trimmed_mean", "csagg", "coord_median"]
SHORT = ["FedAvg", "FedProx", "Krum", "Unweighted mean",
         "Trimmed mean", "CS-Agg", "Coord. median"]
COLOR = {"fedavg": "#e15759", "uniform_mean": "#6b6b6b", "fedprox": "#f28e2b", "csagg": "#4e79a7",
         "krum": "#9c755f", "trimmed_mean": "#af7aa1", "coord_median": "#59a14f"}

UNT_GREEN = "#007439"   # taken from the COI template itself

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 20,
    "axes.titlesize": 20, "axes.labelsize": 20,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    "axes.linewidth": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "lines.linewidth": 3.0, "lines.markersize": 10,
    "xtick.major.width": 2.5, "ytick.major.width": 2.5,
    "xtick.major.size": 7, "ytick.major.size": 0,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "svg.fonttype": "none",
})

BAR_EDGE = dict(edgecolor=INK, linewidth=1.5)


def save(name, dpi=300):
    plt.savefig(f"{OUT}/{name}", dpi=dpi, facecolor=PAPER)
    plt.close()
    print("  saved", name)


def method_axis(ax):
    """Shared y-axis treatment for the three comparison figures."""
    ax.set_yticks(np.arange(len(ORDER)))
    ax.set_yticklabels(SHORT)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    # the control is the reference the whole poster turns on -- mark its label
    labels = ax.get_yticklabels()          # empty on the shared-y right panel
    if labels:
        labels[ORDER.index("uniform_mean")].set_fontweight("bold")


# ─────────────────────────────────────────────────────────────────────────
# A | Where does the gain over FedAvg come from?
# ─────────────────────────────────────────────────────────────────────────
print("A: decomposition at alpha=0.1 ...")
a01 = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]
mu = a01.groupby("method")["auc"].mean()
uc = mu["uniform_mean"]

fig, ax = plt.subplots(figsize=(COL, 4.8), layout="constrained")
y = np.arange(len(ORDER))
vals = [mu[m] for m in ORDER]
bars = ax.barh(y, vals, 0.68, color=[COLOR[m] for m in ORDER], zorder=3, **BAR_EDGE)
bars[ORDER.index("uniform_mean")].set_linewidth(3.5)

ax.axvline(uc, color=INK, linestyle="--", linewidth=2.6, zorder=4)
for yi, v in zip(y, vals):
    ax.text(v + 0.005, yi, f"{v:.3f}", va="center", ha="left",
            fontsize=18, fontweight="bold")

ax.text(uc, len(ORDER) - 0.25, "unweighted-mean control\n(no robust statistic)",
        ha="center", va="bottom", fontsize=16, style="italic", linespacing=1.25)
method_axis(ax)
ax.set_xlabel("mean AUC-ROC at alpha = 0.1")
ax.set_xlim(0.50, 0.845)
ax.xaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8]))
save("posterA_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────
# B | Both datasets, side by side.
# ─────────────────────────────────────────────────────────────────────────
print("B: seven-strategy comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(COL, 4.6), sharey=True, layout="constrained")
for ax, ds, title, xlim, ticks in zip(
        axes, ["brfss", "breast_cancer"],
        ["BRFSS survey data", "Breast Cancer"],
        [(0.50, 0.84), (0.86, 1.02)],
        [[0.5, 0.6, 0.7, 0.8], [0.9, 1.0]]):
    sub = main[main["dataset"] == ds]
    m = sub.groupby("method")["auc"].mean().reindex(ORDER)
    e = sub.groupby("method")["auc"].sem().reindex(ORDER)
    ax.barh(np.arange(len(ORDER)), m, 0.68, xerr=e, capsize=4,
            color=[COLOR[k] for k in ORDER], error_kw={"elinewidth": 2.0, "ecolor": INK},
            zorder=3, **BAR_EDGE)
    ax.set_title(title, fontsize=17, pad=10)
    ax.set_xlabel("AUC-ROC")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    method_axis(ax)

save("posterB_both_datasets.png")


# ─────────────────────────────────────────────────────────────────────────
# C | Krum's instability: every individual run at alpha=0.1.
# Points, not bars -- the spread is the whole message, and a bar behind it
# only restates the mean the marker already carries.
# ─────────────────────────────────────────────────────────────────────────
print("C: run-level spread at alpha=0.1 ...")
fig, ax = plt.subplots(figsize=(COL, 5.0), layout="constrained")
rng = np.random.default_rng(0)
for i, m in enumerate(ORDER):
    runs = a01[a01["method"] == m]["auc"].values
    ax.scatter(runs, i + rng.uniform(-0.22, 0.22, len(runs)), color=COLOR[m],
               s=110, alpha=0.85, zorder=3, edgecolors="white", linewidth=1.4)
    ax.plot([mu[m], mu[m]], [i - 0.36, i + 0.36], color=INK, linewidth=3.0, zorder=4)

ax.axvline(0.5, color="#555555", linestyle="--", linewidth=2.4, zorder=1)
ax.text(0.5, len(ORDER) - 0.25, "random guessing", ha="center", va="bottom",
        fontsize=16, color="#555555")
ax.annotate("Krum's worst run: 0.328,\nworse than a coin flip",
            xy=(0.3275, 2.05), xytext=(0.345, 3.30), va="center", ha="left",
            fontsize=16, fontweight="bold", linespacing=1.3,
            arrowprops=dict(arrowstyle="->", lw=2.4, color=INK))

method_axis(ax)
ax.set_xlabel("AUC-ROC of each run  (bar = mean)")
ax.set_xlim(0.29, 0.83)
ax.xaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
save("posterC_runlevel.png")


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

fig, axes = plt.subplots(2, 1, figsize=(COL, 5.6), sharex=True, sharey=True,
                         layout="constrained")
for ax, s, lab in zip(axes, [s01, s10],
                      ["alpha = 0.1  (the harshest split we test)",
                       "alpha = 1.0  (mild split, for comparison)"]):
    ax.bar(np.arange(20), s / 1000, 0.72, color=UNT_GREEN, zorder=3, **BAR_EDGE)
    ax.set_title(lab, fontsize=17)
    ax.set_ylabel("training rows\n(thousands)", fontsize=15, linespacing=1.2)
    ax.tick_params(labelsize=15)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))

axes[0].annotate("one site holds 106,481 rows:\n34.8% of all training data",
                 xy=(0.45, 100), xytext=(3.2, 78), fontsize=15, fontweight="bold",
                 linespacing=1.25,
                 arrowprops=dict(arrowstyle="->", lw=2.2, color=INK))
axes[0].annotate("two sites get\nnothing at all", xy=(19, 3), xytext=(13.6, 40),
                 fontsize=14, linespacing=1.25,
                 arrowprops=dict(arrowstyle="->", lw=2.0, color=INK))
axes[1].set_xlabel("the 20 sites, largest to smallest", fontsize=16)
axes[1].xaxis.set_major_locator(FixedLocator([0, 4, 9, 14, 19]))
axes[1].set_xticklabels(["1", "5", "10", "15", "20"])
save("posterF_partition.png", dpi=600)


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

fig, ax = plt.subplots(figsize=(COL, 3.0), layout="constrained")
ax.barh([0, 1], [noi, het], 0.52, color=["#9aa0a6", COLOR["fedavg"]], zorder=3, **BAR_EDGE)
ax.text(noi + 0.006, 0, f"{noi:.4f}", va="center", ha="left", fontsize=19, fontweight="bold")
ax.text(het + 0.006, 1, f"{het:.4f}", va="center", ha="left", fontsize=19, fontweight="bold")

ax.annotate("", xy=(noi, 0.5), xytext=(het, 0.5),
            arrowprops=dict(arrowstyle="<->", lw=2.4, color=INK))
ax.text((noi + het) / 2, 0.40, "149x larger", fontsize=19, fontweight="bold",
        ha="center", va="top")

ax.set_yticks([0, 1])
ax.set_yticklabels(["Broken labels\n(0% to 30%)", "Lopsided data\n(alpha 1.0 to 0.1)"],
                   fontsize=17, linespacing=1.25)
ax.spines["left"].set_visible(False)
ax.set_xlabel("AUC lost by FedAvg", fontsize=18)
ax.set_xlim(0, 0.30)
ax.set_ylim(-0.55, 1.55)
ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3]))
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
qr.make_image(fill_color=INK, back_color="white").save(f"{OUT}/posterE_qr.png")
print("  saved posterE_qr.png")

print("\nposter figures written to", OUT)
print("the schematic is built separately:  python poster/render_diagram.py")
