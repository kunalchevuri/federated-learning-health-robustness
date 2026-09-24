r"""
figures_poster.py — figures 2-6 for the UNT Research Day poster.

Styling follows ChenLiu-1996/figures4papers, both its `scientific-figure-making`
reference and the patterns its own `figure_*` scripts use:

  * Arial/Helvetica stack, top and right spines off, heavy `axes.linewidth`,
    no grid, `svg.fonttype = "none"`, `dpi = 300` (600 for the dense panel).
  * Vertical bars with the value printed on top, y-limits tightened to the
    range that carries the comparison, `FixedLocator` for tick control.
  * Where seven categories will not fit as tick labels (figure 6), the ticks
    come off and a dedicated legend carries the names -- the same move as
    figure_ImmunoStruct/plot_bars.py.

Every figure carries a legend, an x-axis label, a y-axis label and a value
label on each bar. Descriptive captions live in build_poster.py, next to the
figure numbers, so the caption text sits in the poster's own type.

Series colours are unchanged, as asked; the figures4papers blue/green/red
semantic palette is therefore not applied. Every figure is drawn on #E2F0D9,
the COI template's own slide fill, so the panels sit on the board rather than
floating as white rectangles.

Figure 1, the schematic, is authored as HTML + SVG in diagram_federated.html
and rendered by render_diagram.py.

Run from the repo root:  python poster/figures_poster.py
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

OUT = "poster/figures"
os.makedirs(OUT, exist_ok=True)

COL = 7.26          # width of one poster column, in inches
PAPER = "#E2F0D9"   # the COI template's slide fill
INK = "#1A1A1A"

df = pd.read_csv("results/experiment_results_merged.csv")
final = df[df["round"] == 50]
main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]

# worst to best by mean AUC on BRFSS; every comparison figure uses this order
ORDER = ["fedavg", "fedprox", "krum", "uniform_mean", "trimmed_mean", "csagg", "coord_median"]
SHORT = ["FedAvg", "FedProx", "Krum", "Unweighted\nmean", "Trimmed\nmean", "CS-Agg", "Coord.\nmedian"]
FULL = ["FedAvg", "FedProx", "Krum", "Unweighted mean",
        "Trimmed mean", "CS-Agg", "Coord. median"]
COLOR = {"fedavg": "#e15759", "uniform_mean": "#6b6b6b", "fedprox": "#f28e2b", "csagg": "#4e79a7",
         "krum": "#9c755f", "trimmed_mean": "#af7aa1", "coord_median": "#59a14f"}

# FedAvg and FedProx are the only rules that weight a site by how much data it
# holds. Hatching them turns the poster's central distinction into a legend.
BY_SIZE = {"fedavg", "fedprox"}
HATCH = "///"

UNT_GREEN = "#007439"      # taken from the COI template itself
UNT_LIGHT = "#8CBF9B"      # a tint of it, for the second series in figure 2

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 18,
    "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 12.5, "ytick.labelsize": 16, "legend.fontsize": 15,
    "axes.linewidth": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "lines.linewidth": 3.0, "lines.markersize": 10,
    "xtick.major.width": 2.5, "ytick.major.width": 2.5,
    "xtick.major.size": 0, "ytick.major.size": 7,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "hatch.linewidth": 2.0,
    "svg.fonttype": "none",
})

BAR_EDGE = dict(edgecolor=INK, linewidth=1.5)


def save(name, dpi=300):
    plt.savefig(f"{OUT}/{name}", dpi=dpi, facecolor=PAPER)
    plt.close()
    print("  saved", name)


def method_bars(ax, values, width=0.72):
    """Vertical bars in the shared method order, hatched by weighting scheme."""
    bars = ax.bar(np.arange(len(ORDER)), [values[m] for m in ORDER], width,
                  color=[COLOR[m] for m in ORDER], zorder=3, **BAR_EDGE)
    for b, m in zip(bars, ORDER):
        if m in BY_SIZE:
            b.set_hatch(HATCH)
        if m == "uniform_mean":
            b.set_linewidth(3.5)
    return bars


def value_labels(ax, bars, fmt="{:.3f}", size=14, pad=0.004, **kw):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=size, fontweight="bold", **kw)


def weighting_legend(ax, **kw):
    """A legend that carries the poster's distinction, not just the names."""
    handles = [
        Patch(facecolor="white", edgecolor=INK, hatch=HATCH, linewidth=1.5,
              label="Weights a site by its data size"),
        Patch(facecolor="white", edgecolor=INK, linewidth=1.5,
              label="Ignores data size"),
        Patch(facecolor="white", edgecolor=INK, linewidth=3.5,
              label="Unweighted-mean control"),
    ]
    ax.legend(handles=handles, loc="upper left", handlelength=1.6,
              handleheight=1.2, labelspacing=0.35, borderpad=0.2, **kw)


def method_ticks(ax):
    ax.set_xticks(np.arange(len(ORDER)))
    ax.set_xticklabels(SHORT, linespacing=1.15)
    ax.set_xlim(-0.7, len(ORDER) - 0.3)
    ax.get_xticklabels()[ORDER.index("uniform_mean")].set_fontweight("bold")


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 | Where does the gain over FedAvg come from?
# ─────────────────────────────────────────────────────────────────────────
print("figure 3: decomposition at alpha=0.1 ...")
a01 = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]
mu = a01.groupby("method")["auc"].mean()
uc = mu["uniform_mean"]

fig, ax = plt.subplots(figsize=(COL, 5.3), layout="constrained")
bars = method_bars(ax, mu)
value_labels(ax, bars)
ax.axhline(uc, color=INK, linestyle="--", linewidth=2.6, zorder=4)
ax.text(6.55, uc - 0.007, "control", ha="right", va="top", fontsize=14, style="italic")

method_ticks(ax)
ax.set_xlabel("Merge rule", labelpad=8)
ax.set_ylabel("Mean AUC-ROC at alpha = 0.1\n(higher is better)", linespacing=1.25, fontsize=17)
ax.set_ylim(0.50, 0.90)
ax.yaxis.set_major_locator(FixedLocator([0.5, 0.6, 0.7, 0.8]))
weighting_legend(ax)
save("posterA_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 6 | Both datasets. Seven bars will not take tick labels in a half
# column, so the ticks come off and a dedicated legend names them.
# ─────────────────────────────────────────────────────────────────────────
print("figure 6: seven-strategy comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(COL, 5.1), layout="constrained")
for ax, ds, title, ylim, ticks in zip(
        axes, ["brfss", "breast_cancer"],
        ["BRFSS survey data", "Breast Cancer"],
        [(0.50, 0.90), (0.86, 1.07)],
        [[0.5, 0.6, 0.7, 0.8], [0.9, 1.0]]):
    sub = main[main["dataset"] == ds]
    m = sub.groupby("method")["auc"].mean()
    e = sub.groupby("method")["auc"].sem().reindex(ORDER)
    bars = method_bars(ax, m, width=0.76)
    ax.errorbar(np.arange(len(ORDER)), [m[k] for k in ORDER], yerr=e, fmt="none",
                ecolor=INK, elinewidth=2.0, capsize=4, zorder=4)
    value_labels(ax, bars, size=11.5, pad=(ylim[1] - ylim[0]) * 0.075, rotation=90)
    ax.set_title(title, fontsize=17, pad=8)
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(ORDER) - 0.3)
    ax.set_xlabel("Merge rule", labelpad=6, fontsize=16)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
axes[0].set_ylabel("Mean AUC-ROC", fontsize=17)

fig.legend(handles=[Patch(facecolor=COLOR[k], edgecolor=INK, linewidth=1.5,
                          hatch=HATCH if k in BY_SIZE else None, label=lab)
                    for k, lab in zip(ORDER, FULL)],
           loc="outside lower center", ncol=3, fontsize=12.5,
           handlelength=1.3, handleheight=1.0, labelspacing=0.25, columnspacing=1.0)
save("posterB_both_datasets.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 | Krum's instability: the mean hides it, the runs do not.
# ─────────────────────────────────────────────────────────────────────────
print("figure 4: run-level spread at alpha=0.1 ...")
fig, ax = plt.subplots(figsize=(COL, 5.4), layout="constrained")
bars = ax.bar(np.arange(len(ORDER)) - 0.19, [mu[m] for m in ORDER], 0.36,
              color=[COLOR[m] for m in ORDER], alpha=0.5, zorder=3, **BAR_EDGE)
for b, m in zip(bars, ORDER):
    if m in BY_SIZE:
        b.set_hatch(HATCH)
    if m == "uniform_mean":
        b.set_linewidth(3.5)
value_labels(ax, bars, size=12.5, pad=0.012)

rng = np.random.default_rng(0)
for i, m in enumerate(ORDER):
    runs = a01[a01["method"] == m]["auc"].values
    ax.scatter(i + 0.19 + rng.uniform(-0.11, 0.11, len(runs)), runs, color=COLOR[m],
               s=80, alpha=0.95, zorder=5, edgecolors="white", linewidth=1.3)

ax.axhline(0.5, color="#555555", linestyle=":", linewidth=2.6, zorder=2)
ax.text(6.6, 0.507, "random guessing", ha="right", va="bottom",
        fontsize=13.5, color="#555555")
ax.annotate("Krum's worst run: 0.328,\nworse than a coin flip",
            xy=(2.26, 0.3275), xytext=(2.7, 0.333), va="bottom", ha="left",
            fontsize=13.5, fontweight="bold", linespacing=1.25,
            arrowprops=dict(arrowstyle="->", lw=2.2, color=INK))

method_ticks(ax)
ax.set_xlabel("Merge rule", labelpad=8)
ax.set_ylabel("AUC-ROC at alpha = 0.1", fontsize=17)
ax.set_ylim(0.29, 0.84)
ax.yaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
fig.legend(handles=[
    Patch(facecolor="#9a9a9a", edgecolor=INK, linewidth=1.5, alpha=0.5,
          label="Mean of 12 runs"),
    Line2D([], [], marker="o", linestyle="none", color="#9a9a9a",
           markeredgecolor="white", markeredgewidth=1.3, markersize=10,
           label="One individual run"),
    Patch(facecolor="white", edgecolor=INK, hatch=HATCH, linewidth=1.5,
          label="Weights a site by its data size"),
], loc="outside lower center", ncol=2, fontsize=13.5, handlelength=1.5,
   handleheight=1.1, labelspacing=0.3, columnspacing=1.4)
save("posterC_runlevel.png")


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 | The partition itself.
# Client sizes depend only on the per-class counts, the seed and alpha, so
# src/data.py's partition_data is reproduced here exactly without touching the
# 1.2 GB XPT. Verified against the real pipeline: at alpha=0.1, seed 42, the
# largest client holds 106,481 rows (34.8% of train) and two clients get none.
# ─────────────────────────────────────────────────────────────────────────
print("figure 2: Dirichlet partition ...")
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

fig, axes = plt.subplots(2, 1, figsize=(COL, 5.0), sharex=True, sharey=True,
                         layout="constrained")
for ax, s, colour in zip(axes, [s01, s10], [UNT_GREEN, UNT_LIGHT]):
    bars = ax.bar(np.arange(20), s / 1000, 0.74, color=colour, zorder=3, **BAR_EDGE)
    for b, v in zip(bars, s / 1000):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                f"{v:.0f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 136)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_locator(FixedLocator([0, 50, 100]))

axes[0].annotate("one site holds 106,481 rows:\n34.8% of all training data",
                 xy=(0.45, 108), xytext=(3.4, 88), fontsize=13.5, fontweight="bold",
                 linespacing=1.25, arrowprops=dict(arrowstyle="->", lw=2.2, color=INK))
axes[0].annotate("two sites get\nnothing at all", xy=(19, 5), xytext=(14.2, 46),
                 fontsize=13, linespacing=1.25,
                 arrowprops=dict(arrowstyle="->", lw=2.0, color=INK))
axes[1].set_xlabel("The 20 sites, largest to smallest", fontsize=17, labelpad=6)
axes[1].xaxis.set_major_locator(FixedLocator([0, 4, 9, 14, 19]))
axes[1].set_xticklabels(["1", "5", "10", "15", "20"])
fig.supylabel("Training rows held (thousands)", fontsize=17)
fig.legend(handles=[
    Patch(facecolor=UNT_GREEN, edgecolor=INK, linewidth=1.5,
          label="alpha = 0.1  (the harshest split we test)"),
    Patch(facecolor=UNT_LIGHT, edgecolor=INK, linewidth=1.5,
          label="alpha = 1.0  (mild split, for comparison)"),
], loc="outside upper center", ncol=1, fontsize=14.5,
   handlelength=1.5, handleheight=1.1, labelspacing=0.25)
save("posterF_partition.png", dpi=600)


# ─────────────────────────────────────────────────────────────────────────
# Figure 5 | Heterogeneity vs label noise, on FedAvg.
# Exactly the contrast the paper reports: alpha 1.0->0.1 costs 0.2348 AUC,
# while 0%->30% label noise at alpha=0.5 costs 0.0016. Ratio 149x.
# ─────────────────────────────────────────────────────────────────────────
print("figure 5: heterogeneity vs label noise ...")
fa = main[(main["dataset"] == "brfss") & (main["method"] == "fedavg")]
het = fa[fa["alpha"] == 1.0]["auc"].mean() - fa[fa["alpha"] == 0.1]["auc"].mean()
noi = (fa[(fa["alpha"] == 0.5) & (fa["noise_rate"] == 0.0)]["auc"].mean()
       - fa[(fa["alpha"] == 0.5) & (fa["noise_rate"] == 0.3)]["auc"].mean())
assert abs(het - 0.2348) < 5e-4 and abs(noi - 0.0016) < 5e-4, "drifted from the paper"

fig, ax = plt.subplots(figsize=(COL, 4.3), layout="constrained")
bars = ax.bar([0, 1], [het, noi], 0.5, color=[COLOR["fedavg"], "#9aa0a6"],
              zorder=3, **BAR_EDGE)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.007,
            f"{b.get_height():.4f}", ha="center", va="bottom",
            fontsize=17, fontweight="bold")

ax.annotate("", xy=(0.32, het), xytext=(0.32, noi),
            arrowprops=dict(arrowstyle="<->", lw=2.4, color=INK))
ax.text(0.38, het / 2, "149x larger", fontsize=17, fontweight="bold",
        ha="left", va="center")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Lopsided data", "Broken labels"], fontsize=16)
ax.set_xlim(-0.55, 1.55)
ax.set_xlabel("Stressor applied to the same model", labelpad=8, fontsize=17)
ax.set_ylabel("AUC-ROC lost by FedAvg", fontsize=17)
ax.set_ylim(0, 0.31)
ax.yaxis.set_major_locator(FixedLocator([0, 0.1, 0.2, 0.3]))
ax.legend(handles=[
    Patch(facecolor=COLOR["fedavg"], edgecolor=INK, linewidth=1.5,
          label="Split skew, alpha 1.0 to 0.1"),
    Patch(facecolor="#9aa0a6", edgecolor=INK, linewidth=1.5,
          label="Label noise, 0% to 30%"),
], loc="upper right", handlelength=1.5, handleheight=1.1, labelspacing=0.3,
   borderpad=0.2)
save("posterG_stressors.png")


# ─────────────────────────────────────────────────────────────────────────
# QR to the public GitHub repo.
# ─────────────────────────────────────────────────────────────────────────
print("QR: repo link ...")
import qrcode

qr = qrcode.QRCode(box_size=20, border=1,
                   error_correction=qrcode.constants.ERROR_CORRECT_M)
qr.add_data("https://github.com/kunalchevuri/federated-learning-health-robustness")
qr.make(fit=True)
qr.make_image(fill_color=INK, back_color="white").save(f"{OUT}/posterE_qr.png")
print("  saved posterE_qr.png")

print("\nposter figures written to", OUT)
print("figure 1 is built separately:  python poster/render_diagram.py")
