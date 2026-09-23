r"""
figures_v2.py — New figures for the TPS resubmission (7-strategy comparison,
noisy-fraction sweep, feature-set ablation). Run from fl-brfss/ directory
after analyze_v2.py has produced results/experiment_results_merged.csv.

Figures are generated at their final printed size so that LaTeX applies no
scaling: COL matches IEEEtran's \columnwidth and FULL matches \textwidth.
Font sizes set here are therefore the sizes that appear on the page. No
figure carries an embedded title or figure number; all explanation belongs
in the LaTeX caption.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/experiment_results_merged.csv")
final = df[df["round"] == 50].copy()

os.makedirs("results/figures", exist_ok=True)

# IEEEtran conference geometry, in inches.
COL = 3.5    # \columnwidth  -> use with \includegraphics[width=\columnwidth]
FULL = 7.16  # \textwidth    -> use inside figure* spanning both columns

METHOD_ORDER = ["fedavg", "uniform_mean", "fedprox", "csagg", "krum", "trimmed_mean", "coord_median"]
LABEL = {
    "fedavg": "FedAvg", "uniform_mean": "Uniform Mean", "fedprox": "FedProx", "csagg": "CS-Agg",
    "krum": "Krum", "trimmed_mean": "Trimmed Mean", "coord_median": "Coord. Median",
}
COLOR = {
    "fedavg": "#e15759", "uniform_mean": "#bab0ac", "fedprox": "#f28e2b", "csagg": "#4e79a7",
    "krum": "#9c755f", "trimmed_mean": "#af7aa1", "coord_median": "#59a14f",
}

# Single source of truth for figure style. Nothing below overrides these
# except the in-plot value annotations, which are deliberately one step
# smaller than the tick labels.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--", "grid.linewidth": 0.5,
    "legend.frameon": True, "legend.framealpha": 0.95, "legend.edgecolor": "lightgray",
    "legend.borderpad": 0.3, "legend.labelspacing": 0.3, "legend.handlelength": 1.6,
})

ANNOT = 6  # in-plot value labels

def save(name):
    # No bbox_inches="tight": that would trim the canvas and let LaTeX scale
    # the image back up, which is what made font sizes inconsistent before.
    plt.savefig(f"results/figures/{name}", dpi=400)
    plt.close()
    print(f"  Saved: results/figures/{name}")


# ─────────────────────────────────────────────────────────────────────────
# FIGURE 1 (fig8) | Seven-strategy AUC comparison, both datasets
# Two-column figure* in the manuscript.
# ─────────────────────────────────────────────────────────────────────────
print("Figure 1 (fig8): Seven-strategy comparison ...")

main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]

fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.7), layout="constrained")

for ax, dataset, panel_label in zip(
    axes, ["brfss", "breast_cancer"],
    ["BRFSS (Depression Detection)", "Breast Cancer (Benchmark)"]
):
    sub = main[main["dataset"] == dataset]
    means = sub.groupby("method")["auc"].mean().reindex(METHOD_ORDER)
    sems = sub.groupby("method")["auc"].sem().reindex(METHOD_ORDER)
    x = np.arange(len(METHOD_ORDER))
    colors = [COLOR[m] for m in METHOD_ORDER]
    ax.bar(x, means, 0.62, yerr=sems, capsize=2.5, color=colors,
           error_kw={"elinewidth": 0.9, "ecolor": "#333333"}, zorder=3)
    for xi, m in zip(x, METHOD_ORDER):
        ax.text(xi, means[m] + sems[m] + (0.006 if dataset == "brfss" else 0.0004),
                f"{means[m]:.3f}", ha="center", va="bottom", fontsize=ANNOT)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=30, ha="right")
    ax.set_ylabel("AUC-ROC (mean ± SE)")
    ax.set_title(panel_label)
    ax.set_axisbelow(True)
    if dataset == "breast_cancer":
        ax.set_ylim(0.85, 1.005)
    else:
        ax.set_ylim(0.5, 0.86)
        ax.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6, zorder=1)

save("fig8_six_method_comparison.png")


# ─────────────────────────────────────────────────────────────────────────
# FIGURE 2 (fig9) | Collapse at alpha=0.1, all seven strategies, seed-level dots
# Single-column figure.
# ─────────────────────────────────────────────────────────────────────────
print("Figure 2 (fig9): Seven-strategy collapse at alpha=0.1 ...")

collapse = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]

fig, ax = plt.subplots(figsize=(COL, 2.7), layout="constrained")
x_pos = {m: i for i, m in enumerate(METHOD_ORDER)}
seeds = sorted(collapse["seed"].unique())
seed_markers = {seeds[0]: "o", seeds[1]: "s", seeds[2]: "^"}

for method in METHOD_ORDER:
    mn = collapse[collapse["method"] == method]["auc"].mean()
    ax.bar(x_pos[method], mn, 0.55, color=COLOR[method], alpha=0.55, zorder=2)

rng = np.random.default_rng(0)
for _, row in collapse.iterrows():
    jitter = rng.uniform(-0.18, 0.18)
    ax.scatter(x_pos[row["method"]] + jitter, row["auc"], color=COLOR[row["method"]],
               marker=seed_markers[row["seed"]], s=11, alpha=0.85, zorder=3,
               edgecolors="white", linewidth=0.3)

ax.axhline(0.5, color="#555555", linewidth=1.0, linestyle="--", zorder=1)
ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], rotation=40, ha="right")
ax.set_ylabel("AUC-ROC (final round)")
ax.set_ylim(0.25, 0.85)
ax.set_axisbelow(True)
# Legend above the axes: Krum's worst run (0.328) sits in the lower right,
# which is the observation this figure exists to show.
seed_handles = [plt.Line2D([0], [0], marker=seed_markers[s], color="gray", linestyle="",
                           markersize=3.5, label=f"seed={s}") for s in seeds]
ax.legend(handles=seed_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=3, columnspacing=1.0, handletextpad=0.4)

save("fig9_six_method_collapse.png")


# ─────────────────────────────────────────────────────────────────────────
# FIGURE 4 (fig10) | Noisy-fraction sweep (0.2-0.6), BRFSS, alpha=0.1, noise=0.2
# Single-column figure.
# ─────────────────────────────────────────────────────────────────────────
print("Figure 4 (fig10): Noisy-fraction sweep ...")

nf = final[(final["dataset"] == "brfss") & (final["alpha"] == 0.1) &
           (final["noise_rate"] == 0.2) & (final["feature_set"] == "original")]

fig, ax = plt.subplots(figsize=(COL, 2.6), layout="constrained")
for method in METHOD_ORDER:
    grp = nf[nf["method"] == method].groupby("noisy_fraction")["auc"].mean()
    ax.plot(grp.index, grp.values, color=COLOR[method], label=LABEL[method], marker="o")

ax.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6)
ax.text(0.605, 0.503, "Random baseline", color="#555555", fontsize=ANNOT, va="bottom", ha="right")
ax.set_xlabel("Fraction of clients with corrupted labels")
ax.set_ylabel("AUC-ROC (final round)")
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim(0.47, 0.80)
ax.set_axisbelow(True)
# Legend below the axes: the FedAvg curve runs through the lower-left corner,
# so an in-axes legend hides it.
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28),
          columnspacing=1.0, handletextpad=0.5)

save("fig10_noisy_fraction_sweep.png")


# ─────────────────────────────────────────────────────────────────────────
# FIGURE 3 (fig11) | Feature-set ablation: 4-feature vs 12-feature
# Single-column figure, panels stacked vertically so each panel keeps a
# usable aspect ratio at 3.5 in.
# ─────────────────────────────────────────────────────────────────────────
print("Figure 3 (fig11): Feature-set ablation ...")

# The expanded-feature runs only cover noise rates 0.0 and 0.2, so restrict the
# original-feature panel to the same two rates. Otherwise the two panels average
# over different noise grids and are not directly comparable.
fa = final[(final["dataset"] == "brfss") & (final["method"].isin(["fedavg", "csagg"])) &
           (final["noisy_fraction"] == 0.2) & (final["noise_rate"].isin([0.0, 0.2]))]
fa_alpha = fa.groupby(["feature_set", "alpha", "method"])["auc"].mean().unstack("method")

fig, axes = plt.subplots(2, 1, figsize=(COL, 3.3), sharex=True, layout="constrained")

for ax, fs, title in zip(axes, ["original", "expanded"],
                         ["4-feature set (original)", "12-feature set (expanded)"]):
    sub = fa_alpha.loc[fs]
    for method in ["fedavg", "csagg"]:
        ax.plot(sub.index, sub[method], color=COLOR[method], label=LABEL[method], marker="o")
    ax.set_ylabel("AUC-ROC")
    ax.set_xticks([0.1, 0.5, 1.0])
    ax.set_ylim(0.5, 0.85)
    ax.set_title(title)
    ax.set_axisbelow(True)

axes[0].legend(loc="lower right", ncol=2, columnspacing=0.8)
axes[1].set_xlabel("Dirichlet alpha")

save("fig11_feature_ablation.png")

print("\nAll new figures saved to results/figures/")
