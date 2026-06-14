"""
figures.py  —  Publication-quality figures for FL-BRFSS paper
Run from the fl-brfss/ directory: python figures.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Data ──────────────────────────────────────────────────────────────────────
df    = pd.read_csv("results/experiment_results.csv")
final = df[df["round"] == 50].copy()
brfss_final       = final[final["dataset"] == "brfss"]
bc_final          = final[final["dataset"] == "breast_cancer"]
brfss_all_rounds  = df[df["dataset"] == "brfss"]

os.makedirs("results/figures", exist_ok=True)

# ── Shared style constants ─────────────────────────────────────────────────────
METHOD_ORDER  = ["fedavg", "csagg", "fedprox"]
LABEL         = {"fedavg": "FedAvg", "csagg": "CS-Agg", "fedprox": "FedProx"}
COLOR         = {"fedavg": "#e15759", "csagg": "#4e79a7", "fedprox": "#f28e2b"}
ALPHA_LABELS  = {0.1: "0.1 (high)", 0.5: "0.5 (mid)", 1.0: "1.0 (low)"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

def save(name):
    plt.savefig(f"results/figures/{name}", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved: results/figures/{name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  |  AUC Convergence Curves — BRFSS, faceted by noise rate
#   One line per method. Averaged over alpha x seed. No shading (too noisy).
#   Vertical dotted line = end of CS-Agg warmup (round 10).
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 1: Convergence curves ...")

noise_rates = [0.0, 0.1, 0.2, 0.3]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.06, top=0.82)

for ax, nr in zip(axes, noise_rates):
    sub = brfss_all_rounds[brfss_all_rounds["noise_rate"] == nr]
    for method in METHOD_ORDER:
        mn = sub[sub["method"] == method].groupby("round")["auc"].mean()
        ax.plot(mn.index, mn.values,
                color=COLOR[method], label=LABEL[method],
                linewidth=2.5)
    ax.axvline(10, color="gray", linewidth=1.2, linestyle=":", alpha=0.7)
    ax.set_title(f"Noise Rate = {nr:.1f}", fontweight="bold", pad=8)
    ax.set_xlabel("Round")
    ax.set_xlim(1, 50)
    ax.set_ylim(0.50, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

axes[0].set_ylabel("AUC-ROC")
axes[-1].legend(loc="lower right", framealpha=0.95, edgecolor="lightgray")

fig.suptitle(
    "Figure 1  |  AUC-ROC Convergence on BRFSS by Noise Rate\n"
    "(averaged over Dirichlet alpha and random seeds; dotted line = CS-Agg warmup end)",
    fontsize=12, fontweight="bold", y=1.0)

save("fig1_convergence.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  |  Final-Round AUC by Method x Noise Rate — both datasets
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: Bar chart — noise rate effect ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.38)

for ax, dataset, panel_label in zip(
    axes,
    ["brfss", "breast_cancer"],
    ["BRFSS — Depression Detection", "Breast Cancer (Benchmark)"]
):
    sub   = final[final["dataset"] == dataset]
    means = sub.groupby(["noise_rate", "method"])["auc"].mean().unstack("method")
    sems  = sub.groupby(["noise_rate", "method"])["auc"].sem().unstack("method")

    x      = np.arange(len(means))
    width  = 0.28
    for i, method in enumerate(METHOD_ORDER):
        offset = (i - 1) * width
        ax.bar(x + offset, means[method], width,
               yerr=sems[method], capsize=3,
               color=COLOR[method], label=LABEL[method],
               error_kw={"elinewidth": 1.3, "ecolor": "#444444"},
               zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{nr:.1f}" for nr in means.index])
    ax.set_xlabel("Noise Rate")
    ax.set_ylabel("AUC-ROC (mean ± SE)")
    ax.set_title(panel_label, fontweight="bold")
    if dataset == "breast_cancer":
        # Show tight range to make small differences visible
        ax.set_ylim(0.994, 1.001)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    else:
        ypad = 0.018
        ax.set_ylim(max(0.0, means.values.min() - ypad * 3),
                    min(1.0, means.values.max() + ypad * 3))
    ax.set_axisbelow(True)

# Shared legend above panels
handles = [mpatches.Patch(color=COLOR[m], label=LABEL[m]) for m in METHOD_ORDER]
fig.legend(handles=handles, loc="upper center", ncol=3,
           framealpha=0.95, edgecolor="lightgray",
           bbox_to_anchor=(0.5, 1.01))

save("fig2_bar_noise.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  |  Heterogeneity vs Label Noise — which drives performance?
#   Left:  AUC vs Alpha  (heterogeneity effect — large)
#   Right: AUC vs Noise Rate  (noise effect — small)
#   BRFSS only. Lines only, no shading.
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: Heterogeneity vs noise ...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.subplots_adjust(wspace=0.40)

# Left: AUC vs Alpha
ax = axes[0]
for method in METHOD_ORDER:
    grp = brfss_final[brfss_final["method"] == method].groupby("alpha")["auc"].mean()
    ax.plot(grp.index, grp.values,
            color=COLOR[method], label=LABEL[method],
            linewidth=2.5, marker="o", markersize=9)

ax.set_xlabel("Dirichlet Alpha  (smaller = more heterogeneous)")
ax.set_ylabel("AUC-ROC (mean over noise rates & seeds)")
ax.set_title("Effect of Data Heterogeneity (Alpha)", fontweight="bold")
ax.set_xticks([0.1, 0.5, 1.0])
ax.set_xlim(0.0, 1.15)
ax.set_ylim(0.48, 0.82)
ax.legend(framealpha=0.95, edgecolor="lightgray")
ax.annotate("", xy=(1.0, 0.788), xytext=(0.1, 0.553),
            arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.5))
ax.text(0.68, 0.655, "~0.22 AUC\ngap (FedAvg)", color="#888888",
        fontsize=9.5, ha="left", style="italic")

# Right: AUC vs Noise Rate
ax = axes[1]
for method in METHOD_ORDER:
    grp = brfss_final[brfss_final["method"] == method].groupby("noise_rate")["auc"].mean()
    ax.plot(grp.index, grp.values,
            color=COLOR[method], label=LABEL[method],
            linewidth=2.5, marker="s", markersize=9)

ax.set_xlabel("Label Noise Rate")
ax.set_ylabel("AUC-ROC (mean over alpha & seeds)")
ax.set_title("Effect of Label Noise Rate", fontweight="bold")
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.set_xlim(-0.025, 0.35)
ax.set_ylim(0.48, 0.82)
ax.legend(framealpha=0.95, edgecolor="lightgray")
ax.text(0.17, 0.507,
        "Max drop ~0.002 AUC\n(noise barely matters)",
        color="#888888", fontsize=9.5, ha="center", style="italic",
        va="bottom")

save("fig3_heterogeneity_vs_noise.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4  |  Heatmap — Alpha x Noise Rate per method on BRFSS
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: Heatmap ...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.subplots_adjust(wspace=0.20)

vmin = brfss_final["auc"].min()
vmax = brfss_final["auc"].max()

for ax, method in zip(axes, METHOD_ORDER):
    pivot = (brfss_final[brfss_final["method"] == method]
             .groupby(["alpha", "noise_rate"])["auc"]
             .mean()
             .unstack("noise_rate"))

    show_cbar = (method == "fedprox")
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".3f", annot_kws={"size": 11, "fontweight": "bold"},
        cmap="RdYlGn", vmin=vmin, vmax=vmax,
        cbar=show_cbar,
        cbar_kws={"shrink": 0.85, "pad": 0.02},
        linewidths=1.0, linecolor="white",
    )
    ax.set_title(LABEL[method], fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Noise Rate", labelpad=8)
    # y-axis label and ticks only on leftmost panel
    if method == "fedavg":
        ax.set_ylabel("Alpha (heterogeneity)", labelpad=8)
        ax.set_yticklabels(["0.1 (high)", "0.5 (mid)", "1.0 (low)"],
                           rotation=0, fontsize=10, va="center")
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
    ax.set_xticklabels(["0.0", "0.1", "0.2", "0.3"], rotation=0, fontsize=10)

if axes[-1].collections:
    cbar = axes[-1].collections[0].colorbar
    if cbar:
        cbar.set_label("AUC-ROC", fontsize=11)

save("fig4_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5  |  Model Collapse at alpha=0.1 — the key finding
#   Left:  Mean AUC bar + individual seed dots at alpha=0.1, all noise rates
#   Right: AUC vs Round convergence at alpha=0.1, noise=0.0, per seed
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 5: Collapse visualization ...")

collapse_final = brfss_final[brfss_final["alpha"] == 0.1].copy()
collapse_conv  = brfss_all_rounds[
    (brfss_all_rounds["alpha"] == 0.1) &
    (brfss_all_rounds["noise_rate"] == 0.0)
].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0.35, top=0.80)

# ── Left: bar + strip at alpha=0.1 ──
ax = axes[0]
x_pos  = {m: i for i, m in enumerate(METHOD_ORDER)}
seeds  = sorted(collapse_final["seed"].unique())
seed_markers = {seeds[0]: "o", seeds[1]: "s", seeds[2]: "^"}

# Bars (mean)
for method in METHOD_ORDER:
    mn = collapse_final[collapse_final["method"] == method]["auc"].mean()
    ax.bar(x_pos[method], mn, 0.5,
           color=COLOR[method], alpha=0.55, zorder=2)

# Individual dots (each condition = one seed x one noise_rate)
rng = np.random.default_rng(0)
for _, row in collapse_final.iterrows():
    jitter = rng.uniform(-0.18, 0.18)
    ax.scatter(
        x_pos[row["method"]] + jitter, row["auc"],
        color=COLOR[row["method"]],
        marker=seed_markers[row["seed"]],
        s=55, alpha=0.85, zorder=3,
        edgecolors="white", linewidth=0.6,
    )

ax.axhline(0.5, color="#555555", linewidth=1.5, linestyle="--", zorder=1)
ax.text(2.55, 0.503, "Random baseline", color="#555555", fontsize=10, va="bottom")

ax.set_xticks(list(x_pos.values()))
ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], fontsize=12)
ax.set_ylabel("AUC-ROC (final round, round 50)")
ax.set_ylim(0.35, 0.84)
ax.set_title("AUC Distribution at alpha=0.1 (High Heterogeneity)\nAll noise rates x 3 seeds shown",
             fontweight="bold")

# Seed legend
seed_handles = [
    plt.Line2D([0], [0], marker=seed_markers[s], color="gray",
               linestyle="", markersize=8, label=f"seed={s}")
    for s in seeds
]
seed_handles.append(
    mpatches.Patch(color="gray", alpha=0.5, label="bar = mean")
)
ax.legend(handles=seed_handles, fontsize=10, framealpha=0.95,
          edgecolor="lightgray", loc="upper right")

# Annotate means
for method in METHOD_ORDER:
    mn = collapse_final[collapse_final["method"] == method]["auc"].mean()
    ax.text(x_pos[method], mn + 0.02, f"{mn:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=COLOR[method])

# ── Right: convergence per seed at alpha=0.1, noise=0.0 ──
ax = axes[1]
linestyles = {seeds[0]: "solid", seeds[1]: "dashed", seeds[2]: "dotted"}

for method in METHOD_ORDER:
    for seed in seeds:
        sub = collapse_conv[
            (collapse_conv["method"] == method) &
            (collapse_conv["seed"] == seed)
        ]
        ax.plot(sub["round"], sub["auc"],
                color=COLOR[method],
                linestyle=linestyles[seed],
                linewidth=2.0, alpha=0.85)

ax.axhline(0.5, color="#555555", linewidth=1.5, linestyle="--", alpha=0.7)
ax.text(42, 0.503, "Random", color="#555555", fontsize=10, va="bottom")

ax.set_xlabel("Round")
ax.set_ylabel("AUC-ROC")
ax.set_xlim(1, 50)
ax.set_ylim(0.35, 0.84)
ax.set_title("Convergence at alpha=0.1, Noise=0.0\n(one line per method x seed combination)",
             fontweight="bold")

# Combined legend: method color + seed linestyle
method_handles = [
    plt.Line2D([0], [0], color=COLOR[m], linewidth=2.5, label=LABEL[m])
    for m in METHOD_ORDER
]
seed_handles2 = [
    plt.Line2D([0], [0], color="gray", linestyle=linestyles[s],
               linewidth=2.0, label=f"seed={s}")
    for s in seeds
]
ax.legend(handles=method_handles + seed_handles2,
          fontsize=10, framealpha=0.95, edgecolor="lightgray",
          ncol=2, loc="upper left")

fig.suptitle(
    "Figure 5  |  Model Collapse at High Heterogeneity (alpha=0.1) — BRFSS\n"
    "FedAvg collapses below random (AUC < 0.5) for 2 of 3 seeds; CS-Agg remains stable across all seeds.",
    fontsize=12, fontweight="bold", y=1.0)

save("fig5_collapse.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6  |  AUC by Method, split by Alpha — BRFSS
#   Grouped bar chart. Shows where the differences actually come from.
#   Replaces misleading boxplot where collapse cases were invisible outliers.
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 6: AUC by alpha group (BRFSS) ...")

fig, ax = plt.subplots(figsize=(7, 6))

alphas = [0.1, 0.5, 1.0]
alpha_labels = ["alpha=0.1\n(high heterogeneity)", "alpha=0.5\n(moderate)", "alpha=1.0\n(near-IID)"]
n_alphas  = len(alphas)
n_methods = len(METHOD_ORDER)
group_width = 0.7
bar_width   = group_width / n_methods
group_centers = np.arange(n_alphas)

for i, method in enumerate(METHOD_ORDER):
    offset = (i - 1) * bar_width
    for j, alpha in enumerate(alphas):
        sub = brfss_final[
            (brfss_final["method"] == method) &
            (brfss_final["alpha"] == alpha)
        ]["auc"]
        mn  = sub.mean()
        sem = sub.sem()
        xpos = group_centers[j] + offset
        ax.bar(xpos, mn, bar_width * 0.92,
               color=COLOR[method],
               label=LABEL[method] if j == 0 else "",
               yerr=sem, capsize=4,
               error_kw={"elinewidth": 1.4, "ecolor": "#333333"},
               zorder=3)
        ax.text(xpos, mn + sem + 0.004, f"{mn:.3f}",
                ha="center", va="bottom", fontsize=8.5,
                color=COLOR[method], fontweight="bold")

ax.axhline(0.5, color="#555555", linewidth=1.5, linestyle="--", alpha=0.7, zorder=1)
ax.text(n_alphas - 0.05, 0.503, "Random baseline", color="#555555",
        fontsize=10, va="bottom", ha="right")

ax.set_xticks(group_centers)
ax.set_xticklabels(alpha_labels, fontsize=12)
ax.set_ylabel("AUC-ROC (mean +/- SE)")
ax.set_ylim(0.45, 0.84)
ax.set_axisbelow(True)
ax.legend(framealpha=0.95, edgecolor="lightgray", fontsize=11)

save("fig6_auc_by_alpha.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7  |  Violin plots with Wilcoxon brackets — both datasets
#   Shows full AUC distribution per method including bimodal FedAvg on BRFSS.
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 7: Violin plots with significance ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.32, top=0.80)

for ax, dataset, ds_label in zip(
    axes,
    ["brfss", "breast_cancer"],
    ["BRFSS — Depression Detection", "Breast Cancer (Benchmark)"]
):
    sub  = final[final["dataset"] == dataset]
    data = [sub[sub["method"] == m]["auc"].values for m in METHOD_ORDER]

    parts = ax.violinplot(data, positions=range(len(METHOD_ORDER)),
                          showmedians=True, showextrema=True)

    for i, (pc, method) in enumerate(zip(parts["bodies"], METHOD_ORDER)):
        pc.set_facecolor(COLOR[method])
        pc.set_alpha(0.65)
        pc.set_edgecolor("white")

    for part_name in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if part_name in parts:
            parts[part_name].set_color("#222222")
            parts[part_name].set_linewidth(1.8)

    # Overlay individual points
    for i, (d, method) in enumerate(zip(data, METHOD_ORDER)):
        rng2 = np.random.default_rng(i)
        jitter = rng2.uniform(-0.06, 0.06, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d,
                   color=COLOR[method], s=18, alpha=0.4, zorder=3)

    # Wilcoxon brackets (BRFSS only — breast cancer differences too small to display)
    if dataset == "brfss":
        vals = {m: sub[sub["method"] == m]["auc"].values for m in METHOD_ORDER}
        pairs = [
            (0, 1, "fedavg", "csagg"),
            (0, 2, "fedavg", "fedprox"),
        ]
        y_tops = [max(d.max() for d in data) + 0.02,
                  max(d.max() for d in data) + 0.055]
        for (xi, xj, m1, m2), yt in zip(pairs, y_tops):
            _, p = stats.wilcoxon(vals[m1], vals[m2])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.plot([xi, xi, xj, xj],
                    [yt - 0.004, yt, yt, yt - 0.004],
                    color="black", linewidth=1.3)
            ax.text((xi + xj) / 2, yt + 0.003, sig,
                    ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels([LABEL[m] for m in METHOD_ORDER], fontsize=12)
    ax.set_ylabel("AUC-ROC (final round)")
    ax.set_title(ds_label, fontweight="bold")
    ax.set_axisbelow(True)

fig.suptitle(
    "Figure 7  |  AUC Distribution per Method — All Conditions (n=36 each)\n"
    "Wilcoxon signed-rank test: *** p < 0.001. "
    "BRFSS: FedAvg shows bimodal distribution (collapse at alpha=0.1).",
    fontsize=12, fontweight="bold", y=1.0)

save("fig7_violin.png")


print("\nAll 7 figures saved to results/figures/")
print("Files:")
for i in range(1, 8):
    fnames = {
        1: "fig1_convergence.png",
        2: "fig2_bar_noise.png",
        3: "fig3_heterogeneity_vs_noise.png",
        4: "fig4_heatmap.png",
        5: "fig5_collapse.png",
        6: "fig6_auc_by_alpha.png",
        7: "fig7_violin.png",
    }
    print(f"  {fnames[i]}")
