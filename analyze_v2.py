import glob
import pandas as pd
import numpy as np
from scipy import stats

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

# ── Merge original + all v2 shards into one frame ──────────────────
orig = pd.read_csv("results/experiment_results.csv")
orig["feature_set"] = "original"

shards = [pd.read_csv(f) for f in glob.glob("results/v2_shards/*.csv")]
v2 = pd.concat(shards, ignore_index=True)

df = pd.concat([orig, v2], ignore_index=True)
df.to_csv("results/experiment_results_merged.csv", index=False)
print(f"Merged: {len(df)} rows ({len(orig)} original + {len(v2)} new)")

final = df[df["round"] == 50].copy()
key_cols = ["method", "dataset", "feature_set", "alpha", "noise_rate", "noisy_fraction", "seed"]
dupes = final[final.duplicated(subset=key_cols, keep=False)]
if len(dupes):
    print(f"\nWARNING: {len(dupes)} duplicate condition rows found:")
    print(dupes[key_cols].drop_duplicates())

METHOD_ORDER = ["fedavg", "uniform_mean", "fedprox", "csagg", "krum", "trimmed_mean", "coord_median"]

# ── 1. Full 6-method comparison, matching the original 216-condition grid ──
print("\n" + "=" * 70)
print("1. FULL 6-METHOD COMPARISON (original grid, feature_set=original, noisy_fraction=0.2)")
print("=" * 70)
main = final[(final["feature_set"] == "original") & (final["noisy_fraction"] == 0.2)]
summary = main.groupby(["dataset", "method"])["auc"].agg(["mean", "std", "count"]).round(4)
summary = summary.reindex(METHOD_ORDER, level="method")
print(summary)

print("\nBy alpha (BRFSS, averaged over noise rates & seeds):")
brfss = main[main["dataset"] == "brfss"]
by_alpha = brfss.groupby(["alpha", "method"])["auc"].mean().round(4).unstack("method")[METHOD_ORDER]
print(by_alpha)

# ── 2. Wilcoxon signed-rank: CS-Agg vs every other method, per dataset ──
print("\n" + "=" * 70)
print("2. WILCOXON SIGNED-RANK: CS-Agg vs each baseline (n=36 pairs per dataset)")
print("=" * 70)
for dataset in ["brfss", "breast_cancer"]:
    print(f"\n--- {dataset.upper()} ---")
    sub = main[main["dataset"] == dataset]
    csagg = sub[sub["method"] == "csagg"].set_index(["alpha", "noise_rate", "seed"])["auc"]
    for other in ["fedavg", "uniform_mean", "fedprox", "krum", "trimmed_mean", "coord_median"]:
        vals = sub[sub["method"] == other].set_index(["alpha", "noise_rate", "seed"])["auc"]
        common = csagg.index.intersection(vals.index)
        if len(common) == 0:
            continue
        a, b = csagg[common].values, vals[common].values
        stat, p = stats.wilcoxon(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        diff = a.mean() - b.mean()
        winner = "csagg" if diff > 0 else other
        print(f"  csagg vs {other:14s}: mean_diff={diff:+.4f}  p={p:.4g} {sig}  ({winner} higher)  n={len(common)}")

# ── 3. Collapse check at alpha=0.1 across all 6 methods (BRFSS) ────
print("\n" + "=" * 70)
print("3. ALPHA=0.1 SEED-LEVEL AUC (BRFSS, noise=0.0) -- collapse check, all methods")
print("=" * 70)
collapse = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1) & (main["noise_rate"] == 0.0)]
piv = collapse.pivot_table(index="seed", columns="method", values="auc")[METHOD_ORDER]
print(piv.round(4))
print("\nStd across seeds (lower = more stable):")
print(piv.std().round(4).reindex(METHOD_ORDER))

# ── 4. Noisy-fraction sweep ─────────────────────────────────────────
print("\n" + "=" * 70)
print("4. NOISY-FRACTION SWEEP (BRFSS, alpha=0.1, noise_rate=0.2, all methods)")
print("=" * 70)
nf = final[(final["dataset"] == "brfss") & (final["alpha"] == 0.1) &
           (final["noise_rate"] == 0.2) & (final["feature_set"] == "original")]
nf_table = nf.groupby(["noisy_fraction", "method"])["auc"].mean().round(4).unstack("method")
nf_table = nf_table[[m for m in METHOD_ORDER if m in nf_table.columns]]
print(nf_table)

# ── 5. Feature ablation (expanded 12-feature set) ──────────────────
print("\n" + "=" * 70)
print("5. FEATURE-SET ABLATION (BRFSS, original 4-feature vs expanded 12-feature)")
print("=" * 70)
fa = final[(final["dataset"] == "brfss") & (final["method"].isin(["fedavg", "csagg"])) &
           (final["noisy_fraction"] == 0.2)]
fa_table = fa.groupby(["feature_set", "alpha", "noise_rate", "method"])["auc"].mean().round(4).unstack("method")
print(fa_table)

print("\nBy alpha only (averaged over noise rate, both feature sets):")
fa_alpha = fa.groupby(["feature_set", "alpha", "method"])["auc"].mean().round(4).unstack("method")
print(fa_alpha)

# ── 6. Weighting vs. robustness decomposition ──────────────────────
# FedAvg weights client i by n_i; every other strategy here ignores n_i.
# uniform_mean shares FedAvg's total lack of robustness but the others'
# indifference to n_i, so it splits the FedAvg->X gap into the part caused
# by dropping sample-count weighting and the part caused by robustness.
print("\n" + "=" * 70)
print("6. WEIGHTING vs ROBUSTNESS DECOMPOSITION (BRFSS, alpha=0.1)")
print("=" * 70)
a01 = main[(main["dataset"] == "brfss") & (main["alpha"] == 0.1)]
mu = a01.groupby("method")["auc"].mean()
fed, uni = mu["fedavg"], mu["uniform_mean"]
print(f"  FedAvg            {fed:.4f}")
print(f"  Uniform mean      {uni:.4f}   (weighting effect alone: {uni - fed:+.4f})")
for m in ["fedprox", "csagg", "krum", "trimmed_mean", "coord_median"]:
    total, robust = mu[m] - fed, mu[m] - uni
    share = 100 * (uni - fed) / total if total > 0 else float("nan")
    print(f"  {m:17s} {mu[m]:.4f}   total vs FedAvg {total:+.4f} = "
          f"weighting {uni - fed:+.4f} + robustness {robust:+.4f}"
          f"   ({share:.0f}% of the gap is weighting)" if total > 0 else
          f"  {m:17s} {mu[m]:.4f}   total vs FedAvg {total:+.4f}")

print("\nWilcoxon vs the uniform-mean control (BRFSS, full grid, n=36):")
sub = main[main["dataset"] == "brfss"]
base = sub[sub["method"] == "uniform_mean"].set_index(["alpha", "noise_rate", "seed"])["auc"].sort_index()
for m in ["fedavg", "fedprox", "csagg", "krum", "trimmed_mean", "coord_median"]:
    v = sub[sub["method"] == m].set_index(["alpha", "noise_rate", "seed"])["auc"].sort_index()
    common = base.index.intersection(v.index)
    if len(common) == 0:
        continue
    a, b = v[common].values, base[common].values
    stat, p = stats.wilcoxon(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {m:14s} vs uniform_mean: mean_diff={a.mean() - b.mean():+.4f}  p={p:.4g} {sig}")
