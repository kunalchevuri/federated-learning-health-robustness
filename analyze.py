import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/experiment_results.csv")
final = df[df["round"] == 50].copy()

print("=" * 60)
print("MEAN AUC BY METHOD AND DATASET (final round)")
print("=" * 60)
summary = final.groupby(["method", "dataset"])["auc"].agg(["mean", "std"]).round(4)
print(summary)

print("\n" + "=" * 60)
print("MEAN AUC BY METHOD / DATASET / NOISE RATE")
print("=" * 60)
noise_table = final.groupby(["method", "dataset", "noise_rate"])["auc"].mean().round(4).unstack("method")
print(noise_table)

print("\n" + "=" * 60)
print("WILCOXON SIGNED-RANK TESTS (AUC, final round)")
print("=" * 60)

for dataset in ["brfss", "breast_cancer"]:
    print(f"\n--- {dataset.upper()} ---")
    sub = final[final["dataset"] == dataset]

    fedavg = sub[sub["method"] == "fedavg"]["auc"].values
    csagg  = sub[sub["method"] == "csagg"]["auc"].values
    fedprox = sub[sub["method"] == "fedprox"]["auc"].values

    for m1, v1, m2, v2 in [
        ("fedavg", fedavg, "csagg",   csagg),
        ("fedavg", fedavg, "fedprox", fedprox),
        ("csagg",  csagg,  "fedprox", fedprox),
    ]:
        stat, p = stats.wilcoxon(v1, v2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        winner = m1 if np.mean(v1) > np.mean(v2) else m2
        print(f"  {m1} vs {m2}: p={p:.4f} {sig}  >>  {winner} wins  "
              f"(means: {np.mean(v1):.4f} vs {np.mean(v2):.4f})")
