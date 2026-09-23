"""
Batched launcher for run_experiments_v2.py shards.

Runs at most MAX_CONCURRENT shard processes at a time (this machine has
only ~5GB free RAM, so we can't fire all shards at once -- each process
holds the BRFSS train/test arrays plus a small BinaryMLP). Each shard is
resume-safe on its own (skips already-completed conditions in its --out
CSV), so this script can simply be re-run if interrupted.
"""
import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor

PYTHON = os.path.join("venv", "Scripts", "python.exe")
SCRIPT = os.path.join("src", "run_experiments_v2.py")
MAX_CONCURRENT = int(sys.argv[1]) if len(sys.argv) > 1 else 3

COMMON = dict(seeds="42,123,456", feature_set="original")

SHARDS = [
    # --- Set A: new Byzantine-robust baselines, full original grid ---
    dict(out="results/v2_shards/krum.csv", methods="krum",
         datasets="brfss,breast_cancer", alphas="0.1,0.5,1.0",
         noise_rates="0.0,0.1,0.2,0.3", noisy_fractions="0.2", **COMMON),
    dict(out="results/v2_shards/trimmed_mean.csv", methods="trimmed_mean",
         datasets="brfss,breast_cancer", alphas="0.1,0.5,1.0",
         noise_rates="0.0,0.1,0.2,0.3", noisy_fractions="0.2", **COMMON),
    dict(out="results/v2_shards/coord_median.csv", methods="coord_median",
         datasets="brfss,breast_cancer", alphas="0.1,0.5,1.0",
         noise_rates="0.0,0.1,0.2,0.3", noisy_fractions="0.2", **COMMON),

    # --- Set B: noisy-fraction sweep, BRFSS only, alpha=0.1, noise=0.2 ---
    dict(out="results/v2_shards/nf_fedavg.csv", methods="fedavg",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
    dict(out="results/v2_shards/nf_csagg.csv", methods="csagg",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
    dict(out="results/v2_shards/nf_fedprox.csv", methods="fedprox",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
    dict(out="results/v2_shards/nf_krum.csv", methods="krum",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
    dict(out="results/v2_shards/nf_trimmed_mean.csv", methods="trimmed_mean",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
    dict(out="results/v2_shards/nf_coord_median.csv", methods="coord_median",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),

    # --- Set C: feature-selection ablation, BRFSS only, expanded features ---
    dict(out="results/v2_shards/exp_fedavg.csv", methods="fedavg",
         datasets="brfss", alphas="0.1,0.5,1.0", noise_rates="0.0,0.2",
         noisy_fractions="0.2", seeds="42,123,456", feature_set="expanded"),
    dict(out="results/v2_shards/exp_csagg.csv", methods="csagg",
         datasets="brfss", alphas="0.1,0.5,1.0", noise_rates="0.0,0.2",
         noisy_fractions="0.2", seeds="42,123,456", feature_set="expanded"),

    # --- Set D: uniform (unweighted) mean control ------------------------
    # Isolates "dropping sample-count weighting" from "using a robust
    # statistic". FedAvg weights client i by n_i; Krum, trimmed mean,
    # coordinate median and post-warmup CS-Agg all ignore n_i. Without this
    # control their advantage under extreme Dirichlet skew is confounded.
    dict(out="results/v2_shards/um_brfss.csv", methods="uniform_mean",
         datasets="brfss", alphas="0.1,0.5,1.0",
         noise_rates="0.0,0.1,0.2,0.3", noisy_fractions="0.2", **COMMON),
    dict(out="results/v2_shards/um_bc.csv", methods="uniform_mean",
         datasets="breast_cancer", alphas="0.1,0.5,1.0",
         noise_rates="0.0,0.1,0.2,0.3", noisy_fractions="0.2", **COMMON),
    dict(out="results/v2_shards/um_nf.csv", methods="uniform_mean",
         datasets="brfss", alphas="0.1", noise_rates="0.2",
         noisy_fractions="0.3,0.4,0.5,0.6", **COMMON),
]


def run_shard(cfg):
    cmd = [PYTHON, SCRIPT,
           "--out", cfg["out"], "--methods", cfg["methods"],
           "--datasets", cfg["datasets"], "--alphas", cfg["alphas"],
           "--noise-rates", cfg["noise_rates"],
           "--noisy-fractions", cfg["noisy_fractions"],
           "--seeds", cfg["seeds"], "--feature-set", cfg["feature_set"]]
    log_path = cfg["out"].replace(".csv", ".log")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    print(f"[START] {cfg['out']}")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"[DONE]  {cfg['out']} -- {status}")
    return cfg["out"], result.returncode


if __name__ == "__main__":
    os.makedirs("results/v2_shards", exist_ok=True)
    print(f"Running {len(SHARDS)} shards, {MAX_CONCURRENT} concurrent")
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        results = list(ex.map(run_shard, SHARDS))
    print("\n=== SUMMARY ===")
    failed = [o for o, rc in results if rc != 0]
    for out, rc in results:
        print(f"  {'OK ' if rc == 0 else 'FAIL'} {out}")
    if failed:
        print(f"\n{len(failed)} shard(s) FAILED -- check .log files")
        sys.exit(1)
    print("\nAll shards completed successfully.")
