"""
Extended experiment runner for the TPS resubmission.

Adds, on top of the original FedAvg / CS-Agg / FedProx grid:
  - Three classical Byzantine-robust baselines: Krum, trimmed mean,
    coordinate-wise median (server_robust.py).
  - A noisy-client-fraction sweep (0.2-0.6) characterizing where CS-Agg's
    majority-consensus assumption breaks down (BRFSS only, alpha=0.1).
  - A 12-feature "expanded" BRFSS feature set to check the primary
    findings aren't an artifact of using only 4 hand-picked features.

Designed to be sharded across multiple OS processes (see run_shards.py)
rather than parallelized with multiprocessing, to avoid pickling closures
across process boundaries on Windows. Each shard writes to its own CSV;
merge_results.py concatenates and dedupes at the end.

Usage:
  python run_experiments_v2.py --out results/v2_shard1.csv \
      --methods krum,trimmed_mean --datasets brfss,breast_cancer \
      --alphas 0.1,0.5,1.0 --noise-rates 0.0,0.1,0.2,0.3 \
      --noisy-fractions 0.2 --seeds 42,123,456 --feature-set original
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from data import (load_brfss, preprocess, preprocess_expanded, split_data,
                  load_breast_cancer_data, split_arrays, partition_data)
from client import BRFSSClient, FedProxClient
from server_fedavg import run_fedavg
from server_csagg import run_csagg
from server_fedprox import run_fedprox
from server_robust import (run_krum, run_trimmed_mean, run_coordinate_median,
                           run_uniform_mean)
from run_experiments import make_eval_fn

NUM_CLIENTS = 20
NUM_ROUNDS = 50
FEDPROX_MU = 0.01
NUM_BYZANTINE = 4  # matches the fixed 20% noisy_fraction (4 of 20 clients)

RESULT_COLUMNS = [
    "round", "accuracy", "auc", "method", "dataset", "feature_set",
    "noise_type", "alpha", "noise_rate", "noisy_fraction", "seed",
]

METHOD_RUNNERS = {
    "fedavg":        lambda cfn, ef: run_fedavg(cfn, NUM_CLIENTS, NUM_ROUNDS, eval_fn=ef),
    "csagg":         lambda cfn, ef: run_csagg(cfn, NUM_CLIENTS, NUM_ROUNDS, eval_fn=ef),
    "fedprox":       lambda cfn, ef: run_fedprox(cfn, NUM_CLIENTS, NUM_ROUNDS, eval_fn=ef),
    "krum":          lambda cfn, ef: run_krum(cfn, NUM_CLIENTS, NUM_ROUNDS, num_byzantine=NUM_BYZANTINE, eval_fn=ef),
    "trimmed_mean":  lambda cfn, ef: run_trimmed_mean(cfn, NUM_CLIENTS, NUM_ROUNDS, num_byzantine=NUM_BYZANTINE, eval_fn=ef),
    "coord_median":  lambda cfn, ef: run_coordinate_median(cfn, NUM_CLIENTS, NUM_ROUNDS, eval_fn=ef),
    "uniform_mean":  lambda cfn, ef: run_uniform_mean(cfn, NUM_CLIENTS, NUM_ROUNDS, eval_fn=ef),
}


def make_client_fn(partitions, noise_rate, noisy_fraction, seed, noise_type, method):
    num_noisy = int(NUM_CLIENTS * noisy_fraction)

    def client_fn(cid):
        cid = int(cid)
        X, y = partitions[cid]
        client_noise = noise_rate if cid < num_noisy else 0.0
        if method == "fedprox":
            return FedProxClient(X, y, noise_rate=client_noise,
                                 seed=seed, noise_type=noise_type, mu=FEDPROX_MU)
        return BRFSSClient(X, y, noise_rate=client_noise,
                           seed=seed, noise_type=noise_type)
    return client_fn


def load_completed(path):
    completed = set()
    if not os.path.exists(path):
        return completed
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            return completed
        keys = df.drop_duplicates(
            subset=["method", "dataset", "feature_set", "alpha", "noise_rate", "noisy_fraction", "seed"]
        )[["method", "dataset", "feature_set", "alpha", "noise_rate", "noisy_fraction", "seed"]]
        completed = {
            (r["method"], r["dataset"], r["feature_set"], float(r["alpha"]),
             float(r["noise_rate"]), float(r["noisy_fraction"]), int(r["seed"]))
            for _, r in keys.iterrows()
        }
    except Exception as e:
        print(f"  WARNING: could not parse {path}: {e}")
    return completed


def run_condition(method, dataset_name, feature_set, noise_type, alpha,
                  noise_rate, noisy_fraction, seed, X_train, y_train, eval_fn):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n>>> {method.upper()} | {dataset_name}/{feature_set} | alpha={alpha} | "
          f"noise={noise_rate} | noisy_frac={noisy_fraction} | seed={seed}")

    partitions = partition_data(X_train, y_train, num_clients=NUM_CLIENTS,
                                alpha=alpha, seed=seed)
    client_fn = make_client_fn(partitions, noise_rate, noisy_fraction, seed, noise_type, method)
    history = METHOD_RUNNERS[method](client_fn, eval_fn)

    auc_by_round = {r: v for r, v in history.metrics_distributed.get("auc", [])}
    results = []
    for round_num, acc in history.metrics_distributed.get("accuracy", []):
        results.append({
            "round": round_num, "accuracy": acc, "auc": auc_by_round.get(round_num, float("nan")),
            "method": method, "dataset": dataset_name, "feature_set": feature_set,
            "noise_type": noise_type, "alpha": alpha, "noise_rate": noise_rate,
            "noisy_fraction": noisy_fraction, "seed": seed,
        })
    return results


def parse_list(s, cast=str):
    return [cast(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--methods", required=True, help="comma list from: " + ",".join(METHOD_RUNNERS))
    p.add_argument("--datasets", default="brfss,breast_cancer")
    p.add_argument("--alphas", default="0.1,0.5,1.0")
    p.add_argument("--noise-rates", default="0.0,0.1,0.2,0.3")
    p.add_argument("--noisy-fractions", default="0.2")
    p.add_argument("--seeds", default="42,123,456")
    p.add_argument("--feature-set", default="original", choices=["original", "expanded"])
    args = p.parse_args()

    methods = parse_list(args.methods)
    datasets = parse_list(args.datasets)
    alphas = parse_list(args.alphas, float)
    noise_rates = parse_list(args.noise_rates, float)
    noisy_fractions = parse_list(args.noisy_fractions, float)
    seeds = parse_list(args.seeds, int)
    feature_set = args.feature_set

    for m in methods:
        assert m in METHOD_RUNNERS, f"unknown method {m}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(args.out, index=False)
    completed = load_completed(args.out)
    print(f"[{args.out}] {len(completed)} conditions already completed")

    dataset_configs = {}
    if "brfss" in datasets:
        print("\nLoading BRFSS data...")
        raw = load_brfss()
        df = preprocess_expanded(raw) if feature_set == "expanded" else preprocess(raw)
        del raw  # the 350-col raw frame is ~1.3GB and unneeded after preprocessing
        X_train, X_test, y_train, y_test = split_data(df)
        dataset_configs["brfss"] = {
            "noise_type": "asymmetric", "X_train": X_train, "y_train": y_train,
            "eval_fn": make_eval_fn(X_test, y_test),
        }
    if "breast_cancer" in datasets:
        print("\nLoading Breast Cancer data...")
        X_bc, y_bc = load_breast_cancer_data()
        X_train_bc, X_test_bc, y_train_bc, y_test_bc = split_arrays(X_bc, y_bc)
        dataset_configs["breast_cancer"] = {
            "noise_type": "symmetric", "X_train": X_train_bc, "y_train": y_train_bc,
            "eval_fn": make_eval_fn(X_test_bc, y_test_bc),
        }

    conditions = []
    for dataset_name, cfg in dataset_configs.items():
        for method in methods:
            for alpha in alphas:
                for noise_rate in noise_rates:
                    for noisy_fraction in noisy_fractions:
                        if noise_rate == 0.0 and noisy_fraction != noisy_fractions[0]:
                            continue
                        for seed in seeds:
                            conditions.append((method, dataset_name, alpha, noise_rate, noisy_fraction, seed, cfg))

    total = len(conditions)
    print(f"\nTotal conditions in this shard: {total}\n")

    for i, (method, dataset_name, alpha, noise_rate, noisy_fraction, seed, cfg) in enumerate(conditions, 1):
        key = (method, dataset_name, feature_set, float(alpha), float(noise_rate), float(noisy_fraction), int(seed))
        if key in completed:
            print(f"[{i}/{total}] SKIP: {key}")
            continue

        results = run_condition(method, dataset_name, feature_set, cfg["noise_type"],
                                alpha, noise_rate, noisy_fraction, seed,
                                cfg["X_train"], cfg["y_train"], cfg["eval_fn"])
        df_res = pd.DataFrame(results)
        with open(args.out, "a", newline="") as f:
            df_res.to_csv(f, header=False, index=False)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{i}/{total}] Saved {len(results)} rows -> {args.out}")

    print(f"\nDone with shard. Results in {args.out}")


if __name__ == "__main__":
    main()
