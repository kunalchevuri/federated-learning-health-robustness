import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

from data import (load_brfss, preprocess, split_data,
                  load_breast_cancer_data, split_arrays, partition_data)
from model import get_model
from client import BRFSSClient, FedProxClient
from server_fedavg import run_fedavg
from server_csagg import run_csagg
from server_fedprox import run_fedprox

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Experiment parameters ─────────────────────────────────────────
NUM_CLIENTS    = 20
NUM_ROUNDS     = 50
ALPHAS         = [0.1, 0.5, 1.0]
NOISE_RATES    = [0.0, 0.1, 0.2, 0.3]
NOISY_FRACTIONS = [0.2]
SEEDS          = [42, 123, 456]
METHODS        = ["fedavg", "csagg", "fedprox"]
FEDPROX_MU     = 0.01   # standard proximal coefficient from Li et al. 2020

RESULTS_PATH = "results/experiment_results.csv"
LOCAL_BACKUP  = "/content/results_backup.csv"   # survives Drive hiccups within session

RESULT_COLUMNS = [
    "round", "accuracy", "auc", "method", "dataset",
    "noise_type", "alpha", "noise_rate", "noisy_fraction", "seed",
]


def make_eval_fn(X_test, y_test):
    """Server-side evaluation on the clean held-out test set."""
    input_size = X_test.shape[1]

    def eval_fn(parameters):
        model = get_model(input_size=input_size).to(DEVICE)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(y_test, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=256)

        correct, total = 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for Xb, yb in loader:
                out = model(Xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += len(yb)
                probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.cpu().numpy())

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            auc = 0.5

        return correct / total, auc

    return eval_fn


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


def load_completed():
    completed = set()
    for path in [RESULTS_PATH, LOCAL_BACKUP]:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            print(f"  [{path}] {len(df)} rows found")
            if len(df) == 0:
                continue
            keys = df.drop_duplicates(
                subset=["method", "dataset", "alpha", "noise_rate", "noisy_fraction", "seed"]
            )[["method", "dataset", "alpha", "noise_rate", "noisy_fraction", "seed"]]
            parsed = {
                (r["method"], r["dataset"], float(r["alpha"]),
                 float(r["noise_rate"]), float(r["noisy_fraction"]), int(r["seed"]))
                for _, r in keys.iterrows()
            }
            print(f"  [{path}] {len(parsed)} completed conditions parsed")
            completed |= parsed
        except Exception as e:
            print(f"  WARNING: could not parse {path}: {e}")
    return completed


def run_condition(method, dataset_name, noise_type, alpha,
                  noise_rate, noisy_fraction, seed,
                  X_train, y_train, eval_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n>>> {method.upper()} | {dataset_name} | alpha={alpha} | "
          f"noise={noise_rate} | noisy_frac={noisy_fraction} | seed={seed}")

    partitions = partition_data(X_train, y_train,
                                num_clients=NUM_CLIENTS,
                                alpha=alpha, seed=seed)

    client_fn = make_client_fn(partitions, noise_rate,
                               noisy_fraction, seed, noise_type, method)

    if method == "fedavg":
        history = run_fedavg(client_fn,
                             num_clients=NUM_CLIENTS,
                             num_rounds=NUM_ROUNDS,
                             eval_fn=eval_fn)
    elif method == "csagg":
        history = run_csagg(client_fn,
                            num_clients=NUM_CLIENTS,
                            num_rounds=NUM_ROUNDS,
                            eval_fn=eval_fn)
    elif method == "fedprox":
        history = run_fedprox(client_fn,
                              num_clients=NUM_CLIENTS,
                              num_rounds=NUM_ROUNDS,
                              eval_fn=eval_fn)
    else:
        raise ValueError(f"Unknown method: {method}")

    auc_by_round = {r: v for r, v in history.metrics_distributed.get("auc", [])}
    results = []
    for round_num, acc in history.metrics_distributed.get("accuracy", []):
        results.append({
            "round":          round_num,
            "accuracy":       acc,
            "auc":            auc_by_round.get(round_num, float("nan")),
            "method":         method,
            "dataset":        dataset_name,
            "noise_type":     noise_type,
            "alpha":          alpha,
            "noise_rate":     noise_rate,
            "noisy_fraction": noisy_fraction,
            "seed":           seed,
        })

    return results


def main():
    os.makedirs("results", exist_ok=True)

    for path in [RESULTS_PATH, LOCAL_BACKUP]:
        if not os.path.exists(path):
            pd.DataFrame(columns=RESULT_COLUMNS).to_csv(path, index=False)
    if os.path.getsize(RESULTS_PATH) > len(",".join(RESULT_COLUMNS)) + 5:
        print("Found existing results — will skip completed conditions")
    else:
        print("Starting fresh")

    completed = load_completed()

    # ── Load datasets and build eval functions once ───────────────
    print("\nLoading BRFSS data...")
    df = load_brfss()
    df = preprocess(df)
    X_train_b, X_test_b, y_train_b, y_test_b = split_data(df)
    eval_fn_b = make_eval_fn(X_test_b, y_test_b)
    print(f"  Test set: {len(y_test_b)} samples | "
          f"depression rate: {y_test_b.mean():.2%}")

    print("\nLoading Breast Cancer data...")
    X_bc, y_bc = load_breast_cancer_data()
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = split_arrays(X_bc, y_bc)
    eval_fn_bc = make_eval_fn(X_test_bc, y_test_bc)
    print(f"  Test set: {len(y_test_bc)} samples | "
          f"malignant rate: {y_test_bc.mean():.2%}")

    dataset_configs = {
        "brfss": {
            "noise_type": "asymmetric",
            "X_train":    X_train_b,
            "y_train":    y_train_b,
            "eval_fn":    eval_fn_b,
        },
        "breast_cancer": {
            "noise_type": "symmetric",
            "X_train":    X_train_bc,
            "y_train":    y_train_bc,
            "eval_fn":    eval_fn_bc,
        },
    }

    # Count total conditions
    total = 0
    for dataset_name in dataset_configs:
        for method in METHODS:
            for alpha in ALPHAS:
                for noise_rate in NOISE_RATES:
                    for noisy_fraction in NOISY_FRACTIONS:
                        if noise_rate == 0.0 and noisy_fraction > NOISY_FRACTIONS[0]:
                            continue
                        for seed in SEEDS:
                            total += 1

    print(f"\nTotal conditions: {total} | "
          f"Completed: {len(completed)} | "
          f"Remaining: {total - len(completed)}\n")

    condition_num = 0
    for dataset_name, cfg in dataset_configs.items():
        for method in METHODS:
            for alpha in ALPHAS:
                for noise_rate in NOISE_RATES:
                    for noisy_fraction in NOISY_FRACTIONS:
                        if noise_rate == 0.0 and noisy_fraction > NOISY_FRACTIONS[0]:
                            continue
                        for seed in SEEDS:
                            condition_num += 1
                            key = (method, dataset_name, float(alpha),
                                   float(noise_rate), float(noisy_fraction), int(seed))

                            if key in completed:
                                print(f"[{condition_num}/{total}] SKIP: "
                                      f"{method} | {dataset_name} | "
                                      f"alpha={alpha} | noise={noise_rate} | seed={seed}")
                                continue

                            print(f"\n[{condition_num}/{total}]", end="")
                            results = run_condition(
                                method, dataset_name,
                                cfg["noise_type"], alpha,
                                noise_rate, noisy_fraction,
                                seed, cfg["X_train"], cfg["y_train"],
                                cfg["eval_fn"],
                            )

                            df_res = pd.DataFrame(results)
                            for path in [RESULTS_PATH, LOCAL_BACKUP]:
                                with open(path, "a", newline="") as f:
                                    df_res.to_csv(f, header=False, index=False)
                                    f.flush()
                                    os.fsync(f.fileno())
                            print(f"  Saved {len(results)} rows")

    print(f"\nDone! Results in {RESULTS_PATH}")
    df_out = pd.read_csv(RESULTS_PATH)
    print(f"Total rows: {len(df_out)}")


if __name__ == "__main__":
    main()
