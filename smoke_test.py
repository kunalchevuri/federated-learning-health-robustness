"""
Smoke test — run this BEFORE the full experiment.
Pastes as a cell in Colab or run via: %run smoke_test.py
All lines must print OK. Any FAIL = fix before running.
Takes ~3-5 minutes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd

PASS = "[OK] "
FAIL = "[FAIL]"
failures = 0

def check(name, fn):
    global failures
    try:
        result = fn()
        msg = f"  ({result})" if result else ""
        print(f"{PASS} {name}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {name} — {e}")
        failures += 1
        return False

print("=" * 55)
print("SMOKE TEST — 2 rounds, both datasets, all methods")
print("=" * 55)

# ── Load data ─────────────────────────────────────────────
from data import (load_brfss, preprocess, split_data,
                  load_breast_cancer_data, split_arrays, partition_data)
from client import BRFSSClient
from server_fedavg import run_fedavg
from server_csagg import run_csagg

print("\n[1] Loading datasets...")
df = None
X_train_b = X_train_bc = y_train_b = y_train_bc = None

def _load_brfss():
    global df, X_train_b, y_train_b
    df = preprocess(load_brfss())
    X_train_b, _, y_train_b, _ = split_data(df)
    return f"{len(X_train_b)} train rows"
check("BRFSS load", _load_brfss)

def _load_bc():
    global X_train_bc, y_train_bc
    X_bc, y_bc = load_breast_cancer_data()
    X_train_bc, _, y_train_bc, _ = split_arrays(X_bc, y_bc)
    return f"{len(X_train_bc)} train rows"
check("Breast Cancer load", _load_bc)

# ── Mini runs — 5 clients, 2 rounds ──────────────────────
print("\n[2] Mini runs (5 clients, 2 rounds each)...")
NC = 5
NR = 2

results_rows = []

for dataset_name, X_train, y_train, noise_type in [
    ("brfss",         X_train_b,  y_train_b,  "asymmetric"),
    ("breast_cancer", X_train_bc, y_train_bc, "symmetric"),
]:
    for method in ["fedavg", "csagg"]:
        label = f"{dataset_name}/{method}"

        def _run(ds=dataset_name, xt=X_train, yt=y_train, nt=noise_type, m=method, lbl=label):
            global results_rows
            parts = partition_data(xt, yt, num_clients=NC, alpha=0.5, seed=42)
            def cfn(cid):
                X, y = parts[int(cid)]
                return BRFSSClient(X, y, noise_rate=0.2, seed=42, noise_type=nt)
            if m == "fedavg":
                h = run_fedavg(cfn, num_clients=NC, num_rounds=NR)
            else:
                h = run_csagg(cfn, num_clients=NC, num_rounds=NR)
            for rn, acc in h.metrics_distributed["accuracy"]:
                auc = dict(h.metrics_distributed.get("auc", [])).get(rn, float("nan"))
                results_rows.append({
                    "dataset": ds, "method": m, "round": rn,
                    "accuracy": acc, "auc": auc,
                })
            aucs = [v for _, v in h.metrics_distributed.get("auc", [])]
            accs = [v for _, v in h.metrics_distributed["accuracy"]]
            return f"acc={[round(a,3) for a in accs]} auc={[round(a,3) for a in aucs]}"

        check(label, _run)

# ── Validate CSV output ───────────────────────────────────
print("\n[3] Validating output...")
df_out = pd.DataFrame(results_rows)

def _no_nan_auc():
    bad = df_out["auc"].isna().sum()
    assert bad == 0, f"{bad} NaN AUC values"
    return f"all {len(df_out)} auc values clean"
check("No NaN in AUC", _no_nan_auc)

def _both_datasets():
    ds = set(df_out["dataset"].unique())
    assert ds == {"brfss", "breast_cancer"}, f"got {ds}"
    return "brfss and breast_cancer present"
check("Both datasets in output", _both_datasets)

def _both_methods():
    ms = set(df_out["method"].unique())
    assert ms == {"fedavg", "csagg"}, f"got {ms}"
    return "fedavg and csagg present"
check("Both methods in output", _both_methods)

def _acc_sane():
    lo, hi = df_out["accuracy"].min(), df_out["accuracy"].max()
    assert 0.0 <= lo and hi <= 1.0, f"acc out of range: {lo:.3f}-{hi:.3f}"
    return f"range {lo:.3f}-{hi:.3f}"
check("Accuracy in [0, 1]", _acc_sane)

def _auc_sane():
    lo, hi = df_out["auc"].min(), df_out["auc"].max()
    assert 0.0 <= lo and hi <= 1.0, f"auc out of range: {lo:.3f}-{hi:.3f}"
    return f"range {lo:.3f}-{hi:.3f}"
check("AUC in [0, 1]", _auc_sane)

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 55)
if failures == 0:
    print("ALL OK — safe to run full experiment")
else:
    print(f"{failures} FAILURE(S) — fix before running full experiment")
print("=" * 55)
