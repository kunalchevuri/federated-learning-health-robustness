"""
Pre-flight validation script.
Run this on Colab before starting the full experiment.
All checks must print PASS. Any FAIL = fix before running.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(name, fn):
    try:
        result = fn()
        msg = f"  ({result})" if result else ""
        print(f"{PASS}  {name}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL}  {name} — {e}")
        return False

failures = 0

# ── 1. Imports ────────────────────────────────────────────────────
print("\n[1] Imports")
failures += not check("torch",         lambda: __import__("torch").__version__)
failures += not check("flwr",          lambda: __import__("flwr").__version__)
failures += not check("sklearn",       lambda: __import__("sklearn").__version__)
failures += not check("pyreadstat",    lambda: __import__("pyreadstat").__version__)
failures += not check("numpy",         lambda: __import__("numpy").__version__)
failures += not check("pandas",        lambda: __import__("pandas").__version__)

# ── 2. GPU ────────────────────────────────────────────────────────
print("\n[2] Hardware")
import torch
failures += not check("CUDA available (expected on Colab)",
                       lambda: "YES" if torch.cuda.is_available() else (_ for _ in ()).throw(RuntimeError("No GPU — change runtime to T4")))

# ── 3. BRFSS data ─────────────────────────────────────────────────
print("\n[3] BRFSS dataset")
from data import load_brfss, preprocess, split_data, partition_data

failures += not check("data/LLCP2023.XPT exists",
                       lambda: "found" if os.path.exists("data/LLCP2023.XPT") else (_ for _ in ()).throw(FileNotFoundError("Upload data/LLCP2023.XPT to Drive")))

df = None
def _load_brfss():
    global df
    df = load_brfss()
    df = preprocess(df)
    return f"{len(df)} rows"
failures += not check("load + preprocess BRFSS", _load_brfss)

X_train_b, X_test_b, y_train_b, y_test_b = [None]*4
def _split_brfss():
    global X_train_b, X_test_b, y_train_b, y_test_b
    X_train_b, X_test_b, y_train_b, y_test_b = split_data(df)
    return f"train={len(X_train_b)} test={len(X_test_b)}"
failures += not check("split BRFSS", _split_brfss)

failures += not check("partition BRFSS (alpha=0.5, 5 clients)",
                       lambda: f"{len(partition_data(X_train_b, y_train_b, num_clients=5, alpha=0.5))} partitions")

# ── 4. Breast Cancer data ─────────────────────────────────────────
print("\n[4] Breast Cancer dataset")
from data import load_breast_cancer_data, split_arrays

X_bc, y_bc = None, None
def _load_bc():
    global X_bc, y_bc
    X_bc, y_bc = load_breast_cancer_data()
    return f"{len(y_bc)} samples, {X_bc.shape[1]} features"
failures += not check("load Breast Cancer", _load_bc)

X_train_bc, X_test_bc, y_train_bc, y_test_bc = [None]*4
def _split_bc():
    global X_train_bc, X_test_bc, y_train_bc, y_test_bc
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = split_arrays(X_bc, y_bc)
    return f"train={len(X_train_bc)} test={len(X_test_bc)}"
failures += not check("split Breast Cancer", _split_bc)

failures += not check("partition Breast Cancer (alpha=0.5, 5 clients)",
                       lambda: f"{len(partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5))} partitions")

# ── 5. Noise functions ────────────────────────────────────────────
print("\n[5] Noise injection")
import numpy as np
from noise import inject_noise, inject_symmetric_noise

y_dummy = np.array([1]*50 + [0]*50)
def _asym():
    yn = inject_noise(y_dummy, 0.2, seed=42)
    assert yn.shape == y_dummy.shape
    flipped = (yn != y_dummy).sum()
    return f"{flipped} labels flipped"
failures += not check("asymmetric noise (BRFSS)", _asym)

def _sym():
    yn = inject_symmetric_noise(y_dummy, 0.2, seed=42)
    assert yn.shape == y_dummy.shape
    flipped = (yn != y_dummy).sum()
    return f"{flipped} labels flipped"
failures += not check("symmetric noise (Breast Cancer)", _sym)

# ── 6. Model ──────────────────────────────────────────────────────
print("\n[6] Model")
from model import get_model

failures += not check("model input_size=4  (BRFSS)",
                       lambda: str(get_model(4)(torch.randn(8,4)).shape))
failures += not check("model input_size=30 (Breast Cancer)",
                       lambda: str(get_model(30)(torch.randn(8,30)).shape))

# ── 7. Client ─────────────────────────────────────────────────────
print("\n[7] Client")
from client import BRFSSClient

partitions_b  = partition_data(X_train_b,  y_train_b,  num_clients=5, alpha=0.5)
partitions_bc = partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5)

def _client_brfss():
    X, y = partitions_b[0]
    c = BRFSSClient(X, y, noise_rate=0.2, seed=42, noise_type="asymmetric")
    params = c.get_parameters(config={})
    return f"{len(params)} param arrays, model on {next(c.model.parameters()).device}"
failures += not check("BRFSSClient (BRFSS, asymmetric)", _client_brfss)

def _client_bc():
    X, y = partitions_bc[0]
    c = BRFSSClient(X, y, noise_rate=0.2, seed=42, noise_type="symmetric")
    params = c.get_parameters(config={})
    return f"{len(params)} param arrays, model on {next(c.model.parameters()).device}"
failures += not check("BRFSSClient (Breast Cancer, symmetric)", _client_bc)

# ── 8. End-to-end mini runs ───────────────────────────────────────
print("\n[8] End-to-end (2 clients, 3 rounds each dataset)")
from server_fedavg import run_fedavg
from server_csagg import run_csagg

def _e2e_brfss_fedavg():
    p = partition_data(X_train_b, y_train_b, num_clients=2, alpha=0.5)
    def cfn(cid):
        X, y = p[int(cid)]
        return BRFSSClient(X, y, noise_rate=0.2, seed=42, noise_type="asymmetric")
    h = run_fedavg(cfn, num_clients=2, num_rounds=3)
    accs = [round(a,4) for _,a in h.metrics_distributed["accuracy"]]
    return f"acc={accs}"
failures += not check("BRFSS + FedAvg", _e2e_brfss_fedavg)

def _e2e_bc_csagg():
    p = partition_data(X_train_bc, y_train_bc, num_clients=2, alpha=0.5)
    def cfn(cid):
        X, y = p[int(cid)]
        return BRFSSClient(X, y, noise_rate=0.2, seed=42, noise_type="symmetric")
    h = run_csagg(cfn, num_clients=2, num_rounds=3)
    accs = [round(a,4) for _,a in h.metrics_distributed["accuracy"]]
    return f"acc={accs}"
failures += not check("Breast Cancer + CS-Agg", _e2e_bc_csagg)

# ── 9. Results directory writable ────────────────────────────────
print("\n[9] Output")
def _results_dir():
    os.makedirs("results", exist_ok=True)
    test_path = "results/.write_test"
    with open(test_path, "w") as f:
        f.write("ok")
    os.remove(test_path)
    return "results/ is writable"
failures += not check("results/ directory writable", _results_dir)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
if failures == 0:
    print("ALL CHECKS PASSED — safe to run full experiment")
else:
    print(f"{failures} CHECK(S) FAILED — fix before running")
print("="*50)
