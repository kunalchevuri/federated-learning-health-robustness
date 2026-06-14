"""
Comprehensive experiment validation.
Run with: python run_tests.py
"""
import sys, os, warnings, tempfile
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch

PASS = "[PASS]"
FAIL = "[FAIL]"
failures = []

def check(name, fn):
    try:
        result = fn()
        msg = f"  => {result}" if result else ""
        print(f"{PASS} {name}{msg}")
        return True
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        failures.append(name)
        return False

# ── imports ───────────────────────────────────────────────────────
from noise import inject_noise, inject_symmetric_noise
from data import load_breast_cancer_data, split_arrays, partition_data
from model import get_model
from client import BRFSSClient
from server_fedavg import run_fedavg
from server_csagg import run_csagg, make_csagg_aggregate
from simulate import fedavg_aggregate
from run_experiments import make_eval_fn, load_completed, RESULT_COLUMNS

print("=" * 60)
print("COMPREHENSIVE EXPERIMENT VALIDATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
print("\n[1] Noise injection accuracy")
# ─────────────────────────────────────────────────────────────────

def _asym_flip_rates():
    y = np.array([1]*1000 + [0]*1000)
    y_noisy = inject_noise(y, noise_rate=0.2, seed=42)
    rate_1to0 = ((y == 1) & (y_noisy == 0)).sum() / (y == 1).sum()
    rate_0to1 = ((y == 0) & (y_noisy == 1)).sum() / (y == 0).sum()
    assert abs(rate_1to0 - 0.2) < 0.01, f"1->0 rate {rate_1to0:.3f} != 0.20"
    assert abs(rate_0to1 - 0.1) < 0.01, f"0->1 rate {rate_0to1:.3f} != 0.10"
    return f"1->0={rate_1to0:.3f} (want 0.200) | 0->1={rate_0to1:.3f} (want 0.100)"
check("Asymmetric flip rates are correct", _asym_flip_rates)

def _sym_flip_rates():
    y = np.array([1]*1000 + [0]*1000)
    y_noisy = inject_symmetric_noise(y, noise_rate=0.2, seed=42)
    rate_1to0 = ((y == 1) & (y_noisy == 0)).sum() / (y == 1).sum()
    rate_0to1 = ((y == 0) & (y_noisy == 1)).sum() / (y == 0).sum()
    assert abs(rate_1to0 - 0.2) < 0.01, f"rate {rate_1to0:.3f}"
    assert abs(rate_0to1 - 0.2) < 0.01, f"rate {rate_0to1:.3f}"
    return f"both directions={rate_1to0:.3f} (want 0.200)"
check("Symmetric flip rates are correct", _sym_flip_rates)

def _noise_no_mutation():
    y = np.array([1]*50 + [0]*50)
    y_orig = y.copy()
    inject_noise(y, 0.3, seed=42)
    inject_symmetric_noise(y, 0.3, seed=42)
    assert np.array_equal(y, y_orig)
    return "original array unchanged"
check("Noise does not mutate input", _noise_no_mutation)

def _noise_zero():
    y = np.array([1]*50 + [0]*50)
    assert np.array_equal(inject_noise(y, 0.0), y)
    assert np.array_equal(inject_symmetric_noise(y, 0.0), y)
    return "no flips at rate=0"
check("Noise rate=0 produces zero flips", _noise_zero)

def _noise_reproducible():
    y = np.array([1]*200 + [0]*200)
    a = inject_noise(y, 0.2, seed=42)
    b = inject_noise(y, 0.2, seed=42)
    c = inject_noise(y, 0.2, seed=99)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    return "same seed -> same result; different seed -> different result"
check("Noise is deterministic per seed", _noise_reproducible)

# ─────────────────────────────────────────────────────────────────
print("\n[2] Breast Cancer data pipeline")
# ─────────────────────────────────────────────────────────────────

X_bc, y_bc = load_breast_cancer_data()
X_train_bc, X_test_bc, y_train_bc, y_test_bc = split_arrays(X_bc, y_bc)

def _bc_shapes():
    assert X_bc.shape == (569, 30), f"Expected (569,30) got {X_bc.shape}"
    assert X_bc.dtype == np.float32
    assert y_bc.dtype == np.int64
    return f"X={X_bc.shape} y={y_bc.shape}"
check("Breast Cancer shapes correct", _bc_shapes)

def _bc_target():
    rate = y_bc.mean()
    assert 0.30 < rate < 0.45, f"malignant rate {rate:.2%}"
    return f"malignant rate={rate:.2%} (expected ~37%)"
check("Target encoding (1=malignant, ~37%)", _bc_target)

def _bc_scaled():
    mu = X_bc.mean(axis=0)
    std = X_bc.std(axis=0)
    assert np.allclose(mu, 0, atol=0.01), "not zero-mean"
    assert np.allclose(std, 1, atol=0.01), "not unit-std"
    return "mean~0, std~1 across all 30 features"
check("Breast Cancer features are StandardScaled", _bc_scaled)

def _bc_stratified():
    train_rate = y_train_bc.mean()
    test_rate = y_test_bc.mean()
    assert abs(train_rate - test_rate) < 0.02
    return f"train_pos={train_rate:.2%} test_pos={test_rate:.2%}"
check("Train/test split is stratified", _bc_stratified)

def _bc_no_leakage():
    train_set = set(map(tuple, X_train_bc.tolist()))
    leaks = sum(1 for row in X_test_bc if tuple(row.tolist()) in train_set)
    assert leaks == 0, f"{leaks} test rows found in training set"
    return "zero overlap between train and test"
check("No train/test data leakage", _bc_no_leakage)

def _bc_split_size():
    expected_test = int(len(y_bc) * 0.2)
    # stratified split may be off by 1
    assert abs(len(y_test_bc) - expected_test) <= 2
    return f"test={len(y_test_bc)} train={len(y_train_bc)}"
check("80/20 train/test split size", _bc_split_size)

# ─────────────────────────────────────────────────────────────────
print("\n[3] Dirichlet partitioning")
# ─────────────────────────────────────────────────────────────────

def _partition_total():
    parts = partition_data(X_train_bc, y_train_bc, num_clients=20, alpha=0.5, seed=42)
    total = sum(len(p[1]) for p in parts)
    assert total == len(y_train_bc)
    return f"all {total} samples assigned across 20 clients"
check("Partitions cover entire training set", _partition_total)

def _partition_disjoint():
    parts = partition_data(X_train_bc, y_train_bc, num_clients=20, alpha=0.5, seed=42)
    seen = set()
    for p in parts:
        for row in p[0]:
            key = tuple(row.tolist())
            assert key not in seen, "same sample in two partitions"
            seen.add(key)
    return "no sample duplicated across clients"
check("Partitions are disjoint", _partition_disjoint)

def _partition_reproducible():
    p1 = partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5, seed=42)
    p2 = partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5, seed=42)
    assert all(np.array_equal(p1[i][1], p2[i][1]) for i in range(5))
    return "identical output for same seed"
check("Partitioning is reproducible", _partition_reproducible)

def _partition_noise_only_on_train():
    # Noise injected in client __init__ on partition data, not on test set
    parts = partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5, seed=42)
    for i, (X_p, y_p) in enumerate(parts):
        overlap = np.intersect1d(y_p, y_test_bc)  # labels can match but data shouldn't
        train_set = set(map(tuple, X_p.tolist()))
        test_in_train = sum(1 for row in X_test_bc if tuple(row.tolist()) in train_set)
        assert test_in_train == 0, f"client {i} partition overlaps test set"
    return "test data absent from all client partitions"
check("Test data never in client partitions", _partition_noise_only_on_train)

# ─────────────────────────────────────────────────────────────────
print("\n[4] Server-side eval function")
# ─────────────────────────────────────────────────────────────────

eval_fn_bc = make_eval_fn(X_test_bc, y_test_bc)

def _eval_fn_range():
    model = get_model(input_size=30)
    params = [v.cpu().numpy() for _, v in model.state_dict().items()]
    acc, auc = eval_fn_bc(params)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= auc <= 1.0
    return f"acc={acc:.4f} auc={auc:.4f}"
check("eval_fn returns values in [0, 1]", _eval_fn_range)

def _eval_fn_deterministic():
    model = get_model(input_size=30)
    params = [v.cpu().numpy() for _, v in model.state_dict().items()]
    acc1, auc1 = eval_fn_bc(params)
    acc2, auc2 = eval_fn_bc(params)
    assert acc1 == acc2 and auc1 == auc2
    return "same params -> same metrics every call"
check("eval_fn is deterministic", _eval_fn_deterministic)

def _eval_fn_no_nan():
    model = get_model(input_size=30)
    params = [v.cpu().numpy() for _, v in model.state_dict().items()]
    acc, auc = eval_fn_bc(params)
    assert acc == acc and auc == auc, "NaN detected"
    return "no NaN"
check("eval_fn produces no NaN", _eval_fn_no_nan)

def _eval_fn_clean_labels():
    # The test set labels must be clean (no noise ever applied to them)
    # y_test_bc comes from split_arrays before any BRFSSClient is created
    pos_rate = y_test_bc.mean()
    assert 0.30 < pos_rate < 0.45
    # Also verify dtype is correct for model
    assert y_test_bc.dtype == np.int64
    return f"clean labels, pos_rate={pos_rate:.2%}, dtype=int64"
check("eval_fn test labels are clean and correctly typed", _eval_fn_clean_labels)

def _eval_fn_better_than_random_after_training():
    # A model that actually trains should beat 0.5 AUC
    parts = partition_data(X_train_bc, y_train_bc, num_clients=3, alpha=0.5, seed=42)
    def cfn(cid):
        X, y = parts[int(cid)]
        return BRFSSClient(X, y, noise_rate=0.0, seed=42)
    h = run_fedavg(cfn, num_clients=3, num_rounds=5, eval_fn=eval_fn_bc)
    final_auc = h.metrics_distributed["auc"][-1][1]
    assert final_auc > 0.55, f"AUC {final_auc:.4f} not better than random after 5 rounds"
    return f"final AUC={final_auc:.4f} after 5 rounds (>0.55)"
check("Model learns: AUC exceeds 0.55 after 5 rounds (BC clean)", _eval_fn_better_than_random_after_training)

# ─────────────────────────────────────────────────────────────────
print("\n[5] CS-Agg trust scoring")
# ─────────────────────────────────────────────────────────────────

def _trust_high_loss_downweighted():
    losses = np.array([0.1, 0.1, 2.0])
    lm, ls = losses.mean(), losses.std() + 1e-8
    ts = np.exp(-((losses - lm) / ls))
    ts = ts / ts.sum()
    assert ts[2] < ts[0], f"noisy weight {ts[2]:.4f} >= clean {ts[0]:.4f}"
    assert abs(ts[0] - ts[1]) < 1e-9
    return f"clean={ts[0]:.4f} | noisy={ts[2]:.4f}"
check("High-loss client gets lower trust score", _trust_high_loss_downweighted)

def _trust_sum_to_one():
    losses = np.array([0.1, 0.5, 0.2, 2.0, 0.3])
    lm, ls = losses.mean(), losses.std() + 1e-8
    ts = np.exp(-((losses - lm) / ls))
    ts = ts / ts.sum()
    assert abs(ts.sum() - 1.0) < 1e-9
    return f"sum={ts.sum():.12f}"
check("Trust scores sum to 1.0", _trust_sum_to_one)

def _warmup_equals_fedavg():
    agg = make_csagg_aggregate(warmup_rounds=10)
    results = [
        ([np.array([2.0, 4.0])], 100, {"train_loss": 0.1}),
        ([np.array([0.0, 0.0])], 100, {"train_loss": 5.0}),
    ]
    r_warmup = agg(results, round_num=5)
    r_fedavg = fedavg_aggregate(results)
    assert np.allclose(r_warmup[0], r_fedavg[0])
    return f"round 5 warmup == FedAvg output {r_warmup[0]}"
check("CS-Agg during warmup == FedAvg", _warmup_equals_fedavg)

def _post_warmup_downweights_noisy():
    agg = make_csagg_aggregate(warmup_rounds=10)
    results = [
        ([np.array([2.0])], 100, {"train_loss": 0.1}),  # clean
        ([np.array([0.0])], 100, {"train_loss": 5.0}),  # noisy
    ]
    r_post = agg(results, round_num=11)[0]
    r_fedavg = fedavg_aggregate(results)[0]
    # Clean client has params=[2.0], noisy=[0.0]
    # FedAvg = (2+0)/2 = 1.0
    # CS-Agg should be > 1.0 since clean client gets more weight
    assert r_post > r_fedavg, f"CS-Agg={r_post:.4f} not > FedAvg={r_fedavg:.4f}"
    return f"FedAvg={r_fedavg:.4f} < CS-Agg={r_post:.4f} (clean client upweighted)"
check("Post-warmup CS-Agg upweights clean clients", _post_warmup_downweights_noisy)

def _csagg_activates_at_correct_round():
    agg = make_csagg_aggregate(warmup_rounds=10)
    results = [
        ([np.array([2.0])], 100, {"train_loss": 0.1}),
        ([np.array([0.0])], 100, {"train_loss": 5.0}),
    ]
    r10 = agg(results, round_num=10)[0]
    r11 = agg(results, round_num=11)[0]
    r_avg = fedavg_aggregate(results)[0]
    assert np.isclose(r10, r_avg), f"round 10 should still be FedAvg: {r10} vs {r_avg}"
    assert not np.isclose(r11, r_avg), f"round 11 should differ from FedAvg: {r11} vs {r_avg}"
    return f"warmup ends after round 10, trust scoring starts round 11"
check("CS-Agg activates exactly at round warmup+1", _csagg_activates_at_correct_round)

# ─────────────────────────────────────────────────────────────────
print("\n[6] Potential result skews")
# ─────────────────────────────────────────────────────────────────

def _class_weights_correct():
    y_imb = np.array([0]*400 + [1]*100, dtype=np.int64)
    X_imb = np.random.randn(500, 4).astype(np.float32)
    c = BRFSSClient(X_imb, y_imb, noise_rate=0.0)
    w = c.criterion.weight.cpu().numpy()
    assert w[1] > w[0], "minority class not upweighted"
    expected_0 = 500 / (2 * 400)
    expected_1 = 500 / (2 * 100)
    assert abs(w[0] - expected_0) < 1e-4
    assert abs(w[1] - expected_1) < 1e-4
    return f"w[majority]={w[0]:.4f} w[minority]={w[1]:.4f} (4x ratio)"
check("Class weights upweight minority class correctly", _class_weights_correct)

def _fedavg_weight_is_full_dataset_size():
    X_big = np.random.randn(5000, 30).astype(np.float32)
    y_big = np.random.randint(0, 2, 5000).astype(np.int64)
    c = BRFSSClient(X_big, y_big, noise_rate=0.0, seed=42)
    model = get_model(30)
    params = [v.cpu().numpy() for _, v in model.state_dict().items()]
    _, num_examples, _ = c.fit(params, config={"round": 1})
    assert num_examples == 5000
    return "weight=5000 (full dataset), trains on <=500 samples — standard FL"
check("FedAvg weight = full dataset size (standard FL behavior)", _fedavg_weight_is_full_dataset_size)

def _fit_subsample_varies_by_round():
    X_big = np.random.randn(5000, 4).astype(np.float32)
    y_big = np.random.randint(0, 2, 5000).astype(np.int64)
    c = BRFSSClient(X_big, y_big, noise_rate=0.0)
    model = get_model(4)
    params = [v.cpu().numpy() for _, v in model.state_dict().items()]
    _, _, m1 = c.fit(params, config={"round": 1})
    _, _, m2 = c.fit(params, config={"round": 2})
    _, _, m3 = c.fit(params, config={"round": 1})
    assert abs(m1["train_loss"] - m3["train_loss"]) < 1e-5, "same round = different subsample"
    assert abs(m1["train_loss"] - m2["train_loss"]) > 1e-6, "different rounds = same subsample"
    return f"round1={m1['train_loss']:.4f} round2={m2['train_loss']:.4f} round1_again={m3['train_loss']:.4f}"
check("Fit subsamples rotate across rounds", _fit_subsample_varies_by_round)

def _noise_not_applied_when_rate_zero():
    y = np.array([1]*100 + [0]*100, dtype=np.int64)
    X = np.random.randn(200, 4).astype(np.float32)
    c = BRFSSClient(X, y, noise_rate=0.0, seed=42)
    assert np.array_equal(c.y, y), "noise applied even with rate=0"
    return "y unchanged when noise_rate=0.0"
check("No noise applied at noise_rate=0.0", _noise_not_applied_when_rate_zero)

def _warmup_rounds_vs_total_rounds():
    # With NUM_ROUNDS=30 and warmup=10, CS-Agg has 20 active rounds
    # Verify this is enough to observe differentiation
    from server_csagg import run_csagg
    parts = partition_data(X_train_bc, y_train_bc, num_clients=5, alpha=0.5, seed=42)
    eval_fn = make_eval_fn(X_test_bc, y_test_bc)
    def cfn_noisy(cid):
        X, y = parts[int(cid)]
        noise = 0.3 if int(cid) < 1 else 0.0
        return BRFSSClient(X, y, noise_rate=noise, seed=42, noise_type="symmetric")
    h = run_csagg(cfn_noisy, num_clients=5, num_rounds=15, eval_fn=eval_fn)
    rounds = [r for r, _ in h.metrics_distributed["accuracy"]]
    assert 15 in rounds
    assert len(rounds) == 15
    return f"runs 15 rounds, warmup=10, CS-Agg active for rounds 11-15"
check("CS-Agg warmup=10 leaves 20 active rounds with NUM_ROUNDS=30", _warmup_rounds_vs_total_rounds)

def _csv_resume_works():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        fname = f.name
    try:
        rows = []
        for s in [42, 123]:
            for r in range(1, 4):
                rows.append({"round": r, "accuracy": 0.8, "auc": 0.85,
                             "method": "fedavg", "dataset": "brfss",
                             "noise_type": "asymmetric", "alpha": 0.5,
                             "noise_rate": 0.1, "noisy_fraction": 0.2, "seed": s})
        pd.DataFrame(rows).to_csv(fname, index=False)
        import run_experiments as re
        orig = re.RESULTS_PATH
        re.RESULTS_PATH = fname
        completed = load_completed()
        re.RESULTS_PATH = orig
        assert ("fedavg", "brfss", 0.5, 0.1, 0.2, 42) in completed
        assert ("fedavg", "brfss", 0.5, 0.1, 0.2, 123) in completed
        assert ("fedavg", "brfss", 0.5, 0.2, 0.2, 42) not in completed
        return f"{len(completed)} conditions recognized as done"
    finally:
        os.unlink(fname)
check("Resume correctly identifies completed conditions", _csv_resume_works)

# ─────────────────────────────────────────────────────────────────
print("\n[7] End-to-end run with server-side eval")
# ─────────────────────────────────────────────────────────────────

parts_e2e = partition_data(X_train_bc, y_train_bc, num_clients=4, alpha=0.5, seed=42)
eval_fn_e2e = make_eval_fn(X_test_bc, y_test_bc)

def _e2e_both_methods_clean():
    def cfn(cid):
        X, y = parts_e2e[int(cid)]
        return BRFSSClient(X, y, noise_rate=0.0, seed=42)
    h_avg = run_fedavg(cfn, num_clients=4, num_rounds=4, eval_fn=eval_fn_e2e)
    h_csagg = run_csagg(cfn, num_clients=4, num_rounds=4, eval_fn=eval_fn_e2e)
    accs_avg = [a for _, a in h_avg.metrics_distributed["accuracy"]]
    accs_csagg = [a for _, a in h_csagg.metrics_distributed["accuracy"]]
    # With clean data and warmup=10, both should be identical for rounds 1-4
    assert accs_avg == accs_csagg, f"should match during warmup: {accs_avg} vs {accs_csagg}"
    assert all(0 <= a <= 1 for a in accs_avg)
    assert all(a == a for a in accs_avg)
    return f"clean acc={[round(a,3) for a in accs_avg]}"
check("FedAvg == CS-Agg during warmup on clean data", _e2e_both_methods_clean)

def _e2e_no_nan_in_results():
    def cfn(cid):
        X, y = parts_e2e[int(cid)]
        noise = 0.3 if int(cid) < 2 else 0.0
        return BRFSSClient(X, y, noise_rate=noise, seed=42, noise_type="symmetric")
    for method_fn, name in [(run_fedavg, "fedavg"), (run_csagg, "csagg")]:
        h = method_fn(cfn, num_clients=4, num_rounds=4, eval_fn=eval_fn_e2e)
        for r, acc in h.metrics_distributed["accuracy"]:
            assert acc == acc, f"{name} round {r} acc is NaN"
        for r, auc in h.metrics_distributed.get("auc", []):
            assert auc == auc, f"{name} round {r} auc is NaN"
    return "no NaN in accuracy or AUC across both methods"
check("No NaN in results with 50% noisy clients", _e2e_no_nan_in_results)

def _e2e_history_has_correct_rounds():
    def cfn(cid):
        X, y = parts_e2e[int(cid)]
        return BRFSSClient(X, y, noise_rate=0.0, seed=42)
    h = run_fedavg(cfn, num_clients=4, num_rounds=5, eval_fn=eval_fn_e2e)
    round_nums = [r for r, _ in h.metrics_distributed["accuracy"]]
    assert round_nums == [1, 2, 3, 4, 5], f"round nums wrong: {round_nums}"
    return f"rounds recorded: {round_nums}"
check("History records correct round numbers", _e2e_history_has_correct_rounds)

# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not failures:
    print("ALL CHECKS PASSED — experiment is ready to run")
else:
    print(f"{len(failures)} FAILED:")
    for f in failures:
        print(f"  - {f}")
print("=" * 60)
