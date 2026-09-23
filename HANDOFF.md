# FL-BRFSS Project — Full Handoff Document
**Author:** Kunal Chevuri (ckchevuri@gmail.com)
**Original date:** May 24, 2026 | **Last updated:** August 12, 2026

> ## ⚠ READ THIS FIRST — CURRENT STATUS (Aug 12, 2026)
>
> **The paper is `overleaf/main.tex` + `overleaf/references.bib`. That is
> the only manuscript source.** Lives in a self-contained `overleaf/` folder
> -- `main.tex`, `references.bib`, a compiled `main.pdf`, and a `figures/`
> subfolder holding all 11 PNGs (fig8-fig11 are the ones actually
> `\includegraphics`'d; fig1-fig7 ride along unused, see below) -- so the
> whole folder drops straight into an Overleaf project with nothing external
> to fetch. `\graphicspath{{figures/}}` is local to the folder; there is no
> `../` dependency on the repo's `results/figures/` anymore. It is a
> complete IEEE conference paper (IEEEtran, two-column) that compiles
> standalone from inside `overleaf/` -- confirmed with a real
> `pdflatex`+`bibtex`+`pdflatex`×2 run: 9 pages excluding references, zero
> undefined `\ref`/`\cite`, all 15 bibliography entries resolved. The old
> `paper_edits/*.tex` fragment files (meant to be pasted
> piecewise into an Overleaf doc this session never had access to) are
> **deleted** -- do not recreate that workflow. `verify_paper_numbers.py`
> (repo root) asserts 181 numbers in the paper against
> `results/experiment_results_merged.csv`; re-run it after any edit to
> either.
>
> **Target venue is IEEE TPS 2026, Round 2, deadline 22 Aug 2026.**
> 10 pages excluding bibliography, two-column IEEE format, **anonymous
> submission** (no names/affiliations/emails in the PDF). The paper was
> submitted to IEEE WF-PST 2026 and **rejected**; this round responds to
> that review.
>
> **What changed since the original May 24 study:**
> - Three Byzantine-robust baselines added: Krum, trimmed mean,
>   coordinate-wise median (`src/server_robust.py`).
> - A **uniform (unweighted) mean control** added — isolates "dropped
>   sample-count weighting" from "used a robust statistic". Not a proposed
>   method; a confound check.
> - Noisy-client-fraction sweep 0.2→0.6 at alpha=0.1.
> - 12-feature BRFSS ablation (`preprocess_expanded`) answering the
>   reviewer objection that 4 of ~350 features was unjustified.
> - Runner rewritten as `src/run_experiments_v2.py` (argparse, resume-safe,
>   shardable) + `run_shards.py`. **Everything now runs on local CPU**, not
>   Colab. Sections 14 and 19 below are obsolete — ignore them.
> - Analysis: `analyze_v2.py`; figures 8-11: `figures_v2.py`.
> - Merged results: `results/experiment_results_merged.csv`.
>
> **Headline result changed.** Coordinate-wise median *beats* CS-Agg on
> BRFSS (0.7828 vs 0.7746, Wilcoxon p<0.0001). The paper is now framed as a
> comparative robustness study, not a CS-Agg advocacy paper. Do not
> reintroduce "CS-Agg is the best method" language anywhere.
>
> **The 149× figure is correct, but only for one specific slice.** It is
> 0.234808 / 0.001576 = 148.9, where the heterogeneity effect is FedAvg
> alpha=1.0→0.1 **averaged over all four noise rates** (0.7877→0.5529) and
> the noise effect is noise 0.0→0.3 at alpha=0.5 (0.78496→0.78338).
> If you instead take the heterogeneity effect at **noise=0.0 only**
> (0.7876→0.5680, effect 0.2196) the ratio is **139×**, not 149×. Quote the
> slice with the number, every time — an early draft of
> `paper_edits/04_results.tex` paired the noise=0.0 numbers with the 149×
> ratio, which is internally inconsistent.
>
> Sections 1, 16, 17, 18c below describe the *original* WF-PST study and are
> kept for provenance only.

---

## 1. WHO YOU ARE AND WHAT YOU'RE DOING (ORIGINAL — SUPERSEDED)

You are a researcher (undergrad/early researcher) building a paper submittable to **IEEE ICHI** or **IEEE BigData**. The research compares three federated learning methods under label noise conditions on two datasets. The goal is to show that trust-weighted aggregation (CS-Agg) meaningfully outperforms baselines (FedAvg, FedProx) when client labels are corrupted — which is realistic in health data (e.g., depression underreporting in BRFSS surveys).

**Original Research Question:**
*How do federated learning methods (FedAvg, CS-Agg, FedProx) perform under varying levels of label noise and data heterogeneity across two healthcare classification tasks — BRFSS depression detection and UCI breast cancer diagnosis?*

**Current Research Question (TPS 2026):**
*Among seven aggregation strategies spanning sample-weighted averaging,
trust weighting, and classical Byzantine-robust order statistics, which
actually defends against the joint effect of extreme non-IID partitioning
and asymmetric label noise on self-reported behavioral health data — and how
much of any apparent advantage is attributable to robustness rather than to
simply not weighting by client sample count?*

---

## 2. PROJECT STRUCTURE

```
fl-brfss/
├── overleaf/                  ← THE PAPER. Self-contained; drop straight into Overleaf.
│   ├── main.tex                   single file, compiles standalone
│   ├── references.bib             all 15 citations, each verified against a
│   │                               publisher/indexer record (not from memory)
│   ├── main.pdf                   last known-good compile (9 pages excl. refs)
│   └── figures/                   all 11 PNGs; fig8-11 are \includegraphics'd,
│                                   fig1-7 (3-method study) ride along unused
├── verify_paper_numbers.py    ← 181 assertions: overleaf/main.tex numbers vs. the CSV
├── smoke_test.py              ← pre-run validation
├── validate.py, run_tests.py  ← correctness checks on the pipeline
├── run_shards.py              ← batched shard launcher (memory-capped concurrency)
├── analyze_v2.py              ← merge shards + produce every table in the paper
├── figures.py                 ← figures 1-7 (original 3-method study; NOT used in main.tex)
├── figures_v2.py              ← figures 8-11 (used in main.tex)
├── data/
│   └── LLCP2023.XPT           ← BRFSS 2023 raw data (not redistributed)
├── results/
│   ├── experiment_results.csv        ← original 216-condition grid
│   ├── v2_shards/*.csv               ← per-shard output from run_experiments_v2
│   ├── experiment_results_merged.csv ← analyze_v2.py output; USE THIS ONE
│   └── figures/
└── src/
    ├── data.py                ← preprocess() 4-feature, preprocess_expanded() 12-feature,
    │                            Dirichlet partitioning. NOTE: BRFSS is NOT standardized.
    ├── model.py               ← BinaryMLP; 2,466 (d=4) / 2,978 (d=12) / 4,130 (d=30) params
    ├── noise.py               ← asymmetric + symmetric label noise injection
    ├── client.py              ← BRFSSClient + FedProxClient (500-sample/round train cap)
    ├── simulate.py            ← core simulation loop + fedavg_aggregate
    ├── server_fedavg.py       ← run_fedavg()
    ├── server_csagg.py        ← run_csagg() + CS-Agg cosine similarity aggregation
    ├── server_fedprox.py      ← run_fedprox()
    ├── server_robust.py       ← Krum, trimmed mean, coordinate median, uniform mean
    ├── run_experiments.py     ← original main loop (kept; make_eval_fn lives here)
    └── run_experiments_v2.py  ← CURRENT runner: argparse, resume-safe, shardable
```

**Deleted Aug 10, 2026** (stale/superseded, do not resurrect):
`analyze.py` (subsumed by `analyze_v2.py`), `colab_experiment.ipynb` (used the
retired "FedNoRo" name and a Colab-only workflow), `experiment_results - Copy.csv`
(a truncated 144-condition partial backup of the 216-condition run).

**Deleted Aug 12, 2026:** `paper_edits/` (six numbered `.tex` fragments + a
`00_README.md`, written when the plan was to copy-paste into an Overleaf
project this session never had access to). Superseded by `overleaf/main.tex`,
which contains the same content merged into one compiling document, with the
"insert after X" comments and paste instructions removed since there is
nothing left to paste into. `overleaf/` was later moved to its own folder
(Aug 12) and made fully self-contained -- `main.tex`, `references.bib`,
`main.pdf`, and a `figures/` subfolder with all 11 PNGs copied in and
`\graphicspath` pointed locally -- so it can be dropped straight into an
actual Overleaf project with no external dependency.

---

## 3. DATASETS

### Dataset 1: BRFSS 2023 (Behavioral Risk Factor Surveillance System)
- **Source:** CDC, file `LLCP2023.XPT` (SAS transport format, ~400MB)
- **Task:** Binary classification — depression diagnosis (ADDEPEV3)
- **Features (5 total):**
  - `MENTHLTH` — days of poor mental health in past 30 days (normalized to /30)
  - `EXERANY2` — any physical activity in past 30 days (binary 0/1)
  - `GENHLTH` — general health self-rating (1-5 scale)
  - `_RFBMI5` — BMI category (overweight/not)
  - `ADDEPEV3` — target: ever told you have depression (1=yes, 0=no)
- **After cleaning:** ~430k+ rows
- **Class imbalance:** ~20.74% depression positive (79.26% negative)
- **Train/test split:** 80/20, stratified, random_state=42
- **Noise type used in experiments:** ASYMMETRIC (see Section 6)
- **Why asymmetric:** Mimics real health underreporting — depressed people more likely to underreport than non-depressed people to over-report
- **IMPORTANT:** This file must be manually uploaded to Colab each run — it cannot be loaded from sklearn like Breast Cancer can

### Dataset 2: UCI Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Task:** Binary classification — malignant vs benign tumor
- **Features:** 30 features (mean, SE, worst of 10 nucleus measurements)
- **Preprocessing:** StandardScaler applied (zero mean, unit variance)
- **Label encoding:** 1=malignant, 0=benign (sklearn default is reversed, so `y = 1 - data.target`)
- **Total samples:** 569
- **Train/test split:** 80/20, stratified, random_state=42
- **Noise type used in experiments:** SYMMETRIC (see Section 6)
- **Why symmetric:** Standard benchmark assumption; no domain-specific directionality

---

## 4. MODEL ARCHITECTURE

Both datasets use the same **BinaryMLP** class from `src/model.py`:
```
Input (4 or 30 features)
→ Linear(input_size, 64) → ReLU
→ Linear(64, 32) → ReLU
→ Linear(32, 2)  ← 2-class output (binary classification via CrossEntropyLoss)
```

Output is raw logits. Softmax applied during evaluation to get probabilities for AUC computation.

**Why CrossEntropyLoss instead of BCELoss:** Using 2-class output + CrossEntropyLoss is equivalent to BCE but handles the BRFSS class imbalance via `weight` argument more cleanly.

**Class weighting (BRFSS only):**
```python
weight = [total / (2 * neg_count), total / (2 * pos_count)]
```
This upweights the minority class (depressed=1) to prevent the model from predicting all-zeros. Applied in `BRFSSClient.__init__()`.

---

## 5. FEDERATED LEARNING SETUP

- **Framework:** Flower (flwr) installed but NOT used for actual simulation. Custom sequential simulator in `src/simulate.py` (no Ray, no multiprocessing, no network — runs everything in one Python process)
- **Why custom simulator:** Flower's default simulation uses Ray and is slow to initialize on Colab; the custom loop is ~5x faster and avoids all Ray/memory issues
- **Num clients:** 20
- **Num rounds:** 50
- **Fraction fit:** 1.0 (all 20 clients participate every round)
- **Client training:** 3 local epochs per round, batch size 32, Adam lr=0.001
- **Training subsample cap:** MAX_SAMPLES=500 per client per round (subsampled if client has >500 samples). Subsample uses `np.random.default_rng(seed + round_num)` so different samples each round
- **Evaluation subsample cap:** MAX_EVAL_SAMPLES=2000 per client (subsampled if >2000). Fixed seed per client.

---

## 6. NOISE INJECTION

### Asymmetric Noise (BRFSS only) — `noise.inject_noise()`
```
depressed (1) → not-depressed (0)  at rate = noise_rate
not-depressed (0) → depressed (1)  at rate = noise_rate / 2
```
- Models underreporting: people WITH depression are more likely to deny it than healthy people are to falsely claim it
- Applied once in `BRFSSClient.__init__()` to the client's local labels

### Symmetric Noise (Breast Cancer only) — `noise.inject_symmetric_noise()`
```
both classes flipped at the same rate = noise_rate
```
- Standard benchmark noise model
- Applied once in `BRFSSClient.__init__()` to the client's local labels

### Which clients get noise:
- `noisy_fraction=0.2` means 20% of clients (the first 4 out of 20) get noisy labels
- Clients 0-3: `noise_rate = noise_rate` (e.g., 0.1, 0.2, or 0.3)
- Clients 4-19: `noise_rate = 0.0` (clean labels)
- Noise injected deterministically using `seed` parameter for reproducibility

---

## 7. DATA HETEROGENEITY (NON-IID)

Dirichlet partitioning (`partition_data()` in `data.py`):
- **Alpha (concentration parameter):** Controls how non-IID the data is
  - alpha=0.1 → highly heterogeneous (each client has mostly one class)
  - alpha=0.5 → moderate heterogeneity
  - alpha=1.0 → close to IID (balanced across clients)
- Partitioning is done separately for each class (stratified Dirichlet), ensuring class distribution varies per client
- Uses `np.random.seed(seed)` for reproducibility

---

## 8. FEDERATED LEARNING METHODS

### FedAvg (McMahan et al., 2017) — Baseline
- **Server:** Weighted average of client updates by number of samples
- **Client:** Standard SGD/Adam local training
- **Files:** `src/server_fedavg.py`, `BRFSSClient` in `src/client.py`
- **Citation:** McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017

### CS-Agg (Cosine Similarity Trust Aggregation) — Noise-Robust Method
- **IMPORTANT:** This is NOT the Wu et al. 2023 FedNoRo algorithm. The real FedNoRo uses GMM-based noisy client detection + knowledge distillation. This implementation uses pairwise cosine similarity on update vectors — a simpler trust aggregation approach inspired by the noise-robust FL literature. The method is called "CS-Agg" consistently in all code, CSV, and paper. Cite Wu et al. as motivation, but do not claim to replicate their full method.
- **Server:** Trust-weighted aggregation based on cosine similarity between client update vectors
- **Client:** Same as FedAvg (BRFSSClient, no client-side changes)
- **Algorithm:**
  1. Warmup: rounds 1-10 use standard FedAvg aggregation
  2. After warmup: compute update vector for each client = `local_params - global_params_before_round`
  3. Flatten all update vectors into 1D
  4. For each client i: compute mean cosine similarity with all other clients j≠i
  5. Trust scores = mean cosine similarities, shifted to non-negative, normalized to sum=1
  6. Aggregate: `global_params = sum(trust_score[i] * client_params[i])`
  7. Noisy clients produce updates misaligned with majority → lower trust scores → downweighted
- **Key implementation detail:** `prev_params` (global params BEFORE the round's training) are passed from `simulate()` to the aggregate function. This is critical — using params AFTER training would give wrong update vectors
- **Files:** `src/server_csagg.py`
- **Related work:** Conceptually similar to FLTrust (Cao et al. 2020) but without a server-side root dataset. Simpler than Wu et al. 2023 FedNoRo (no GMM, no knowledge distillation).

### FedProx (Li et al., 2020) — Heterogeneity-Robust Method
- **Server:** Same as FedAvg (standard weighted average)
- **Client:** Adds proximal regularization term to local loss:
  ```
  loss_total = CrossEntropyLoss + (mu/2) * sum((w_local - w_global)^2)
  ```
  Where `w_global` = global model weights received at start of round
- **Mu value:** 0.01 (standard value from Li et al. paper)
- **Purpose:** Prevents local models from drifting too far from global model in heterogeneous settings
- **Files:** `src/server_fedprox.py`, `FedProxClient` in `src/client.py`
- **Citation:** Li et al., "FedProx: Federated Optimization in Heterogeneous Networks," MLSys 2020

---

## 9. EXPERIMENT PARAMETERS (FULL GRID)

```python
NUM_CLIENTS    = 20
NUM_ROUNDS     = 50
ALPHAS         = [0.1, 0.5, 1.0]        # data heterogeneity
NOISE_RATES    = [0.0, 0.1, 0.2, 0.3]   # label noise rate
NOISY_FRACTIONS = [0.2]                  # 20% of clients are noisy
SEEDS          = [42, 123, 456]          # 3 seeds for stronger statistical power
METHODS        = ["fedavg", "csagg", "fedprox"]
DATASETS       = ["brfss", "breast_cancer"]
FEDPROX_MU     = 0.01
```

**Total conditions:** 3 methods × 2 datasets × 3 alphas × 4 noise rates × 1 noisy fraction × 3 seeds = **216 conditions**

**Total rows in results CSV:** 216 conditions × 50 rounds = **10,800 rows**

**Estimated runtime on A100:** ~13-15 hours total across 2 sessions

---

## 10. EVALUATION STRATEGY

### Primary metric: AUC-ROC
- Accuracy is NOT the primary metric for BRFSS because a model predicting all-0s gets 79.26% accuracy (majority class baseline)
- AUC-ROC is threshold-independent and handles class imbalance correctly
- AUC values are always stored as `float("nan")` if `roc_auc_score` throws ValueError (only 1 class in batch)

### Server-side evaluation on clean held-out test set
- The `eval_fn` passed to each server is built from the HELD-OUT TEST SET (not any client's data)
- Test set is clean — NO noise injected
- `make_eval_fn(X_test, y_test)` in `run_experiments.py` builds a closure that:
  1. Loads global params into a fresh model
  2. Runs inference on the full clean test set
  3. Returns (accuracy, auc)
- This ensures metrics are uncontaminated by client noise
- BRFSS test set: ~86k rows
- Breast Cancer test set: ~114 rows

### Why NOT evaluate on client data:
- Noisy clients' labels are corrupted → accuracy/AUC on their data would be misleading
- CS-Agg's trust mechanism already accounts for noisy clients on the server side
- Clean server-side eval is the gold standard for FL research papers

---

## 11. RESULTS FILE FORMAT

**Path:** `results/experiment_results.csv`

**Columns:**
```
round, accuracy, auc, method, dataset, noise_type, alpha, noise_rate, noisy_fraction, seed
```

**Sample row:**
```
5, 0.7923, 0.8341, fedavg, brfss, asymmetric, 0.5, 0.2, 0.2, 42
```

**Resume logic:**
- If CSV already exists and has rows, `load_completed()` reads it and returns a set of `(method, dataset, alpha, noise_rate, noisy_fraction, seed)` tuples
- Each condition is checked against this set; if already present → skipped with `[SKIP]` message
- After each condition completes → results appended to CSV immediately (so a crash only loses the in-progress condition, not all previous work)
- If Colab disconnects at hour 5 → re-run the notebook → experiment picks up where it left off

---

## 12. BUGS FOUND AND FIXED (CRITICAL — DO NOT REVERT)

All of these were found and fixed. The current code is correct. This list explains WHY things are written the way they are.

### Bug 1: Evaluation on noisy client data (MAJOR)
- **Old behavior:** `eval_fn=None` → each client evaluated on its own (possibly noisy) data → federated average of client metrics reported as the result
- **Fix:** Pass `eval_fn` built from clean held-out test set to all servers. `simulate()` checks `if eval_fn is not None` and uses it exclusively.
- **Impact:** Without this, CS-Agg's metrics were inflated by evaluating on the same noisy data it was trained to detect. Results were meaningless.

### Bug 2: CS-Agg old code used training loss as trust proxy (MAJOR)
- **Old behavior:** Trust score = inverse of client training loss. Noisy clients tend to have higher loss → lower trust. But this is fragile and produces inconsistent results.
- **Fix:** Proper cosine similarity between full update vectors (flattened parameter diffs). This is exactly what Wu et al. 2023 describes.
- **Impact:** Old method would confound heterogeneity with noise (high loss could mean heterogeneous data, not just noise). New method is faithful to the paper.

### Bug 3: CS-Agg aggregate received wrong prev_params (MAJOR)
- **Old behavior:** `prev_params` was not passed from `simulate()` to `aggregate_fn`. CS-Agg couldn't compute update vectors.
- **Fix:** `simulate()` saves `prev_params = global_params` before calling `client.fit()`, then passes it to `aggregate_fn(results, round_num=round_num, prev_params=prev_params)`. `fedavg_aggregate` also accepts `prev_params=None` and ignores it.
- **Impact:** Without this, CS-Agg would always fall back to FedAvg behavior (it checks `if prev_params is None`).

### Bug 4: No evaluation subsample cap
- **Old behavior:** `client.evaluate()` ran on the full client dataset (could be 50k+ rows on BRFSS)
- **Fix:** `MAX_EVAL_SAMPLES=2000` cap in `BRFSSClient.evaluate()`
- **Impact:** Without this, each round's evaluation would take 10-20 minutes for BRFSS. Total runtime would exceed 100+ hours.

### Bug 5: Training subsample used same 500 samples every round
- **Old behavior:** `seed` was fixed → same 500 samples every round → model memorizes the same points
- **Fix:** Subsample uses `np.random.default_rng(self.seed + round_num)` so samples vary each round
- **Impact:** Without this, training was effectively doing many epochs on 500 fixed samples → severe overfitting, non-representative gradient updates

### Bug 6: No incremental saving / no resume logic
- **Old behavior:** CSV written only at the end of all 144 conditions. Any crash = 0 results saved.
- **Fix:** Append to CSV after each condition. `load_completed()` at startup to skip done conditions.
- **Impact:** Critical for a 9-10 hour run on Colab which can disconnect at any time.

### Bug 7: NaN AUC propagated silently
- **Old behavior:** `m.get("auc", 0.5)` returns NaN if the stored value IS NaN (not missing). NaN * n = NaN, which propagates through the weighted average.
- **Fix:** Explicit NaN check: `v = m.get("auc", 0.5); return v if (v == v) else 0.5`
- **Impact:** Without this, entire round's AUC could be NaN if even one client had a degenerate batch.

### Bug 8: torch RNG not seeded per condition
- **Old behavior:** `np.random.seed(seed)` called but not `torch.manual_seed(seed)`. PyTorch weight initialization was non-deterministic.
- **Fix:** `torch.manual_seed(seed)` called at the start of `run_condition()`.
- **Impact:** Running the same condition twice gave different results. Results not reproducible.

### Bug 9: FedAvg aggregate signature didn't accept prev_params
- **Old behavior:** `fedavg_aggregate(results)` — calling it with `prev_params=` kwarg would crash
- **Fix:** `fedavg_aggregate(results, round_num=None, prev_params=None)` — accepts and ignores the extra args
- **Impact:** `simulate()` passes `prev_params` to all aggregate functions uniformly. Without this, FedAvg and FedProx would crash.

---

## 13. COMPLETE FILE CONTENTS (FINAL VERSIONS)

### src/noise.py
```python
import numpy as np

def inject_noise(y, noise_rate, seed=42):
    """Asymmetric: 1→0 at rate, 0→1 at rate/2"""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    dep_indices = np.where(y == 1)[0]
    n_flip = int(len(dep_indices) * noise_rate)
    flip_indices = rng.choice(dep_indices, size=n_flip, replace=False)
    y_noisy[flip_indices] = 0
    nodep_indices = np.where(y == 0)[0]
    n_flip = int(len(nodep_indices) * (noise_rate / 2))
    flip_indices = rng.choice(nodep_indices, size=n_flip, replace=False)
    y_noisy[flip_indices] = 1
    return y_noisy

def inject_symmetric_noise(y, noise_rate, seed=42):
    """Symmetric: both classes flipped at same rate"""
    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    for label in [0, 1]:
        indices = np.where(y == label)[0]
        n_flip = int(len(indices) * noise_rate)
        if n_flip > 0:
            flip_idx = rng.choice(indices, size=n_flip, replace=False)
            y_noisy[flip_idx] = 1 - label
    return y_noisy
```

### src/model.py
```python
import torch
import torch.nn as nn

class BinaryMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

def get_model(input_size=4):
    return BinaryMLP(input_size)
```

### src/data.py
```python
import pyreadstat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_brfss(path="data/LLCP2023.XPT"):
    df, meta = pyreadstat.read_xport(path, encoding="latin1")
    return df

def preprocess(df):
    cols = ["MENTHLTH", "EXERANY2", "GENHLTH", "_RFBMI5", "ADDEPEV3"]
    df = df[cols].copy()
    df["MENTHLTH"] = df["MENTHLTH"].replace({77: np.nan, 99: np.nan, 88: 0})
    df["EXERANY2"] = df["EXERANY2"].replace({7: np.nan, 9: np.nan})
    df["GENHLTH"]  = df["GENHLTH"].replace({7: np.nan, 9: np.nan})
    df["_RFBMI5"]  = df["_RFBMI5"].replace({9: np.nan})
    df["ADDEPEV3"] = df["ADDEPEV3"].replace({7: np.nan, 9: np.nan})
    df = df.dropna()
    df["ADDEPEV3"] = df["ADDEPEV3"].apply(lambda x: 1 if x == 1 else 0)
    df["EXERANY2"] = df["EXERANY2"].apply(lambda x: 1 if x == 1 else 0)
    df["MENTHLTH"] = df["MENTHLTH"] / 30.0
    return df

def split_data(df):
    X = df.drop("ADDEPEV3", axis=1).values.astype(np.float32)
    y = df["ADDEPEV3"].values.astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def partition_data(X_train, y_train, num_clients=20, alpha=0.5, seed=42):
    np.random.seed(seed)
    client_data = [[] for _ in range(num_clients)]
    for c in range(2):
        class_indices = np.where(y_train == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        splits = np.split(class_indices, np.cumsum(proportions[:-1]))
        for i, split in enumerate(splits):
            client_data[i].extend(split.tolist())
    return [(X_train[client_data[i]], y_train[client_data[i]]) for i in range(num_clients)]

def load_breast_cancer_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data).astype(np.float32)
    y = (1 - data.target).astype(np.int64)  # 1=malignant, 0=benign
    return X, y

def split_arrays(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
```

### src/simulate.py
```python
import numpy as np

class History:
    def __init__(self):
        self.metrics_distributed = {"accuracy": []}
        self.losses_distributed = []

def fedavg_aggregate(results, round_num=None, prev_params=None):
    results = [(p, n, m) for p, n, m in results if n > 0]
    total_examples = sum(n for _, n, _ in results)
    aggregated = None
    for params, num_examples, _ in results:
        w = num_examples / total_examples
        if aggregated is None:
            aggregated = [w * p for p in params]
        else:
            aggregated = [agg + w * p for agg, p in zip(aggregated, params)]
    return aggregated

def simulate(client_fn, num_clients, num_rounds, aggregate_fn=None, eval_fn=None):
    if aggregate_fn is None:
        aggregate_fn = fedavg_aggregate
    clients = [client_fn(str(i)) for i in range(num_clients)]
    global_params = clients[0].get_parameters(config={})
    history = History()
    for round_num in range(1, num_rounds + 1):
        prev_params = global_params  # saved before training for CS-Agg update vectors
        fit_results = []
        for client in clients:
            params, num_examples, metrics = client.fit(global_params, config={"round": round_num})
            fit_results.append((params, num_examples, metrics))
        global_params = aggregate_fn(fit_results, round_num=round_num, prev_params=prev_params)
        if eval_fn is not None:
            acc, auc = eval_fn(global_params)
        else:
            eval_results = []
            for client in clients:
                loss, num_examples, metrics = client.evaluate(global_params, config={})
                eval_results.append((loss, num_examples, metrics))
            eval_results = [(l, n, m) for l, n, m in eval_results if n > 0]
            total = sum(n for _, n, _ in eval_results)
            acc = sum(m["accuracy"] * n for _, n, m in eval_results) / total
            def _safe_auc(m):
                v = m.get("auc", 0.5)
                return v if (v == v) else 0.5
            auc = sum(_safe_auc(m) * n for _, n, m in eval_results) / total
        history.metrics_distributed["accuracy"].append((round_num, acc))
        history.metrics_distributed.setdefault("auc", []).append((round_num, auc))
        print(f"Round {round_num:>3}/{num_rounds} | accuracy: {acc:.4f} | auc: {auc:.4f}")
    return history
```

### src/client.py
```python
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from model import get_model
from noise import inject_noise, inject_symmetric_noise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BRFSSClient(fl.client.NumPyClient):
    def __init__(self, X, y, noise_rate=0.0, seed=42, noise_type="asymmetric"):
        self.X = X
        self.y = y
        self.noise_rate = noise_rate
        self.seed = seed
        input_size = X.shape[1] if len(X) > 0 else 4
        self.model = get_model(input_size=input_size).to(DEVICE)
        if noise_rate > 0:
            if noise_type == "symmetric":
                self.y = inject_symmetric_noise(y, noise_rate, seed=seed)
            else:
                self.y = inject_noise(y, noise_rate, seed=seed)
        if len(self.y) > 0:
            neg = max((self.y == 0).sum(), 1)
            pos = max((self.y == 1).sum(), 1)
            total = len(self.y)
            weight = torch.tensor([total / (2 * neg), total / (2 * pos)], dtype=torch.float32).to(DEVICE)
        else:
            weight = torch.tensor([1.0, 1.0]).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        if len(self.X) == 0:
            return self.get_parameters(config={}), 0, {"train_loss": 0.0}
        self.set_parameters(parameters)
        MAX_SAMPLES = 500
        X, y = self.X, self.y
        if len(X) > MAX_SAMPLES:
            round_num = config.get("round", 0)
            rng = np.random.default_rng(self.seed + round_num)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        total_loss, total_samples = 0.0, 0
        for epoch in range(3):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)
        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return self.get_parameters(config={}), len(self.X), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        if len(self.X) == 0:
            return 0.0, 0, {"accuracy": 0.0, "auc": 0.5}
        self.set_parameters(parameters)
        MAX_EVAL_SAMPLES = 2000
        X, y = self.X, self.y
        if len(X) > MAX_EVAL_SAMPLES:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), MAX_EVAL_SAMPLES, replace=False)
            X, y = X[idx], y[idx]
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=256)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                output = self.model(X_batch)
                total_loss += self.criterion(output, y_batch).item() * len(y_batch)
                correct += (output.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
                probs = F.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_probs.append(probs)
                all_labels.append(y_batch.cpu().numpy())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        return float(total_loss / total), total, {"accuracy": correct / total, "auc": float(auc)}


class FedProxClient(BRFSSClient):
    """FedProx: adds proximal term (mu/2)||w - w_global||^2 to local loss."""
    def __init__(self, X, y, noise_rate=0.0, seed=42, noise_type="asymmetric", mu=0.01):
        super().__init__(X, y, noise_rate, seed, noise_type)
        self.mu = mu

    def fit(self, parameters, config):
        if len(self.X) == 0:
            return self.get_parameters(config={}), 0, {"train_loss": 0.0}
        self.set_parameters(parameters)
        global_weights = [p.clone().detach() for p in self.model.parameters()]
        MAX_SAMPLES = 500
        X, y = self.X, self.y
        if len(X) > MAX_SAMPLES:
            round_num = config.get("round", 0)
            rng = np.random.default_rng(self.seed + round_num)
            idx = rng.choice(len(X), MAX_SAMPLES, replace=False)
            X, y = X[idx], y[idx]
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.int64).to(DEVICE)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        total_loss, total_samples = 0.0, 0
        for _ in range(3):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                prox = sum(((p - g) ** 2).sum() for p, g in zip(self.model.parameters(), global_weights))
                loss = loss + (self.mu / 2) * prox
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)
        train_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return self.get_parameters(config={}), len(self.X), {"train_loss": train_loss}
```

### src/server_fedavg.py
```python
from simulate import simulate, fedavg_aggregate

def run_fedavg(client_fn, num_clients=20, num_rounds=100, eval_fn=None):
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=fedavg_aggregate, eval_fn=eval_fn)
```

### src/server_fedprox.py
```python
from simulate import simulate, fedavg_aggregate

def run_fedprox(client_fn, num_clients=20, num_rounds=100, eval_fn=None):
    # Proximal term is in FedProxClient.fit(), server uses standard FedAvg aggregation
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=fedavg_aggregate, eval_fn=eval_fn)
```

### src/server_csagg.py
```python
import numpy as np
from simulate import simulate, fedavg_aggregate

def make_csagg_aggregate(warmup_rounds=10):
    def aggregate(results, round_num=None, prev_params=None):
        results = [(p, n, m) for p, n, m in results if n > 0]
        if round_num is None or round_num <= warmup_rounds or prev_params is None:
            return fedavg_aggregate(results)
        # Compute update vectors: local_params - global_params_before_round
        updates = []
        for params, _, _ in results:
            update = np.concatenate([(p - g).flatten() for p, g in zip(params, prev_params)])
            updates.append(update)
        # Trust score = mean cosine similarity with all other clients
        n = len(updates)
        trust_scores = np.zeros(n)
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                norm_i = np.linalg.norm(updates[i])
                norm_j = np.linalg.norm(updates[j])
                if norm_i > 1e-10 and norm_j > 1e-10:
                    cos_sim = np.dot(updates[i], updates[j]) / (norm_i * norm_j)
                else:
                    cos_sim = 0.0
                sims.append(cos_sim)
            trust_scores[i] = np.mean(sims) if sims else 0.0
        # Shift non-negative and normalize
        trust_scores = trust_scores - trust_scores.min() + 1e-8
        trust_scores = trust_scores / trust_scores.sum()
        aggregated = None
        for i, (params, _, _) in enumerate(results):
            if aggregated is None:
                aggregated = [trust_scores[i] * p for p in params]
            else:
                aggregated = [agg + trust_scores[i] * p for agg, p in zip(aggregated, params)]
        return aggregated
    return aggregate

def run_csagg(client_fn, num_clients=20, num_rounds=100, warmup_rounds=10, eval_fn=None):
    aggregate_fn = make_csagg_aggregate(warmup_rounds=warmup_rounds)
    return simulate(client_fn, num_clients, num_rounds,
                    aggregate_fn=aggregate_fn, eval_fn=eval_fn)
```

---

## 14. COLAB SETUP — EXACT STEPS

### Runtime
- Runtime → Change runtime type → **A100 GPU** (requires Colab Pro)
- Do NOT use T4 — too slow for 144 conditions

### Cell 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Install dependencies
```python
!pip install flwr pyreadstat scikit-learn torch --quiet
```

### Cell 3: Set up paths
```python
import sys, os
BASE = '/content/drive/MyDrive/fl-brfss'
sys.path.insert(0, os.path.join(BASE, 'src'))
os.chdir(BASE)
```

### Cell 4: Upload BRFSS data file
```python
# BRFSS data file must be uploaded manually to Colab
# (it's too large for Drive sync to be reliable)
# Option A: Upload from local machine
from google.colab import files
uploaded = files.upload()  # select LLCP2023.XPT
os.makedirs('data', exist_ok=True)
import shutil
shutil.move(list(uploaded.keys())[0], 'data/LLCP2023.XPT')
print("BRFSS data file ready")
```

### Cell 5: Run smoke test
```python
%run smoke_test.py
# Must see "ALL OK — safe to run full experiment"
# Takes ~3-5 minutes
# BRFSS test will pass because we just uploaded the XPT file
```

### Cell 6: Run full experiment
```python
%run src/run_experiments.py
# Takes ~9-10 hours
# Saves incrementally to results/experiment_results.csv
# Can be interrupted and resumed — will skip completed conditions
# Watch for: "Round X/50 | accuracy: ... | auc: ..."
```

### Cell 7: Download results
```python
from google.colab import files
files.download('results/experiment_results.csv')
```

---

## 15. POST-EXPERIMENT ANALYSIS PLAN (TIER 2)

After the experiment, run **Wilcoxon signed-rank test** to establish statistical significance.

### What to compare:
- CS-Agg vs FedAvg: paired by (dataset, alpha, noise_rate, noisy_fraction, seed)
- CS-Agg vs FedProx: same pairing
- Use the **final-round AUC** (round 50) for each condition as the test statistic
- Null hypothesis: no difference in AUC between methods

### Code to run locally after downloading CSV:
```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('results/experiment_results.csv')

# Get final-round AUC per condition
final = df[df['round'] == 50].copy()

# Compare CS-Agg vs FedAvg
keys = ['dataset', 'alpha', 'noise_rate', 'noisy_fraction', 'seed']
csagg  = final[final['method']=='csagg'].set_index(keys)['auc']
fedavg = final[final['method']=='fedavg'].set_index(keys)['auc']

# Align on same conditions
common = csagg.index.intersection(fedavg.index)
stat, p = stats.wilcoxon(csagg[common], fedavg[common])
print(f"CS-Agg vs FedAvg: W={stat:.1f}, p={p:.4f}")
# p < 0.05 = statistically significant

# Compare CS-Agg vs FedProx
fedprox = final[final['method']=='fedprox'].set_index(keys)['auc']
common2 = csagg.index.intersection(fedprox.index)
stat2, p2 = stats.wilcoxon(csagg[common2], fedprox[common2])
print(f"CS-Agg vs FedProx: W={stat2:.1f}, p={p2:.4f}")
```

### What results to expect (hypothesis):
- At noise_rate=0.0: all methods should perform similarly (control condition)
- At noise_rate=0.1-0.3: CS-Agg should outperform FedAvg and FedProx (its trust mechanism helps)
- At alpha=0.1 (high heterogeneity): FedProx might outperform FedAvg even without noise
- BRFSS is harder (more samples, more complex task) than Breast Cancer
- AUC should be the primary metric throughout (not accuracy)

### Visualizations to generate for paper:
1. **Line plot:** AUC vs round (final 10 rounds) for each method, faceted by noise rate — shows convergence
2. **Bar chart:** Final AUC by method × noise rate — shows main comparison
3. **Heatmap:** Method × alpha × noise rate showing average AUC — shows interaction effects
4. **Box plot:** AUC distribution across seeds per method (2 seeds = 2 points, but shows variance)

---

## 16. EXPECTED PAPER STRUCTURE (IEEE ICHI/BigData)

**Title:** "Noise-Robust Federated Learning for Healthcare Classification: A Comparative Study of FedAvg, FedProx, and Cosine-Similarity Trust Aggregation (CS-Agg)"

**Sections:**
1. Abstract (150 words)
2. Introduction — motivation, BRFSS underreporting problem, FL for health data privacy
3. Related Work — FedAvg, FedProx, noise-robust FL (cite Wu et al. 2023 FedNoRo as motivation for CS-Agg), label noise in FL
4. Methodology — datasets, model, noise model, experimental setup, evaluation
5. Results — tables + figures comparing methods
6. Discussion — when does CS-Agg help? When does it not? Limitations vs the real FedNoRo (Wu et al. 2023)
7. Conclusion

**Target venues (in order of preference):**
1. IEEE ICHI 2026 (International Conference on Healthcare Informatics) — health focus is perfect
2. IEEE BigData 2026 — broader audience, still relevant
3. FLTA (Federated Learning workshop at NeurIPS/ICLR) — if short paper is OK

---

## 17. TIMELINE

- **May 24 (today, Sunday):** Code finalized. Drive updated. Start Colab A100 run (~9-10 hours).
- **May 25 (Monday):** Experiment finishes. Download CSV. Run Wilcoxon tests. Start analysis and figures.
- **May 26-27 (Tue-Wed):** Write paper. Generate all figures. First draft.
- **Target deadline:** ~May 27 (Tuesday hard deadline Kunal mentioned)

---

## 18. EXPERIMENT RESULTS (FINAL — May 24, 2026)

**The experiment is DONE. 216 conditions completed (seeds 42, 123, 456). Results file: `results/experiment_results.csv`**

- 10,800 rows total (216 conditions × 50 rounds)
- Seeds: [42, 123, 456] — verified
- Zero NaN values — verified
- All 216 conditions have exactly 50 rounds — verified

### Final AUC (Round 50) — Mean ± Std (3 seeds)

| Method  | BRFSS          | Breast Cancer  |
|---------|----------------|----------------|
| FedAvg  | 0.7081 ± 0.1439| 0.9970 ± 0.0012|
| CS-Agg  | 0.7746 ± 0.0237| 0.9975 ± 0.0007|
| FedProx | 0.7286 ± 0.1079| 0.9975 ± 0.0009|

### AUC by Noise Rate (Final Round, averaged across alpha and seeds)

| Dataset       | Noise Rate | FedAvg | CS-Agg  | FedProx |
|---------------|------------|--------|---------|---------|
| breast_cancer | 0.0        | 0.9970 | 0.9975  | 0.9975  |
| breast_cancer | 0.1        | 0.9972 | 0.9976  | 0.9978  |
| breast_cancer | 0.2        | 0.9970 | 0.9975  | 0.9975  |
| breast_cancer | 0.3        | 0.9969 | 0.9974  | 0.9972  |
| brfss         | 0.0        | 0.7135 | 0.7802  | 0.7341  |
| brfss         | 0.1        | 0.7015 | 0.7741  | 0.7263  |
| brfss         | 0.2        | 0.7083 | 0.7723  | 0.7279  |
| brfss         | 0.3        | 0.7090 | 0.7719  | 0.7260  |

### Wilcoxon Signed-Rank Tests (AUC, Final Round, n=36 per method per dataset)

**BRFSS:**
- FedAvg vs CS-Agg: p=0.0000 *** — CS-Agg wins (0.7081 vs 0.7746, +6.6pp)
- FedAvg vs FedProx: p=0.0003 *** — FedProx wins (0.7081 vs 0.7286, +2.1pp)
- CS-Agg vs FedProx: p=0.0001 *** — CS-Agg wins (0.7746 vs 0.7286, +4.6pp)

**Breast Cancer:**
- FedAvg vs CS-Agg: p=0.0095 ** — CS-Agg wins (0.9970 vs 0.9975)
- FedAvg vs FedProx: p=0.0573 ns — statistical tie (0.9970 vs 0.9975)
- CS-Agg vs FedProx: p=0.7907 ns — statistical tie (0.9975 vs 0.9975)

### Full AUC Table — By Method / Dataset / Alpha / Noise Rate
**3-seed averages (seeds 42, 123, 456). Use these numbers in all tables and figures.**

| Dataset       | Alpha | Noise | FedAvg | CS-Agg | FedProx |
|---------------|-------|-------|--------|--------|---------|
| breast_cancer | 0.1   | 0.0   | 0.9956 | 0.9969 | 0.9970  |
| breast_cancer | 0.1   | 0.1   | 0.9956 | 0.9971 | 0.9972  |
| breast_cancer | 0.1   | 0.2   | 0.9955 | 0.9970 | 0.9972  |
| breast_cancer | 0.1   | 0.3   | 0.9956 | 0.9969 | 0.9969  |
| breast_cancer | 0.5   | 0.0   | 0.9978 | 0.9976 | 0.9979  |
| breast_cancer | 0.5   | 0.1   | 0.9982 | 0.9978 | 0.9982  |
| breast_cancer | 0.5   | 0.2   | 0.9978 | 0.9976 | 0.9978  |
| breast_cancer | 0.5   | 0.3   | 0.9976 | 0.9976 | 0.9975  |
| breast_cancer | 1.0   | 0.0   | 0.9977 | 0.9980 | 0.9977  |
| breast_cancer | 1.0   | 0.1   | 0.9978 | 0.9980 | 0.9978  |
| breast_cancer | 1.0   | 0.2   | 0.9976 | 0.9979 | 0.9976  |
| breast_cancer | 1.0   | 0.3   | 0.9975 | 0.9977 | 0.9974  |
| brfss         | 0.1   | 0.0   | 0.5680 | 0.7670 | 0.6292  |
| brfss         | 0.1   | 0.1   | 0.5334 | 0.7475 | 0.6074  |
| brfss         | 0.1   | 0.2   | 0.5535 | 0.7425 | 0.6124  |
| brfss         | 0.1   | 0.3   | 0.5565 | 0.7417 | 0.6069  |
| brfss         | 0.5   | 0.0   | 0.7850 | 0.7859 | 0.7853  |
| brfss         | 0.5   | 0.1   | 0.7831 | 0.7860 | 0.7837  |
| brfss         | 0.5   | 0.2   | 0.7836 | 0.7861 | 0.7839  |
| brfss         | 0.5   | 0.3   | 0.7834 | 0.7858 | 0.7836  |
| brfss         | 1.0   | 0.0   | 0.7876 | 0.7878 | 0.7877  |
| brfss         | 1.0   | 0.1   | 0.7881 | 0.7889 | 0.7879  |
| brfss         | 1.0   | 0.2   | 0.7877 | 0.7883 | 0.7874  |
| brfss         | 1.0   | 0.3   | 0.7873 | 0.7881 | 0.7874  |

### AUC By Alpha (averaged over noise rates)

| Dataset       | Alpha | FedAvg | CS-Agg  | FedProx |
|---------------|-------|--------|---------|---------|
| breast_cancer | 0.1   | 0.9953 | 0.9974  | 0.9978  |
| breast_cancer | 0.5   | 0.9981 | 0.9979  | 0.9981  |
| breast_cancer | 1.0   | 0.9975 | 0.9980  | 0.9975  |
| brfss         | 0.1   | 0.6127 | 0.7488  | 0.6759  |
| brfss         | 0.5   | 0.7857 | 0.7892  | 0.7859  |
| brfss         | 1.0   | 0.7882 | 0.7886  | 0.7882  |

### Alpha=0.1 BRFSS Seed-Level Detail (the collapse case)

| Method  | Noise | Seed | AUC    | Note                      |
|---------|-------|------|--------|---------------------------|
| fedavg  | 0.0   | 42   | 0.4933 | COLLAPSE — below random   |
| fedavg  | 0.0   | 123  | 0.7740 | normal                    |
| fedavg  | 0.0   | 456  | 0.4368 | COLLAPSE — below random   |
| fedavg  | 0.1   | 42   | 0.4002 | COLLAPSE                  |
| fedavg  | 0.1   | 123  | 0.7714 | normal                    |
| fedavg  | 0.1   | 456  | 0.4287 | COLLAPSE                  |
| fedavg  | 0.2   | 42   | 0.4547 | COLLAPSE                  |
| fedavg  | 0.2   | 123  | 0.7712 | normal                    |
| fedavg  | 0.2   | 456  | 0.4345 | COLLAPSE                  |
| fedavg  | 0.3   | 42   | 0.4691 | COLLAPSE                  |
| fedavg  | 0.3   | 123  | 0.7672 | normal                    |
| fedavg  | 0.3   | 456  | 0.4331 | COLLAPSE                  |
| cs-agg  | 0.0   | 42   | 0.7753 | stable                    |
| cs-agg  | 0.0   | 123  | 0.7724 | stable                    |
| cs-agg  | 0.0   | 456  | 0.7532 | stable                    |
| cs-agg  | 0.1   | 42   | 0.7141 | stable                    |
| cs-agg  | 0.1   | 123  | 0.7743 | stable                    |
| cs-agg  | 0.1   | 456  | 0.7540 | stable                    |
| cs-agg  | 0.2   | 42   | 0.7025 | stable                    |
| cs-agg  | 0.2   | 123  | 0.7725 | stable                    |
| cs-agg  | 0.2   | 456  | 0.7525 | stable                    |
| cs-agg  | 0.3   | 42   | 0.7067 | stable                    |
| cs-agg  | 0.3   | 123  | 0.7724 | stable                    |
| cs-agg  | 0.3   | 456  | 0.7458 | stable                    |
| fedprox | 0.0   | 42   | 0.6184 | partial collapse          |
| fedprox | 0.0   | 123  | 0.7769 | normal                    |
| fedprox | 0.0   | 456  | 0.4924 | partial collapse          |
| fedprox | 0.1   | 42   | 0.5578 | partial collapse          |
| fedprox | 0.1   | 123  | 0.7735 | normal                    |
| fedprox | 0.1   | 456  | 0.4908 | partial collapse          |
| fedprox | 0.2   | 42   | 0.5729 | partial collapse          |
| fedprox | 0.2   | 123  | 0.7742 | normal                    |
| fedprox | 0.2   | 456  | 0.4901 | partial collapse          |
| fedprox | 0.3   | 42   | 0.5624 | partial collapse          |
| fedprox | 0.3   | 123  | 0.7708 | normal                    |
| fedprox | 0.3   | 456  | 0.4874 | partial collapse          |

**Collapse pattern summary:** Seeds 42 and 456 both produce degenerate Dirichlet partitions at alpha=0.1. Seed 123 avoids collapse. 2 out of 3 seeds collapse for FedAvg/FedProx, meaning the 3-seed average AUC is dragged far below what a single "lucky" run would show. CS-Agg remains stable across all 3 seeds (range: 0.70–0.78), confirming its robustness is not seed-dependent.

### Key Findings for Paper

1. **CS-Agg is the clear winner on BRFSS** — +6.6pp over FedAvg (p<0.001), +4.6pp over FedProx (p<0.001). This is the headline result. In the paper this method is called CS-Agg (Cosine Similarity Trust Aggregation), NOT FedNoRo.

2. **FedProx beats FedAvg on BRFSS** — +2.1pp (p=0.0003***). Even without noise-specific design, its proximal regularization helps.

3. **All methods saturate on Breast Cancer** — ceiling effect at ~0.997+ AUC. CS-Agg and FedProx are statistically indistinguishable (p=0.79). The dataset is too easy/small to differentiate methods under noise.

4. **Noise matters less than heterogeneity on BRFSS** — at alpha=0.5 and 1.0, all methods perform similarly (0.785-0.789) regardless of noise rate. The big performance gap only shows at alpha=0.1. Heterogeneity is the harder problem. **CRITICAL RATIO (use this in the paper, NOT any other number):** Moving from alpha=1.0 to alpha=0.1 drops FedAvg mean AUC by ~0.235 (0.7882→0.5529). Increasing noise from 0% to 30% (at alpha=0.5) drops mean AUC by only ~0.0016 (0.7850→0.7834). Ratio = 0.234808 / 0.001576 = **~149x** larger effect from heterogeneity than from noise. Any other ratio (e.g. 310x) in the paper draft is WRONG — correct it to ~149x. **Always state the slice alongside the ratio:** this 149x uses the heterogeneity effect *averaged over all four noise rates*. The same comparison restricted to noise=0.0 gives 0.2196/0.001576 = **139x**, and restricted to noise in {0.0, 0.2} (the slice comparable to the 12-feature ablation) gives 0.2269/0.001355 = **167x**. All three are correct for their slice; mixing the numbers from one with the ratio from another is the mistake to avoid.

5. **CS-Agg is dramatically more stable** — std 0.027 vs FedAvg's 0.126 on BRFSS. This is because FedAvg collapses completely under certain Dirichlet partitions (see finding 6).

6. **Critical finding — FedAvg/FedProx collapse at alpha=0.1, seeds 42 AND 456** — with these specific partitions, FedAvg hits AUC ~0.43–0.49 (worse than random at 0.5) while CS-Agg stays at 0.70+. This is because the Dirichlet draw with these seeds and alpha=0.1 creates some clients with 0 samples and one with 100k+, causing the global model to collapse to majority-class prediction. CS-Agg's trust mechanism downweights the dominant client, preventing collapse. 2 out of 3 seeds collapse for FedAvg/FedProx. This is a major finding: CS-Agg is robust to degenerate partitioning, not just label noise.

7. **2 of 3 seeds collapse at alpha=0.1 — FedAvg seed spread is bimodal** — FedAvg: 0.49 (seed=42), 0.77 (seed=123), 0.44 (seed=456). CS-Agg: 0.75 (seed=42), 0.77 (seed=123), 0.75 (seed=456). The bimodal distribution is visible in fig7_violin.png. The 3-seed mean (0.5680) is far below seed=123's value (0.774), showing how a single lucky seed can misrepresent method performance. CS-Agg is consistent across all seeds.

### Analysis Script

`analyze.py` at project root. Run with `python analyze.py` from fl-brfss directory. Produces summary tables and Wilcoxon results. Full code:

```python
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
    fedavg  = sub[sub["method"] == "fedavg"]["auc"].values
    csagg   = sub[sub["method"] == "csagg"]["auc"].values
    fedprox = sub[sub["method"] == "fedprox"]["auc"].values
    for m1, v1, m2, v2 in [
        ("fedavg", fedavg,  "csagg",   csagg),
        ("fedavg", fedavg,  "fedprox", fedprox),
        ("csagg",  csagg,   "fedprox", fedprox),
    ]:
        stat, p = stats.wilcoxon(v1, v2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        winner = m1 if np.mean(v1) > np.mean(v2) else m2
        print(f"  {m1} vs {m2}: p={p:.4f} {sig}  >>  {winner} wins  "
              f"(means: {np.mean(v1):.4f} vs {np.mean(v2):.4f})")
```

---

## 18b. FIGURES (GENERATED — May 24, 2026)

All 7 figures saved to `results/figures/`. Generated with full 3-seed data (216 conditions). All figures use CS-Agg label (not FedNoRo).

- `fig1_convergence.png` — BRFSS convergence: AUC vs round, 1×4 panels by noise rate, averaged over alpha+seeds. Dotted line at round 10 (warmup end). Lines only, no error bands.
- `fig2_bar_noise.png` — Final AUC by method × noise rate, both datasets (bar chart with SE error bars). Single shared legend at top.
- `fig3_heterogeneity_vs_noise.png` — KEY FIGURE: alpha effect (left) vs noise effect (right) side-by-side on BRFSS. Shows ~0.22 AUC gap from heterogeneity vs ~0.002 from noise (~149x ratio). Annotated directly on the plot.
- `fig4_heatmap.png` — Green RdYlGn heatmap, 3 panels (FedAvg/CS-Agg/FedProx), alpha × noise rate on BRFSS.
- `fig5_collapse.png` — Collapse visualization: left=scatter dots per seed at alpha=0.1 with mean bars, right=convergence curves per seed at alpha=0.1 noise=0.0. Shows bimodal behavior.
- `fig6_auc_by_alpha.png` — AUC by method and alpha group (high/mid/low heterogeneity), BRFSS only. Shows divergence at alpha=0.1.
- `fig7_violin.png` — Violin plots with individual points, Wilcoxon *** brackets, bimodal FedAvg distribution visible on BRFSS.

Regenerate with: `python figures.py` from fl-brfss directory.

## 18c. OPEN ITEMS (updated Aug 12, 2026)

Done since May: paper written and submitted to WF-PST (rejected); Krum /
trimmed mean / coordinate median added; uniform-mean control added;
noisy-fraction sweep; 12-feature ablation; figures 8-11; everything
consolidated into a single compiling `main.tex` + `references.bib`
(paper_edits/ deleted).

Still open:

1. **Anonymized code mirror** — Reviewer 2 explicitly asked for shared code.
   Needs names/emails stripped from files *and* git history, then hosted
   somewhere double-blind-safe (e.g. anonymous.4open.science). The URL fills
   the `[ANONYMOUS-MIRROR-URL]` placeholder in `overleaf/main.tex`'s
   Reproducibility Statement (search for that exact string). Note `.gitignore` currently
   excludes `*.csv`, so results are untracked — the mirror should include
   `results/experiment_results_merged.csv` so the tables can be reproduced
   without a 4-hour rerun.
2. **Page check** once submitting: confirmed 9 pages excluding references via
   a real `pdflatex` compile as of Aug 12 (see status header at the top of
   this file) — but re-check after any further edits, TPS's limit is 10.
3. **Final anonymization sweep** of the compiled `main.pdf` before upload
   (search for name/email/school one more time).
4. **Decide on figures 1-7**: `figures.py` still generates the original
   3-method figures, but `main.tex` does not reference them (superseded by
   the 7-strategy figures 8-11 from `figures_v2.py`). They're not wired into
   the paper, so there's nothing for them to conflict with, but if page
   budget gets tight there's no benefit to keeping `figures.py` running as
   part of the reproduction pipeline either.

### Known methodological caveats to disclose, not hide

- **BRFSS features are not standardized.** `split_data()` feeds raw values to
  the model; only Breast Cancer gets `StandardScaler`. In the 12-feature set
  raw ordinal codes run to 13 (`_AGEG5YR`) and 11 (`INCOME3`) alongside
  0-1 features. This is a confound for the feature ablation.
- **Sample-count weighting is decoupled from data actually used.** Clients
  train on at most 500 rows/round but report `len(self.X)` as `num_examples`,
  so at alpha=0.1 FedAvg puts ~35% of its weight on one client that trained on
  the same 500 rows as everyone else. This is why the `uniform_mean` control
  matters.
- **`noisy_fraction` is a fraction of clients, not of data.** At alpha=0.1 the
  noisy clients hold 13%-80% of training rows depending on seed, and the share
  does not increase smoothly with the fraction. This explains the
  non-monotonic sweep curves and limits how strongly the sweep can be read.
- **Multiple comparisons.** Ten Wilcoxon tests are reported; Holm correction
  demotes CS-Agg vs Krum on BRFSS to p=0.018 and CS-Agg vs coordinate median
  on Breast Cancer to non-significant. The core BRFSS result (coordinate
  median > CS-Agg) survives at p=0.0002.

---

## 19. GOOGLE DRIVE STRUCTURE (CURRENT STATE)

After the May 24 upload session:
```
My Drive/
└── fl-brfss/
    ├── smoke_test.py              ← uploaded May 24
    └── src/
        ├── client.py              ← updated May 24 (added FedProxClient)
        ├── data.py                ← original
        ├── model.py               ← original
        ├── noise.py               ← original
        ├── run_experiments.py     ← rewritten May 24
        ├── server_fedavg.py       ← updated May 24
        ├── server_csagg.py      ← updated May 24
        ├── server_fedprox.py      ← new file, May 24
        └── simulate.py            ← updated May 24
```

Note: `data/LLCP2023.XPT` must be uploaded directly to Colab (not Drive) each session because the file is ~400MB and Drive mounting can be unreliable for large files.

---

## 20. KEY DESIGN DECISIONS (AND WHY)

| Decision | Choice | Reason |
|---|---|---|
| Simulation framework | Custom sequential loop | Faster than Flower+Ray on Colab, no initialization overhead |
| Model architecture | BinaryMLP (64-32-2) | Matches BRFSS feature complexity; small enough to train fast across 20 clients × 50 rounds |
| Optimizer | Adam lr=0.001 | Standard; SGD needs careful tuning, Adam is more robust |
| Local epochs | 3 | Balances computation and communication; standard in FL literature |
| Training subsample | 500 | Caps per-client train time on BRFSS (large dataset); varies by round |
| Eval subsample | 2000 | Caps eval time without losing too much statistical power |
| Noise fraction | 20% (4 of 20 clients) | Realistic fraction; enough to test robustness without making task trivial |
| Seeds | [42, 123, 456] | 3 seeds for stronger statistical power (n=36 per method per dataset for Wilcoxon) |
| Primary metric | AUC-ROC | Handles class imbalance; standard for health classification |
| Test set | Server-side clean set | Only uncontaminated ground truth; required for fair cross-method comparison |
| FedProx mu | 0.01 | Standard value from Li et al. 2020 paper |
| CS-Agg warmup | 10 rounds | Needed so there's a meaningful update "direction" to compute cosine similarity from; inspired by Wu et al. 2023 |

---

## 21. COMPUTE BUDGET NOTES

- **Colab Free:** Disconnects after ~90 minutes, ~0 reliable compute units. DO NOT USE.
- **Colab Pro ($10/month):** ~100 compute units per month. A100 costs ~1 CU/hour.
- **This experiment:** Used significantly more than planned due to multiple session disconnects and restarts (see Section 23).
- **If experiment crashes mid-way:** Resume logic means you just re-run. Only the in-progress condition is lost. Re-running with A100 picks up from where it left off — BUT see Section 23 for how the resume logic can fail.

---

## 23. SESSION HISTORY AND BUGS FIXED (May 24, 2026)

This section documents what went wrong during the actual experiment run and what was fixed. Read this before running any future experiments.

### What happened
1. Smoke test passed (ALL OK)
2. Experiment started on A100, ran 14 conditions (~2 hours)
3. User closed laptop → Colab session disconnected → kernel reset
4. On reconnect, running `%run src/run_experiments.py` without re-running Cell 3 (the `os.chdir` cell) caused `FileNotFoundError` — working directory had reset to `/content/` instead of the Drive path
5. Multiple restart attempts caused the Drive CSV to accumulate corrupt partial rows from interrupted writes
6. `pyreadstat` was missing from pip install (was in requirements but not in the Cell 2 command) — added `pyreadstat` to Cell 2

### Bugs fixed in run_experiments.py during this session

**Bug A: Silent resume failure**
- Old `load_completed()` caught all exceptions with `except Exception: return set()` — silently returned 0 completed if CSV had ANY parse error
- Fix: prints diagnostics, checks both Drive CSV and local backup, unions the two sets

**Bug B: No force-flush on Drive writes**
- Google Drive FUSE filesystem buffers writes. When session dies, buffered writes are lost.
- Fix: after every condition save, `f.flush(); os.fsync(f.fileno())` to force disk sync

**Bug C: Single save location**
- If Drive write buffer not flushed → progress lost on disconnect
- Fix: added `LOCAL_BACKUP = "/content/results_backup.csv"` as second write target. Every condition saves to BOTH Drive and local. On resume, both are checked and unioned.
- NOTE: local backup is `/content/` which is ephemeral — it's lost on session restart. It only helps if the session stays alive but Drive write fails.

**Bug D: Corrupt CSV not recoverable**
- The old CSV had corrupt lines from an interrupted write mid-row
- Recovery: `pd.read_csv(path, on_bad_lines='skip')` to recover good rows, then rewrite clean
- After recovery + new data, CSV had 400 junk NaN rows that needed to be dropped with `df.dropna(subset=['dataset','noise_type','auc'])`

### Lessons for future runs
- **NEVER close the browser tab** — closing the tab kills the session even if laptop stays open
- Closing the laptop lid is usually fine (tab stays open), but risky on battery
- Run a keepalive cell in parallel: `while True: time.sleep(60); clear_output()`
- The resume logic now prints row counts from both files so you can see exactly what's recoverable
- After any crash, before rerunning: check the CSV with `on_bad_lines='skip'` and repair if needed

---

## 22. CONTACTS AND REFERENCES

- **Researcher:** Kunal Chevuri, ckchevuri@gmail.com
- **FedAvg:** McMahan et al., AISTATS 2017. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- **FedProx:** Li et al., MLSys 2020. "Federated Optimization in Heterogeneous Networks"
- **FedNoRo:** Wu et al., IJCAI 2023. "FedNoRo: Towards Noise-Robust Federated Learning by Addressing Class Imbalance and Label Noise Heterogeneity"
- **BRFSS 2023:** CDC Behavioral Risk Factor Surveillance System, https://www.cdc.gov/brfss/
- **Breast Cancer dataset:** UCI ML Repository via sklearn.datasets.load_breast_cancer()
- **Dirichlet partitioning:** Li et al., 2022 (standard FL heterogeneity benchmark method)
