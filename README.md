# Robustness or Weighting? Federated Aggregation on Heterogeneous Health Data

Comparative study of seven federated-learning aggregation strategies under
joint data heterogeneity and label noise, on CDC BRFSS 2023 (self-reported
behavioral health) and UCI Breast Cancer Wisconsin (clinical benchmark).

**Status:** submitted to IEEE TPS 2026, Round 2, and under review. Previously
submitted to IEEE WF-PST 2026 and rejected; this version responds to that
review round (Byzantine-robust baselines, feature-choice sensitivity, code
release).

## The paper

The manuscript source and compiled PDF are **not** in this repository while
the paper is under double-blind review. They land here once a decision comes
back. Everything the paper reports -- the code, the aggregation strategies,
the merged results and every figure -- is here now.

`verify_paper_numbers.py` (repo root) asserts every number in the paper
against `results/experiment_results_merged.csv` (181 checks). That CSV is
committed, so the script runs against a clean clone with no experiments
re-run:

```bash
python verify_paper_numbers.py
```

## Methods compared

| Method | Weighting | Needs corruption bound `f`? |
|---|---|---|
| FedAvg | by client sample count `n_i` | no |
| FedProx (mu=0.01) | by `n_i` | no |
| CS-Agg | pairwise cosine-similarity trust, 10-round warmup | no |
| Krum | winner-take-all, single client | yes |
| Trimmed mean | per-coordinate, drop `f` high + `f` low | yes |
| Coordinate-wise median | per-coordinate median | no |
| Uniform mean | unweighted average (control) | no |

`uniform_mean` is a **control baseline**, not a proposed method. FedAvg
weights by `n_i`; every other method above ignores `n_i`. Under extreme
Dirichlet skew one client can hold >30% of the training data, so without
this control any advantage the robust methods show is confounded between
"uses a robust statistic" and "does not over-weight the largest client".

## Experimental grid

- Datasets: BRFSS 2023 (4-feature primary, 12-feature ablation), Breast Cancer
- Heterogeneity: Dirichlet alpha in {0.1, 0.5, 1.0}, stratified per class
- Label noise: {0.0, 0.1, 0.2, 0.3}; asymmetric on BRFSS (1->0 at `r`,
  0->1 at `r/2`), symmetric on Breast Cancer
- Noisy-client fraction: 0.2 main grid; swept {0.2..0.6} at alpha=0.1
- Seeds: 42, 123, 456 | 20 clients | 50 rounds | primary metric AUC-ROC

## Layout

```
src/            data pipeline, model, noise injection, client, aggregators
  data.py             BRFSS load/preprocess (4- and 12-feature), Dirichlet partition
  model.py            BinaryMLP (d->64->32->2); 2,466 / 2,978 / 4,130 params
  client.py           local training (500-sample/round cap), FedProx variant
  simulate.py         FL loop + fedavg_aggregate
  server_*.py         aggregation strategies
  run_experiments_v2.py   argparse-driven, resume-safe, shardable runner
run_shards.py   batched shard launcher (memory-capped concurrency)
analyze_v2.py   merges shards -> results/experiment_results_merged.csv + all tables
figures.py      figures 1-7 (original three-method study; NOT used in main.tex)
figures_v2.py   figures 8-11 (baselines, sweep, ablation; used in the paper).
                Generated at final print size (COL=3.5in, FULL=7.16in) so
                LaTeX applies no scaling and font sizes match across figures.
                No figure carries an embedded title -- captions live in the .tex.
poster/         UNT Research Day poster: build_poster.py fills the COI
                template via python-pptx, figures_poster.py redraws the
                figures at poster scale (~20pt type for a 24x36in board)
verify_paper_numbers.py   181 assertions checking the paper's numbers vs. the CSV
results/        merged results CSV and figures
```

Figures 1-7 predate the 7-strategy reframing (they only ever showed
FedAvg/FedProx/CS-Agg) and are superseded by figures 8-11, which cover all
seven strategies including the uniform-mean control. `figures.py` is kept for
provenance but its output is not referenced anywhere in the paper -- there is
nothing in the paper that could conflict with it.

## Reproducing

```bash
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn pyreadstat flwr
# BRFSS 2023 LLCP XPT is not redistributed here -- download from
# https://www.cdc.gov/brfss/annual_data/annual_2023.html into data/LLCP2023.XPT
python run_shards.py 3      # 3 concurrent shards; resume-safe, just re-run if interrupted
python analyze_v2.py
python figures.py && python figures_v2.py
```

Breast Cancer Wisconsin loads from `sklearn.datasets`; no download needed.

## Known limitations recorded in code

- BRFSS features are **not** standardized (`split_data` passes raw values).
  Only the Breast Cancer pipeline applies `StandardScaler`. This matters most
  for the 12-feature ablation, where raw ordinal codes range up to 13.
- Local training subsamples to 500 rows/client/round, but FedAvg still weights
  by full partition size `n_i`, so its weights are decoupled from the amount
  of data actually seen. Disclosed in the paper.
- `noisy_fraction` is a fraction of *clients*, not of *data*. Under alpha=0.1
  the noisy clients hold anywhere from 13% to 80% of training rows depending
  on seed, which is why the sweep curves are non-monotonic.

## Status

Under review at IEEE TPS 2026 Round 2. `HANDOFF.md` is the detailed
engineering log (architecture, fixed bugs, full parameter table).
