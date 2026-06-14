# Federated Learning Robustness for Behavioral Health Data
**Paper:** Improving Federated Learning Robustness for Public Health and Safety Applications under Heterogeneous and Noisy Data
**Conference:** IEEE WF-PST 2026 (submitted)
**Authors:** Kunal Chevuri, Dr. Sergei Chuprov

## Overview
Federated learning study examining how data heterogeneity and asymmetric label noise jointly degrade FL performance on self-reported behavioral health data, and whether cosine similarity trust aggregation (CS-Agg) provides meaningful robustness compared to FedAvg and FedProx.

## Key Findings
- Heterogeneity dominates noise degradation by 149x
- FedAvg and FedProx collapse to below-random AUC at alpha=0.1
- CS-Agg prevents collapse and maintains 0.70+ AUC across all conditions
- CS-Agg vs FedAvg: +6.6 percentage points, p<0.0001, 4.7x more stable

## Datasets
- BRFSS 2023 (CDC behavioral health survey) — not included, download from cdc.gov/brfss
- UCI Breast Cancer Wisconsin — not included

## Project Structure
- src/ — data pipeline, model, noise injection, FL client and server implementations
- results/ — experiment output CSVs
- figures/ — paper figures

## Requirements
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn pyreadstat

## Status
Paper under review. Repository will be made public upon acceptance.
