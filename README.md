# Online Payment Fraud Detection System

An end-to-end machine learning pipeline to detect fraudulent financial transactions, built on a dataset of **6.3M+ real-world synthetic transactions** with a severe class imbalance of just **0.17% fraud rate**.

---

## Results

| Model         | ROC-AUC |
|---------------|---------|
| Random Forest | ~0.97+  |
| XGBoost       | ~0.97+  |

> Run the pipeline on your machine to get exact scores — results may vary slightly based on sampling.

---

## Pipeline Overview

```
Load & Validate Data
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering  (5 derived features)
        ↓
Train / Test Split  (80 / 20, stratified)
        ↓
SMOTE Oversampling  (sampling_strategy = 0.1)
        ↓
Model Training  (Random Forest + XGBoost)
        ↓
Evaluation  (ROC-AUC, Confusion Matrix, Classification Report)
```

---

## Tech Stack

| Category       | Tools                                          |
|----------------|------------------------------------------------|
| Language       | Python 3.10+                                   |
| ML Models      | Scikit-learn (Random Forest), XGBoost          |
| Imbalance      | imbalanced-learn (SMOTE)                       |
| Data           | Pandas, NumPy                                  |
| Visualisation  | Matplotlib, Seaborn                            |

---

## Key Features

- **EDA** — fraud rate analysis, breakdown by transaction type, class imbalance visualisation
- **Feature Engineering** — 5 high-signal derived features:
  - `orig_balance_diff` — detects accounts drained by a transaction
  - `dest_balance_diff` — flags suspicious destination balance changes
  - `amount_orig_ratio` — transaction size relative to origin balance
  - `orig_account_drained` — binary flag for fully emptied origin accounts
  - `dest_was_empty` — binary flag for mule account detection
- **SMOTE Oversampling** — handles 0.17% fraud class imbalance without discarding data
- **Model Benchmarking** — Random Forest vs XGBoost with full ROC-AUC evaluation
- **CLI Interface** — flexible `argparse`-based command-line usage
- **Production-style code** — modular functions, docstrings, logging, and error handling throughout

---
