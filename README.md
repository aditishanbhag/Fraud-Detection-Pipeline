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

## Project Structure

```
fraud-detection/
│
├── fraud_detection.py       ← main pipeline script
├── requirements.txt         ← dependencies
├── README.md                ← you are here
├── .gitignore               ← excludes dataset and large files
│
└── outputs/                 ← generated charts (committed for preview)
    ├── eda_overview.png
    ├── correlation_heatmap.png
    └── model_evaluation.png
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The dataset is too large for GitHub. Download it from Kaggle and place it in the project root:

🔗 [PaySim Synthetic Financial Dataset – Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

Rename the file to `new_file.csv` or pass your filename via `--data`.

### 4. Run the pipeline

```bash
# Full dataset
python fraud_detection.py --data new_file.csv

# Quick test on 200k sample
python fraud_detection.py --data new_file.csv --sample

# Custom sample size and output directory
python fraud_detection.py --data new_file.csv --sample --sample-size 300000 --output-dir results/
```

---

## Output Charts

Three charts are saved to the `outputs/` directory after each run:

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `eda_overview.png`        | Fraud count and fraud rate by transaction type   |
| `correlation_heatmap.png` | Feature correlation matrix                       |
| `model_evaluation.png`    | ROC curves, feature importance, confusion matrices|

---

## CLI Reference

```
usage: fraud_detection.py [-h] [--data DATA] [--sample] [--sample-size N] [--output-dir DIR]

arguments:
  --data          Path to the transaction CSV dataset  (default: new_file.csv)
  --sample        Run on a random sample for quick testing
  --sample-size   Number of rows to sample             (default: 200000)
  --output-dir    Directory to save output charts      (default: outputs)
```

---

## Dataset

**PaySim** is a synthetic financial dataset simulating mobile money transactions, generated using real transaction logs from a financial company in Africa. It is widely used for fraud detection research.

- 6,362,620 transactions
- 0.17% fraud rate (severe class imbalance)
- Fraud only occurs in `TRANSFER` and `CASH_OUT` transaction types

---

## Author

**Aditi Shanbhag**
- 📧 shanbhagaditi82@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/your-profile)
- 🐙 [GitHub](https://github.com/your-username)
