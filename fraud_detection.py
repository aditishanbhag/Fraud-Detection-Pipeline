"""
Online Payment Fraud Detection System
======================================
Detects fraudulent transactions in a dataset of 6.3M+ financial records.

Pipeline:
    1. Load & validate data
    2. Exploratory Data Analysis (EDA)
    3. Feature Engineering
    4. Train/Test Split + SMOTE oversampling
    5. Model Training (Random Forest, XGBoost)
    6. Evaluation (ROC-AUC, Confusion Matrix, Classification Report)

Usage:
    python fraud_detection.py --data new_file.csv
    python fraud_detection.py --data new_file.csv --sample --sample-size 200000
    python fraud_detection.py --data new_file.csv --output-dir results/

Author : Aditi Shanbhag
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

#logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

#constants
REQUIRED_COLUMNS = {
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFraud",
}
FRAUD_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT"]
RANDOM_STATE = 42

#loading data
def load_data(filepath: str, sample: bool = False, sample_size: int = 200_000) -> pd.DataFrame:
    """
    Load the transaction dataset from a CSV file.

    Args:
        filepath    : Path to the CSV file.
        sample      : If True, load only a random sample of rows.
        sample_size : Number of rows to sample (used only when sample=True).

    Returns:
        pd.DataFrame: Loaded (and optionally sampled) dataset.

    Raises:
        FileNotFoundError : If the CSV file does not exist.
        ValueError        : If required columns are missing from the dataset.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            f"Download it from https://www.kaggle.com/datasets/ealaxi/paysim1 "
            f"and place it in the same directory as this script."
        )

    logger.info("Loading dataset from '%s'...", filepath)
    df = pd.read_csv(filepath)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if sample:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)
        logger.info("Sampled %s rows for quick testing.", f"{len(df):,}")
    else:
        logger.info("Loaded full dataset: %s rows × %s columns.", f"{len(df):,}", df.shape[1])

    return df

#eda
def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Perform exploratory data analysis and save visualisations.

    Logs key statistics (fraud rate, class imbalance, type breakdown)
    and saves an EDA overview chart to output_dir.

    Args:
        df         : Raw transaction DataFrame.
        output_dir : Directory to save output PNG files.
    """
    logger.info("── EDA ─────────────────────────────────────────")

    fraud_count = df["isFraud"].sum()
    fraud_rate = df["isFraud"].mean() * 100

    logger.info("Total transactions : %s", f"{len(df):,}")
    logger.info("Fraud transactions : %s (%.4f%%)", f"{fraud_count:,}", fraud_rate)
    logger.info("Class imbalance    : Severe — only %.2f%% are fraud", fraud_rate)

    type_counts = df["type"].value_counts()
    fraud_by_type = df[df["isFraud"] == 1]["type"].value_counts()
    logger.info("Transaction types:\n%s", type_counts.to_string())
    logger.info("Fraud by type:\n%s", fraud_by_type.to_string())

    # Chart: fraud count and fraud rate by transaction type
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fraud Detection – EDA Overview", fontsize=14, fontweight="bold")

    type_fraud = df.groupby("type")["isFraud"].sum().sort_values(ascending=False)
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in type_fraud.values]
    axes[0].bar(type_fraud.index, type_fraud.values, color=colors)
    axes[0].set_title("Fraud Count by Transaction Type")
    axes[0].set_xlabel("Transaction Type")
    axes[0].set_ylabel("Fraud Count")
    axes[0].tick_params(axis="x", rotation=15)

    fraud_pct = df.groupby("type")["isFraud"].mean() * 100
    axes[1].bar(fraud_pct.index, fraud_pct.values, color="#e67e22")
    axes[1].set_title("Fraud Rate (%) by Transaction Type")
    axes[1].set_xlabel("Transaction Type")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out_path = output_dir / "eda_overview.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)

#feature engineering
def engineer_features(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Filter, encode, and engineer features for model training.

    Filters to fraud-relevant transaction types (TRANSFER, CASH_OUT),
    encodes categoricals, and derives 5 high-signal fraud indicators.
    Also saves a correlation heatmap.

    Args:
        df         : Raw transaction DataFrame.
        output_dir : Directory to save the correlation heatmap PNG.

    Returns:
        X        : Feature DataFrame.
        y        : Target Series (isFraud).
        features : List of feature column names.
    """
    logger.info("── Feature Engineering ─────────────────────────")

    # Fraud only occurs in TRANSFER and CASH_OUT
    df = df[df["type"].isin(FRAUD_TRANSACTION_TYPES)].copy()
    logger.info(
        "Filtered to %s: %s rows | Fraud rate: %.2f%%",
        FRAUD_TRANSACTION_TYPES, f"{len(df):,}", df["isFraud"].mean() * 100,
    )

    # Encode transaction type
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])

    #Derived features
    df["orig_balance_diff"]   = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_diff"]   = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_orig_ratio"]   = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["orig_account_drained"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)
    df["dest_was_empty"] = (df["oldbalanceDest"] == 0).astype(int)

    features = [
        "type_encoded", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "orig_balance_diff", "dest_balance_diff",
        "amount_orig_ratio", "orig_account_drained", "dest_was_empty",
    ]

    logger.info("Engineered %d features: %s", len(features), features)

    #Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[features + ["isFraud"]].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)

    return df[features], df["isFraud"], features

#SMOTE oversampling
def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to balance the training set.

    Uses a sampling_strategy of 0.1 so the minority (fraud) class
    becomes 10% of the majority class — enough to train on without
    over-inflating the dataset.

    Args:
        X_train : Training features.
        y_train : Training labels.

    Returns:
        X_resampled : Resampled feature DataFrame.
        y_resampled : Resampled label Series.
    """
    logger.info("── SMOTE Oversampling ──────────────────────────")
    logger.info(
        "Before SMOTE — Fraud: %s | Non-fraud: %s",
        f"{y_train.sum():,}", f"{(y_train == 0).sum():,}",
    )

    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(
        "After SMOTE  — Fraud: %s | Non-fraud: %s",
        f"{y_res.sum():,}", f"{(y_res == 0).sum():,}",
    )
    return X_res, y_res

#training the model
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train : Resampled training features.
        y_train : Resampled training labels.

    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("Training Random Forest (n_estimators=100, max_depth=12)...")
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest training complete.")
    return rf


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost classifier with scale_pos_weight to handle imbalance.

    Args:
        X_train : Resampled training features.
        y_train : Resampled training labels.

    Returns:
        Trained XGBClassifier.
    """
    logger.info("Training XGBoost (n_estimators=100, max_depth=6, lr=0.1)...")
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_weight, random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_train, y_train)
    logger.info("XGBoost training complete.")
    return xgb

#evaluation
def evaluate_models(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    output_dir: Path,
) -> dict:
    """
    Evaluate all trained models and save visualisations.

    Logs classification reports and ROC-AUC scores for each model,
    and saves a 4-panel chart (ROC curves, feature importance,
    confusion matrices) to output_dir.

    Args:
        models     : Dict of {"Model Name": trained_model}.
        X_test     : Test features.
        y_test     : True test labels.
        features   : Feature names (for importance plot).
        output_dir : Directory to save the evaluation PNG.

    Returns:
        results : Dict of {"Model Name": {"auc": float, "predictions": array}}.
    """
    logger.info("── Model Evaluation ────────────────────────────")

    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        auc = roc_auc_score(y_test, proba)
        results[name] = {"auc": auc, "proba": proba, "preds": preds}
        logger.info("%s ROC-AUC: %.4f", name, auc)
        logger.info("\n%s", classification_report(y_test, preds, target_names=["Not Fraud", "Fraud"]))

    #4-panel evaluation chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Fraud Detection – Model Evaluation", fontsize=14, fontweight="bold")

    #ROC Curves
    ax = axes[0, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {res['auc']:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3)

    #XGBoost Feature Importance
    ax = axes[0, 1]
    xgb_model = models.get("XGBoost")
    if xgb_model:
        importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True)
        importances.plot(kind="barh", ax=ax, color="#2ecc71")
    ax.set_title("XGBoost – Feature Importance")
    ax.set_xlabel("Importance Score")

    #Confusion Matrices
    cmap_colors = ["Blues", "Oranges"]
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[1, idx]
        cm = confusion_matrix(y_test, res["preds"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap_colors[idx], ax=ax,
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
        )
        ax.set_title(f"{name} – Confusion Matrix\nROC-AUC: {res['auc']:.4f}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    out_path = output_dir / "model_evaluation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", out_path)

    return results

#summary
def print_summary(results: dict, n_features: int) -> None:
    """
    Print a final summary of model results to the console.

    Args:
        results    : Output from evaluate_models().
        n_features : Number of features engineered.
    """
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_auc = results[best_name]["auc"]

    logger.info("=" * 55)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 55)
    for name, res in results.items():
        logger.info("  %-20s ROC-AUC: %.4f", name, res["auc"])
    logger.info("  Best model         : %s (AUC = %.4f)", best_name, best_auc)
    logger.info("  Features engineered: %d", n_features)
    logger.info("  SMOTE applied      : Yes (sampling_strategy=0.1)")
    logger.info("=" * 55)

#arguement parsing
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with fields:
            data        : Path to the input CSV file.
            sample      : Whether to run on a sample.
            sample_size : Number of rows to sample.
            output_dir  : Directory to save output charts.
    """
    parser = argparse.ArgumentParser(
        description="Online Payment Fraud Detection – ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="new_file.csv",
        help="Path to the transaction CSV dataset.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Run on a random sample for quick testing.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=200_000,
        help="Number of rows to sample (only used with --sample).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Directory to save output charts.",
    )
    return parser.parse_args()

#main
def main() -> None:
    """
    Orchestrate the full fraud detection pipeline:
        load → EDA → feature engineering → SMOTE → train → evaluate → summary.
    """
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir.resolve())

    try:
        # 1. Load
        df = load_data(args.data, sample=args.sample, sample_size=args.sample_size)

        # 2. EDA
        run_eda(df, output_dir)

        # 3. Feature Engineering
        X, y, features = engineer_features(df, output_dir)

        # 4. Train/Test Split
        logger.info("── Train/Test Split ────────────────────────────")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        logger.info("Train: %s rows | Test: %s rows", f"{len(X_train):,}", f"{len(X_test):,}")

        # 5. SMOTE
        X_train_res, y_train_res = apply_smote(X_train, y_train)

        # 6. Train Models
        logger.info("── Model Training ──────────────────────────────")
        models = {
            "Random Forest": train_random_forest(X_train_res, y_train_res),
            "XGBoost":       train_xgboost(X_train_res, y_train_res),
        }

        # 7. Evaluate
        results = evaluate_models(models, X_test, y_test, features, output_dir)

        # 8. Summary
        print_summary(results, n_features=len(features))

    except FileNotFoundError as e:
        logger.error("File error: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Data error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
