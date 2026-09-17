"""
evaluation_module.py
----------------------
Functional Module 4: Model Evaluation.

Scores the held-out test set and produces:
  - accuracy / precision / recall / F1 / ROC-AUC
  - a confusion matrix plot
  - an ROC curve plot
  - a feature-importance (top genes) plot

All plots + a metrics.json are written to outputs/, giving an auditable
record of each run (supports the Reliability & Reporting requirements).
"""

import json

import matplotlib
matplotlib.use("Agg")  # headless backend -- required for CLI-only execution
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix,
                              classification_report)

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(config.POSITIVE_CLASS)]

    y_test_bin = (y_test == config.POSITIVE_CLASS).astype(int)
    y_pred_bin = (y_pred == config.POSITIVE_CLASS).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_test_bin, y_pred_bin, zero_division=0),
        "f1_score": f1_score(y_test_bin, y_pred_bin, zero_division=0),
        "roc_auc": roc_auc_score(y_test_bin, y_proba),
    }
    logger.info("Test-set metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, save_path):
    labels = [config.POSITIVE_CLASS, config.NEGATIVE_CLASS]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix -- OSCC Mesenchymal Subtype")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", save_path)


def plot_roc_curve(y_test, y_proba, roc_auc, save_path):
    y_test_bin = (y_test == config.POSITIVE_CLASS).astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- MS vs Other")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve plot to %s", save_path)


def plot_feature_importance(model, feature_names, save_path, top_n=None):
    top_n = top_n or config.TOP_N_GENES
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(6, 5))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="teal")
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Genes Driving MS Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved feature importance plot to %s", save_path)

    return [(feature_names[i], float(importances[i])) for i in order]


def run_evaluation(model=None, X_test=None, y_test=None):
    if model is None:
        import joblib
        model = joblib.load(config.DEFAULT_MODEL_PATH)
        from src.preprocessing_module import run_preprocessing
        _, X_test, _, y_test = run_preprocessing()

    metrics, y_pred, y_proba = compute_metrics(model, X_test, y_test)

    cm_path = f"{config.OUTPUTS_DIR}/confusion_matrix.png"
    roc_path = f"{config.OUTPUTS_DIR}/roc_curve.png"
    fi_path = f"{config.OUTPUTS_DIR}/feature_importance.png"

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"], roc_path)
    top_genes = plot_feature_importance(model, list(X_test.columns), fi_path)

    results = {"metrics": metrics, "top_genes": top_genes}
    with open(f"{config.OUTPUTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved evaluation summary to %s/metrics.json", config.OUTPUTS_DIR)

    return results


if __name__ == "__main__":
    run_evaluation()
