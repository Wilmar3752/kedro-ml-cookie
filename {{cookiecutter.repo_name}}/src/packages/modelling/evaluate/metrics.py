import typing as tp
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# Regression metrics
# ======================================================================


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tp.Dict[str, float]:
    """Compute a standard suite of regression metrics.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.
        sample_weight: Optional per-sample weights.

    Returns:
        Dict with keys: ``mse``, ``rmse``, ``mae``, ``mape``, ``median_ae``,
        ``r2``, ``explained_variance``.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred, sample_weight=sample_weight),
        "mape": mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight),
        "median_ae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred, sample_weight=sample_weight),
        "explained_variance": explained_variance_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
    }


# ======================================================================
# Binary classification metrics
# ======================================================================


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> tp.Dict[str, float]:
    """Compute a comprehensive suite of binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 / 1).
        y_pred: Predicted binary labels (0 / 1).
        y_score: Predicted probabilities for the positive class. When provided,
            AUC-based metrics (``roc_auc``, ``brier_score``) are also computed.
        sample_weight: Optional per-sample weights.

    Returns:
        Dict containing accuracy, F1 variants, precision/recall, Matthews
        correlation coefficient, F-beta scores, and optionally ROC AUC and
        Brier score.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    result = {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "f1": f1_score(y_true, y_pred, sample_weight=sample_weight),
        "f1_micro": f1_score(y_true, y_pred, average="micro", sample_weight=sample_weight),
        "f1_macro": f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "precision": precision_score(y_true, y_pred, sample_weight=sample_weight),
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", sample_weight=sample_weight
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "recall": recall_score(y_true, y_pred, sample_weight=sample_weight),
        "recall_micro": recall_score(
            y_true, y_pred, average="micro", sample_weight=sample_weight
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "f2_score": fbeta_score(y_true, y_pred, beta=2, sample_weight=sample_weight),
        "f2_score_weighted": fbeta_score(
            y_true, y_pred, beta=2, average="weighted", sample_weight=sample_weight
        ),
        "precision_class_1": precision_score(
            y_true, y_pred, pos_label=1, sample_weight=sample_weight
        ),
        "recall_class_1": recall_score(
            y_true, y_pred, pos_label=1, sample_weight=sample_weight
        ),
    }

    if y_score is not None:
        result.update(
            {
                "roc_auc": roc_auc_score(y_true, y_score, sample_weight=sample_weight),
                "brier_score": brier_score_loss(
                    y_true, y_score, sample_weight=sample_weight
                ),
            }
        )

    return result


# ======================================================================
# Multi-class classification metrics
# ======================================================================


def compute_multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tp.Dict[str, float]:
    """Compute a standard suite of multi-class classification metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        sample_weight: Optional per-sample weights.

    Returns:
        Dict containing accuracy, balanced accuracy, F1 / precision / recall
        variants (micro, macro, weighted), and Matthews correlation coefficient.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, sample_weight=sample_weight
        ),
        "f1_micro": f1_score(y_true, y_pred, average="micro", sample_weight=sample_weight),
        "f1_macro": f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "precision_micro": precision_score(
            y_true, y_pred, average="micro", sample_weight=sample_weight
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "recall_micro": recall_score(
            y_true, y_pred, average="micro", sample_weight=sample_weight
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", sample_weight=sample_weight
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", sample_weight=sample_weight
        ),
        "matthews_corrcoef": matthews_corrcoef(
            y_true, y_pred, sample_weight=sample_weight
        ),
    }
