"""Titanic — binary classification performance report (HTML)."""
import typing as tp

import numpy as np
import pandas as pd

from ..charts.performance._calibration_curve import plot_calibration_curve
from ..charts.performance._classification_report import plot_classification_report
from ..charts.performance._confusion_matrix import plot_confusion_matrix
from ..charts.performance._feature_importance import plot_feature_importance
from ..charts.performance._precision_recall_curve import plot_precision_recall_curve
from ..charts.performance._predictions_distribution import plot_predictions_distribution
from ..charts.performance._roc_auc_curve import plot_roc_auc_curve
from ..rendering.html.report_generation import ReportMetaData, generate_html_report
from .identifiers import SectionHeader, Table


def create_titanic_performance_report(
    model,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> str:
    """Generate a self-contained HTML performance report for the Titanic model.

    Sections
    --------
    1. Model Metrics          — CV scores per fold + test-set summary table.
    2. Classification Quality — Confusion matrix, classification report heatmap.
    3. Discrimination         — ROC AUC curve + Precision-Recall curve.
    4. Calibration            — Calibration curve.
    5. Predictions            — Probability distribution by true class.
    6. Feature Importance     — LightGBM gain-based importance.

    Args:
        model: Fitted ``BinaryClassifierSklearnPipeline``.
        df_test: Test DataFrame including the target column.
        params: ``titanic.hypertune`` parameter dict (must contain ``target``).

    Returns:
        Self-contained HTML string.
    """
    target = params["target"]
    threshold = params.get("probability_threshold", 0.5)
    class_names = ["Not Survived", "Survived"]

    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True).to_numpy()
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # ── Model metrics from cross-validation ──────────────────────────────────
    cv_metrics_df = model.df_scores.T.copy()
    cv_metrics_df.columns = [f"fold_{i + 1}" for i in cv_metrics_df.columns]
    cv_summary_df = cv_metrics_df.T.describe().T.round(4)

    # Test metrics
    from packages.modelling.evaluate.metrics import compute_binary_classification_metrics
    test_metrics = compute_binary_classification_metrics(y_test, y_pred, y_score=y_probs)
    test_metrics_df = pd.DataFrame(
        {"Metric": list(test_metrics.keys()), "Test Score": list(test_metrics.values())}
    ).round(4)

    # ── Plotly figures ────────────────────────────────────────────────────────
    fig_roc = plot_roc_auc_curve(y_test, y_probs, title="ROC AUC Curve — Titanic Survival")
    fig_pr = plot_precision_recall_curve(y_test, y_probs, title="Precision-Recall Curve — Titanic Survival")
    fig_cm = plot_confusion_matrix(y_test, y_pred, class_names=class_names, title="Confusion Matrix — Titanic Survival")
    fig_clf_report = plot_classification_report(y_test, y_pred, class_names=class_names, title="Classification Report Heatmap")
    fig_calib = plot_calibration_curve(y_test, y_probs, title="Calibration Curve — Titanic Survival")
    fig_dist = plot_predictions_distribution(y_test, y_probs, title="Probability Distribution by True Class")

    # ── Feature importance ────────────────────────────────────────────────────
    lgbm_model = model.model[-1]
    preprocessor = model.model[:-1]
    try:
        feature_names = list(preprocessor[-1].columns)
    except (AttributeError, IndexError):
        feature_names = [f"feature_{i}" for i in range(lgbm_model.n_features_in_)]
    importances = lgbm_model.booster_.feature_importance(importance_type="gain").tolist()
    fig_importance = plot_feature_importance(feature_names, importances, title="LightGBM Feature Importance (Gain)")

    # ── Report structure ──────────────────────────────────────────────────────
    report_structure = {
        SectionHeader("Model Metrics", description="Cross-validation and test-set performance summary."): {
            "CV scores per fold": Table(cv_metrics_df, title="CV Scores per Fold", width=90),
            "CV summary statistics": Table(cv_summary_df, title="CV Summary Statistics", width=80),
            "Test-set metrics": Table(test_metrics_df, title="Test Set Metrics", show_index=False, width=40),
        },
        SectionHeader("Classification Quality", description="Confusion matrix and per-class metrics."): {
            "Confusion matrix": fig_cm,
            "Classification report": fig_clf_report,
        },
        SectionHeader("Discrimination", description="Ability to separate classes."): {
            "ROC AUC curve": fig_roc,
            "Precision-Recall curve": fig_pr,
        },
        SectionHeader("Calibration", description="How well predicted probabilities match true frequencies."): {
            "Calibration curve": fig_calib,
        },
        SectionHeader("Predictions", description="Distribution of predicted survival probabilities."): {
            "Probability distribution": fig_dist,
        },
        SectionHeader("Feature Importance", description="LightGBM gain-based feature importance."): {
            "Feature importance": fig_importance,
        },
    }

    meta = ReportMetaData(title="Titanic Survival — Performance Report")
    return generate_html_report(
        report_structure=report_structure,
        report_meta_data=meta,
        max_table_of_content_depth=2,
    )
