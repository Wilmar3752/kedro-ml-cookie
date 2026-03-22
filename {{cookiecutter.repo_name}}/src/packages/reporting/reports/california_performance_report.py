"""California Housing — regression performance report (HTML)."""
import typing as tp

import numpy as np
import pandas as pd

from ..charts.performance._feature_importance import plot_feature_importance
from ..charts.performance._regression_plots import (
    plot_actual_vs_predicted,
    plot_cumulative_error,
    plot_distribution_comparison,
    plot_prediction_error_colored,
    plot_residual_distribution,
    plot_residuals,
)
from ..rendering.html.report_generation import ReportMetaData, generate_html_report
from .identifiers import SectionHeader, Table


def create_california_performance_report(
    model,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> str:
    """Generate a self-contained HTML performance report for the California Housing model.

    Sections
    --------
    1. Model Metrics       — CV scores per fold + test-set summary table.
    2. Regression Quality  — Actual vs Predicted, error-colored scatter.
    3. Residuals           — Residuals vs Predicted, residual histogram.
    4. Distributions       — Actual vs Predicted distribution overlay.
    5. Cumulative Error    — Cumulative sorted absolute error curve.
    6. Feature Importance  — LightGBM gain-based importance.

    Args:
        model: Fitted ``RegressionSklearnPipeline``.
        df_test: Test DataFrame including the target column.
        params: ``california.hypertune`` parameter dict (must contain ``target``).

    Returns:
        Self-contained HTML string.
    """
    target = params["target"]

    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True).to_numpy()
    y_pred = model.predict(X_test)

    # ── Model metrics from cross-validation ──────────────────────────────────
    cv_metrics_df = model.df_scores.T.copy()
    cv_metrics_df.columns = [f"fold_{i + 1}" for i in cv_metrics_df.columns]
    cv_summary_df = cv_metrics_df.T.describe().T.round(4)

    from packages.modelling.evaluate.metrics import compute_regression_metrics
    test_metrics = compute_regression_metrics(y_test, y_pred)
    test_metrics_df = pd.DataFrame(
        {"Metric": list(test_metrics.keys()), "Test Score": list(test_metrics.values())}
    ).round(4)

    # ── Plotly figures ────────────────────────────────────────────────────────
    fig_avp = plot_actual_vs_predicted(y_test, y_pred, title="Actual vs Predicted — California Housing")
    fig_avp_colored = plot_prediction_error_colored(y_test, y_pred, title="Actual vs Predicted (colored by |error|)")
    fig_residuals = plot_residuals(y_test, y_pred, title="Residuals vs Predicted")
    fig_residual_dist = plot_residual_distribution(y_test, y_pred, title="Residual Distribution")
    fig_dist = plot_distribution_comparison(y_test, y_pred, title="Actual vs Predicted Distribution")
    fig_cumulative = plot_cumulative_error(y_test, y_pred, title="Cumulative Absolute Error")

    # ── Feature importance ────────────────────────────────────────────────────
    lgbm_model = model.model[-1]
    preprocessor = model.model[:-1]
    try:
        feature_names = list(preprocessor[-1].columns)
    except (AttributeError, IndexError):
        feature_names = [f"feature_{i}" for i in range(lgbm_model.n_features_in_)]
    importances = lgbm_model.booster_.feature_importance(importance_type="gain").tolist()
    fig_importance = plot_feature_importance(
        feature_names, importances, title="LightGBM Feature Importance (Gain)"
    )

    # ── Report structure ──────────────────────────────────────────────────────
    report_structure = {
        SectionHeader("Model Metrics", description="Cross-validation and test-set performance summary."): {
            "CV scores per fold": Table(cv_metrics_df, title="CV Scores per Fold", width=90),
            "CV summary statistics": Table(cv_summary_df, title="CV Summary Statistics", width=80),
            "Test-set metrics": Table(test_metrics_df, title="Test Set Metrics", show_index=False, width=40),
        },
        SectionHeader("Regression Quality", description="How well predictions track actual values."): {
            "Actual vs Predicted": fig_avp,
            "Error magnitude": fig_avp_colored,
        },
        SectionHeader("Residuals", description="Pattern and distribution of prediction errors."): {
            "Residuals vs Predicted": fig_residuals,
            "Residual distribution": fig_residual_dist,
        },
        SectionHeader("Distributions", description="Comparison of actual and predicted value distributions."): {
            "Actual vs Predicted distribution": fig_dist,
        },
        SectionHeader("Cumulative Error", description="What fraction of total error comes from the hardest predictions."): {
            "Cumulative absolute error": fig_cumulative,
        },
        SectionHeader("Feature Importance", description="LightGBM gain-based feature importance."): {
            "Feature importance": fig_importance,
        },
    }

    meta = ReportMetaData(title="California Housing — Performance Report")
    return generate_html_report(
        report_structure=report_structure,
        report_meta_data=meta,
        max_table_of_content_depth=2,
    )
