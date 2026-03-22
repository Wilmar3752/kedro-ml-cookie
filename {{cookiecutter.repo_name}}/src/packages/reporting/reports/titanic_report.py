"""Titanic survival prediction — HTML model report."""

import typing as tp

import pandas as pd

from ..rendering.html.report_generation import ReportMetaData, generate_html_report
from .identifiers import SectionHeader, Table


def create_titanic_html_report(
    model_report: pd.DataFrame,
    cv_scores: pd.DataFrame,
    model_metrics: tp.Dict[str, float],
) -> str:
    """Generate a self-contained HTML report for the Titanic model.

    Sections
    --------
    1. Model Performance      — test-set metrics vs CV statistics.
    2. Cross-Validation       — per-fold scores.
    3. Training Configuration — key hyper-parameters used.

    Args:
        model_report: Output of ``generate_model_report``
            (columns: metric, test_score, cv_mean, cv_std, cv_min, cv_max).
        cv_scores: Per-fold CV metrics from ``compute_cv_report``.
        model_metrics: Test-set metrics dict from ``evaluate_titanic_model``.

    Returns:
        Self-contained HTML string.
    """
    report_structure: tp.Dict = {
        SectionHeader(
            "Model Performance",
            description="Test-set scores alongside cross-validation statistics.",
        ): {
            "Summary table": Table(
                model_report,
                precision=4,
                title="Metrics: Test Score vs Cross-Validation",
                show_index=False,
                width=85,
            ),
        },
        SectionHeader(
            "Cross-Validation Results",
            description="Per-fold scores from stratified k-fold CV.",
        ): {
            "CV scores per fold": Table(
                cv_scores.round(4),
                title="Per-Fold Cross-Validation Scores",
                width=90,
            ),
            "Key metrics": _build_metrics_summary(model_metrics),
        },
    }

    meta = ReportMetaData(title="Titanic Survival — Model Report")
    return generate_html_report(
        report_structure=report_structure,
        report_meta_data=meta,
        max_table_of_content_depth=2,
    )


def _build_metrics_summary(model_metrics: tp.Dict[str, float]) -> pd.DataFrame:
    """Convert the metrics dict to a two-column DataFrame for display."""
    return pd.DataFrame(
        {"Metric": list(model_metrics.keys()), "Value": list(model_metrics.values())}
    ).round(4)
