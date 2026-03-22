"""California Housing — model interpretability report (HTML)."""
import typing as tp

import matplotlib
import numpy as np
import pandas as pd

from ..charts.interpretability._shap_plots import (
    plot_shap_beeswarm,
    plot_shap_dependence,
    plot_shap_feature_importance,
)
from ..charts.performance._feature_importance import plot_feature_importance
from ..rendering.html.report_generation import ReportMetaData, generate_html_report
from .identifiers import SectionHeader

matplotlib.use("Agg")

SHAP_SAMPLE_SIZE = 500
SHAP_TOP_FEATURES = 6


def create_california_interpretability_report(
    model,
    df_train: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> str:
    """Generate a self-contained HTML interpretability report for the California Housing model.

    Sections
    --------
    1. SHAP Overview      — Beeswarm plot (global feature impact).
    2. SHAP Importance    — Mean absolute SHAP value per feature (Plotly).
    3. SHAP Dependencies  — Per-feature dependence plots for top features.
    4. Model Importance   — LightGBM gain-based importance (Plotly).

    Args:
        model: Fitted ``RegressionSklearnPipeline``.
        df_train: Training DataFrame (used for SHAP computation).
        params: ``california.hypertune`` parameter dict (must contain ``target``).

    Returns:
        Self-contained HTML string.
    """
    target = params["target"]
    shap_sample_size = params.get("shap_sample_size", SHAP_SAMPLE_SIZE)

    X_train = df_train.drop(columns=[target]).reset_index(drop=True)

    # ── Compute SHAP values ───────────────────────────────────────────────────
    from packages.modelling.models.supervised.shap_values import BaseShapComputation
    shap_computer = BaseShapComputation()
    shap_dict = shap_computer.compute_shap_values(
        model_pipeline=model.model,
        df=X_train,
        sample_size=shap_sample_size,
    )
    shap_values = shap_dict["shap_values"]
    preprocessor = shap_dict["preprocessor"]
    estimator = shap_dict["estimator"]

    # ── Feature names ─────────────────────────────────────────────────────────
    try:
        feature_names = list(preprocessor[-1].columns)
    except (AttributeError, IndexError):
        feature_names = list(shap_values.feature_names or [])

    # ── Charts ────────────────────────────────────────────────────────────────
    fig_beeswarm = plot_shap_beeswarm(shap_values, max_display=20, title="SHAP Values — California Housing")
    fig_shap_importance = plot_shap_feature_importance(shap_values, top_n=20, title="Mean |SHAP| per Feature")

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_features = [
        feature_names[i]
        for i in np.argsort(mean_abs)[::-1][:SHAP_TOP_FEATURES]
        if i < len(feature_names)
    ]
    dependence_section = {
        f"SHAP dependence — {feat}": plot_shap_dependence(shap_values, feat)
        for feat in top_features
    }

    importances = estimator.booster_.feature_importance(importance_type="gain").tolist()
    fig_model_importance = plot_feature_importance(
        feature_names, importances, title="LightGBM Feature Importance (Gain)"
    )

    # ── Report structure ──────────────────────────────────────────────────────
    report_structure = {
        SectionHeader("SHAP Overview", description="Global view of feature contributions across all samples."): {
            "Beeswarm plot": fig_beeswarm,
        },
        SectionHeader("SHAP Importance", description="Mean absolute SHAP value — average impact on model output."): {
            "Mean |SHAP| importance": fig_shap_importance,
        },
        SectionHeader("SHAP Dependencies", description=f"How the top {SHAP_TOP_FEATURES} features affect SHAP values."): dependence_section,
        SectionHeader("Model Feature Importance", description="LightGBM gain-based importance from the booster."): {
            "LightGBM gain importance": fig_model_importance,
        },
    }

    meta = ReportMetaData(title="California Housing — Interpretability Report")
    return generate_html_report(
        report_structure=report_structure,
        report_meta_data=meta,
        max_table_of_content_depth=2,
    )
