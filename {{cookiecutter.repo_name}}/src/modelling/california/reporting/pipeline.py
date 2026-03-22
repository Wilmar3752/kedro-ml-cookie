from kedro.pipeline import Pipeline, node

from modelling.california.reporting.nodes import (
    create_california_hypertune_report,
    create_california_interpretability_report,
    create_california_performance_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the California Housing reporting pipeline.

    Generates three self-contained HTML reports — no image files.

    Nodes
    -----
    1. ``create_california_performance_report``     — Actual vs Predicted,
       residuals, distribution comparison, cumulative error, feature importance.
    2. ``create_california_interpretability_report`` — SHAP beeswarm, SHAP
       importance, SHAP dependence plots, LightGBM importance.
    3. ``create_california_hypertune_report``       — Optuna history, timeline,
       parameter importance, slice plots, parallel coordinates.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=create_california_performance_report,
                inputs=["california_model", "california_test", "params:california.hypertune"],
                outputs="report_california_performance",
                name="create_california_performance_report_node",
                tags=["california", "reporting"],
            ),
            node(
                func=create_california_interpretability_report,
                inputs=["california_model", "california_train", "params:california.hypertune"],
                outputs="report_california_interpretability",
                name="create_california_interpretability_report_node",
                tags=["california", "reporting"],
            ),
            node(
                func=create_california_hypertune_report,
                inputs=["california_study", "params:california.hypertune"],
                outputs="report_california_hypertune",
                name="create_california_hypertune_report_node",
                tags=["california", "reporting"],
            ),
        ]
    )
