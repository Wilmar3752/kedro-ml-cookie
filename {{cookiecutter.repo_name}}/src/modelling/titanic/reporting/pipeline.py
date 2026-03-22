from kedro.pipeline import Pipeline, node

from modelling.titanic.reporting.nodes import (
    create_titanic_hypertune_report,
    create_titanic_interpretability_report,
    create_titanic_performance_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the reporting pipeline.

    Generates three self-contained HTML reports — no image files.

    Nodes
    -----
    1. ``create_titanic_performance_report``     — ROC, PR, confusion matrix,
       calibration, predictions distribution, feature importance.
    2. ``create_titanic_interpretability_report`` — SHAP beeswarm, SHAP importance,
       SHAP dependence plots, LightGBM importance.
    3. ``create_titanic_hypertune_report``       — Optuna history, timeline,
       parameter importance, slice plots, parallel coordinates.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=create_titanic_performance_report,
                inputs=["titanic_model", "titanic_test", "params:titanic.hypertune"],
                outputs="report_titanic_performance",
                name="create_titanic_performance_report_node",
                tags=["reporting"],
            ),
            node(
                func=create_titanic_interpretability_report,
                inputs=["titanic_model", "titanic_train", "params:titanic.hypertune"],
                outputs="report_titanic_interpretability",
                name="create_titanic_interpretability_report_node",
                tags=["reporting"],
            ),
            node(
                func=create_titanic_hypertune_report,
                inputs=["titanic_study", "params:titanic.hypertune"],
                outputs="report_titanic_hypertune",
                name="create_titanic_hypertune_report_node",
                tags=["reporting"],
            ),
        ]
    )
