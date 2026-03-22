from kedro.pipeline import Pipeline, node

from modelling.california.hypertune.nodes import (
    evaluate_california_model,
    extract_california_study,
    predict_california,
    train_california_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the California Housing hypertune pipeline.

    Nodes
    -----
    1. ``train_california_model``    — Optuna hyperparameter search + final fit.
    2. ``evaluate_california_model`` — Test-set regression metrics.
    3. ``predict_california``        — House value predictions on test set.
    4. ``extract_california_study``  — Save the Optuna study for reporting.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=train_california_model,
                inputs=["california_train", "params:california.hypertune"],
                outputs="california_model",
                name="train_california_model_node",
                tags=["california", "training"],
            ),
            node(
                func=evaluate_california_model,
                inputs=["california_model", "california_test", "params:california.hypertune"],
                outputs="california_model_metrics",
                name="evaluate_california_model_node",
                tags=["california", "evaluation"],
            ),
            node(
                func=predict_california,
                inputs=["california_model", "california_test", "params:california.hypertune"],
                outputs="california_predictions",
                name="predict_california_node",
                tags=["california", "inference"],
            ),
            node(
                func=extract_california_study,
                inputs="california_model",
                outputs="california_study",
                name="extract_california_study_node",
                tags=["california"],
            ),
        ]
    )
