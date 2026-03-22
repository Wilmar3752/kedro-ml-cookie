from kedro.pipeline import Pipeline, node

from modelling.titanic.hypertune.nodes import (
    evaluate_titanic_model,
    extract_titanic_study,
    predict_titanic,
    train_titanic_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the hypertune pipeline.

    Nodes
    -----
    1. ``train_titanic_model``    — Optuna hyperparameter search + final fit.
    2. ``evaluate_titanic_model`` — Test-set metrics (ROC AUC, F1, …).
    3. ``predict_titanic``        — Survival predictions + probabilities.
    4. ``extract_titanic_study``  — Save the Optuna study for the reporting pipeline.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=train_titanic_model,
                inputs=["titanic_train", "params:titanic.hypertune"],
                outputs="titanic_model",
                name="train_titanic_model_node",
                tags=["hypertune", "training"],
            ),
            node(
                func=evaluate_titanic_model,
                inputs=["titanic_model", "titanic_test", "params:titanic.hypertune"],
                outputs="titanic_model_metrics",
                name="evaluate_titanic_model_node",
                tags=["hypertune", "evaluation"],
            ),
            node(
                func=predict_titanic,
                inputs=["titanic_model", "titanic_test", "params:titanic.hypertune"],
                outputs="titanic_predictions",
                name="predict_titanic_node",
                tags=["hypertune", "inference"],
            ),
            node(
                func=extract_titanic_study,
                inputs="titanic_model",
                outputs="titanic_study",
                name="extract_titanic_study_node",
                tags=["hypertune"],
            ),
        ]
    )
