import logging
import typing as tp

import pandas as pd

from packages.modelling.evaluate.metrics import compute_regression_metrics
from packages.modelling.models.supervised.sklearn import RegressionSklearnPipeline

logger = logging.getLogger(__name__)


def train_california_model(
    df_train: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> RegressionSklearnPipeline:
    """Train the California Housing regression model.

    Instantiates a ``RegressionSklearnPipeline``, runs Optuna hyperparameter
    tuning with k-fold cross-validation, then refits the best model on the
    full training set.

    Args:
        df_train: Training DataFrame with a default integer RangeIndex.
        params: ``california.hypertune`` parameter dict.

    Returns:
        Fitted ``RegressionSklearnPipeline``.
    """
    target = params["target"]
    X_train = df_train.drop(columns=[target]).reset_index(drop=True)
    y_train = df_train[target].reset_index(drop=True)

    logger.info(
        f"Training RegressionSklearnPipeline on {len(X_train)} samples "
        f"(target='{target}', mean={y_train.mean():.3f}, std={y_train.std():.3f})"
    )

    pipeline = RegressionSklearnPipeline(params=params)
    pipeline.fit(X_train, y_train)

    logger.info("Training complete. CV scores (mean across folds):")
    for metric, value in pipeline.scores.items():
        logger.info(f"  {metric}: {value:.4f}")

    return pipeline


def evaluate_california_model(
    pipeline: RegressionSklearnPipeline,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> tp.Dict[str, float]:
    """Evaluate the trained model on the held-out test set.

    Args:
        pipeline: Fitted ``RegressionSklearnPipeline``.
        df_test: Test DataFrame.
        params: ``california.hypertune`` parameter dict.

    Returns:
        Dict mapping metric name → float value.
    """
    target = params["target"]
    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True)

    y_pred = pipeline.predict(X_test)
    test_metrics = compute_regression_metrics(y_true=y_test, y_pred=y_pred)

    logger.info("Test set evaluation results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return test_metrics


def predict_california(
    pipeline: RegressionSklearnPipeline,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> pd.DataFrame:
    """Generate predictions on the test set.

    Args:
        pipeline: Fitted ``RegressionSklearnPipeline``.
        df_test: Test DataFrame.
        params: ``california.hypertune`` parameter dict.

    Returns:
        DataFrame with columns ``{target}`` (ground truth) and ``prediction``.
    """
    target = params["target"]
    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True)

    y_pred = pipeline.predict(X_test)

    predictions = pd.DataFrame({target: y_test.values, "prediction": y_pred})
    logger.info(
        f"Predictions generated: {len(predictions)} rows | "
        f"mean_prediction={y_pred.mean():.3f}"
    )
    return predictions


def extract_california_study(
    pipeline: RegressionSklearnPipeline,
):
    """Extract the Optuna study from a trained pipeline.

    Args:
        pipeline: Fitted ``RegressionSklearnPipeline``.

    Returns:
        The ``optuna.Study`` object from the hyperparameter search.
    """
    study = pipeline.hypertune_results["study"]
    logger.info(
        f"Optuna study extracted — {len(study.trials)} trials, "
        f"best value: {study.best_trial.value:.4f}"
    )
    return study
