import logging
import typing as tp

import pandas as pd

from packages.modelling.evaluate.metrics import compute_binary_classification_metrics
from packages.modelling.models.supervised.sklearn import BinaryClassifierSklearnPipeline

logger = logging.getLogger(__name__)


def train_titanic_model(
    df_train: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> BinaryClassifierSklearnPipeline:
    """Train the Titanic survival binary classifier.

    Instantiates a ``BinaryClassifierSklearnPipeline``, runs Optuna
    hyperparameter tuning with stratified k-fold cross-validation, then
    refits the best model on the full training set.

    The sklearn Pipeline is built entirely from the YAML config:
    ``params["pipeline"]`` defines each step via a dotted class path and kwargs.
    ``params["optuna"]`` controls the hyperparameter search.

    Args:
        df_train: Training DataFrame. Must have a default integer RangeIndex.
            All feature columns plus the target column are expected.
        params: ``titanic.hypertune`` parameter dict loaded from
            ``conf/base/parameters/modelling/hypertune.yml``.

    Returns:
        Fitted ``BinaryClassifierSklearnPipeline`` (serialised to
        ``data/06_models/titanic_model.pkl`` via the catalog).
    """
    target = params["target"]
    X_train = df_train.drop(columns=[target]).reset_index(drop=True)
    y_train = df_train[target].reset_index(drop=True)

    logger.info(
        f"Training BinaryClassifierSklearnPipeline on {len(X_train)} samples "
        f"(target='{target}', positive_rate={y_train.mean():.2%})"
    )

    pipeline = BinaryClassifierSklearnPipeline(params=params)
    pipeline.fit(X_train, y_train)

    logger.info(f"Training complete. CV scores (mean across folds):")
    for metric, value in pipeline.scores.items():
        logger.info(f"  {metric}: {value:.4f}")

    return pipeline


def evaluate_titanic_model(
    pipeline: BinaryClassifierSklearnPipeline,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> tp.Dict[str, float]:
    """Evaluate the trained model on the held-out test set.

    Computes the full binary classification metrics suite including
    ROC AUC and Brier score. The returned dict is persisted as a
    ``tracking.MetricsDataset`` in ``data/09_tracking/metrics.json``.

    Args:
        pipeline: Fitted ``BinaryClassifierSklearnPipeline``.
        df_test: Test DataFrame (same schema as ``df_train``).
        params: ``titanic.hypertune`` parameter dict.

    Returns:
        Dict mapping metric name → float value.
    """
    target = params["target"]
    probability_threshold = params.get("probability_threshold", 0.5)

    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True)

    y_score = pipeline.predict_proba(X_test)
    y_probs = y_score[:, 1]
    y_pred = (y_probs > probability_threshold).astype(int)

    test_metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_probs,
    )

    logger.info("Test set evaluation results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return test_metrics


def predict_titanic(
    pipeline: BinaryClassifierSklearnPipeline,
    df_test: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> pd.DataFrame:
    """Generate survival predictions and survival probabilities on the test set.

    Args:
        pipeline: Fitted ``BinaryClassifierSklearnPipeline``.
        df_test: Test DataFrame.
        params: ``titanic.hypertune`` parameter dict.

    Returns:
        DataFrame with columns:
        - ``survived``: Ground-truth labels.
        - ``prediction``: Binary predictions (0 / 1).
        - ``survival_probability``: Predicted probability of survival.
    """
    target = params["target"]
    probability_threshold = params.get("probability_threshold", 0.5)

    X_test = df_test.drop(columns=[target]).reset_index(drop=True)
    y_test = df_test[target].reset_index(drop=True)

    y_score = pipeline.predict_proba(X_test)
    y_probs = y_score[:, 1]
    y_pred = (y_probs > probability_threshold).astype(int)

    predictions = pd.DataFrame(
        {
            target: y_test.values,
            "prediction": y_pred,
            "survival_probability": y_probs,
        }
    )

    logger.info(
        f"Predictions generated: {len(predictions)} rows | "
        f"predicted_positive_rate={predictions['prediction'].mean():.2%}"
    )
    return predictions


def extract_titanic_study(
    pipeline: BinaryClassifierSklearnPipeline,
):
    """Extract the Optuna study from a trained pipeline.

    Args:
        pipeline: Fitted ``BinaryClassifierSklearnPipeline``.

    Returns:
        The ``optuna.Study`` object from the hyperparameter search.
    """
    study = pipeline.hypertune_results["study"]
    logger.info(
        f"Optuna study extracted — {len(study.trials)} trials, "
        f"best value: {study.best_trial.value:.4f}"
    )
    return study


def compute_cv_report(
    pipeline: BinaryClassifierSklearnPipeline,
) -> pd.DataFrame:
    """Extract the per-fold cross-validation scores DataFrame from the trained pipeline.

    This gives a detailed view of model stability across CV folds — high
    variance across folds signals overfitting or data quality issues.

    Args:
        pipeline: Fitted ``BinaryClassifierSklearnPipeline`` (must have been
            trained, so ``pipeline.df_scores`` is populated).

    Returns:
        DataFrame with one row per fold and one column per metric.
    """
    if not hasattr(pipeline, "df_scores"):
        logger.warning("Pipeline has no 'df_scores' attribute. Was it trained?")
        return pd.DataFrame()

    df = pipeline.df_scores.copy()
    logger.info(f"CV report shape: {df.shape}")
    logger.info(f"CV mean scores:\n{df.mean().to_string()}")
    return df
