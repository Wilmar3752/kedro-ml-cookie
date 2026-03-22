import datetime
import logging
import typing as tp
from copy import deepcopy

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from packages.python_utils.load.object_injection import load_object
from packages.python_utils.typing import Tensor
from packages.modelling.evaluate.metrics import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_regression_metrics,
)
from packages.reproducibility.set_seed import seed_file
from packages.modelling.models.mlflow.metrics import MlflowTransformations
from packages.modelling.models.preprocessors.auto_configure import AutoConfigFeaturePreprocessors
from packages.modelling.models.supervised.shap_values import BaseShapComputation

seed_file()

logger = logging.getLogger(__name__)


class BaseSklearnCompatibleModel(
    BaseEstimator,
    MlflowTransformations,
    BaseShapComputation,
    AutoConfigFeaturePreprocessors,
):
    """Base class for sklearn-compatible ML models with Optuna hyperparameter tuning.

    Combines a dynamically-built sklearn ``Pipeline`` (defined entirely in YAML)
    with an Optuna study for cross-validated hyperparameter search. Subclasses
    implement only the ``evaluate`` method to provide task-specific metrics.

    Args:
        params: Full parameter dict loaded from the Kedro parameter catalog. Expected keys:
            - ``target``: Name of the target column.
            - ``features``: Dict with ``columns`` list.
            - ``pipeline``: Ordered dict of pipeline steps (class + kwargs).
            - ``optuna``: Optuna study/optimize/sampler/pruner configuration.
            - ``cv_strategy``: Cross-validation strategy configuration.
            - ``cv_score``: Dict with ``metric_to_optimize`` key.
            - ``probability_threshold`` (optional): Binary decision threshold.
            - ``early_stopping`` (optional): Pruning configuration.
            - ``monotonic_constraints`` (optional): Feature monotonicity constraints.

    Attributes:
        is_fitted: Whether ``fit`` has been called.
        train_date: ISO-format timestamp of the last ``fit`` call.
        model: The final fitted sklearn Pipeline.
        best_params: Best hyperparameter dict found by Optuna.
        scores: Mean cross-validation metrics from the final backtest.
        df_scores: Per-fold cross-validation metrics DataFrame.
    """

    def __init__(self, params: tp.Dict[str, tp.Any]) -> "BaseSklearnCompatibleModel":
        self.params = params
        self.is_fitted = False
        self.target = params.get("target", "target")
        self.features = params.get("features", [])
        self.train_date = None
        # Pipeline step order is determined by the YAML key order
        self.pipeline_order = list(self.params["pipeline"].keys())

    def get_params(self, deep: bool = True) -> tp.Dict[str, tp.Any]:
        """Return the raw ``params`` dict (sklearn estimator interface)."""
        return self.params

    # ------------------------------------------------------------------
    # Public fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> "BaseSklearnCompatibleModel":
        """Hyperparameter-tune and fit the model.

        Steps:
        1. Auto-configure preprocessing variable lists from the data.
        2. Run Optuna cross-validated hyperparameter search.
        3. Build a pipeline with the best params and fit it on all data.

        Args:
            X: Feature DataFrame with a default integer RangeIndex.
            y: Target Series aligned with ``X``.
            sample_weight: Optional per-sample weights.

        Returns:
            self
        """
        seed_file()

        # Inject feature variable lists into preprocessing steps
        self.params = self.configure_feature_processing_params(X=X, params=self.params)

        self.train_date = datetime.datetime.now().isoformat()

        # Hyperparameter search
        self.hypertune_results = self.hypertune_on_validation_framework(
            X=X, y=y, sample_weight=sample_weight
        )
        self.best_params = self.hypertune_results["best_trial_params"]

        # Build + fit final model with best hyperparameters
        self.model = self.build_model_pipeline(self.best_params)
        self.model = self.partial_fit(
            estimator=deepcopy(self.model),
            X_train=X,
            y_train=y,
            sample_weight=sample_weight,
        )

        self.is_fitted = True
        self.X_train = X
        self.y_train = y
        self.sample_weight = sample_weight
        return self

    def refit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> "BaseSklearnCompatibleModel":
        """Refit using previously found best hyperparameters (no new Optuna search).

        Useful for re-training on an updated dataset without repeating the
        expensive hyperparameter search.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            sample_weight: Optional per-sample weights.

        Returns:
            self
        """
        self.model = self.build_model_pipeline(self.best_params)
        self.model = self.partial_fit(
            estimator=deepcopy(self.model),
            X_train=X,
            y_train=y,
            sample_weight=sample_weight,
        )
        return self

    def partial_fit(
        self,
        estimator: Pipeline,
        X_train: Tensor,
        y_train: Tensor,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> Pipeline:
        """Fit a pipeline, optionally applying monotonic constraints.

        When ``monotonic_constraints`` are specified in ``params``, the
        preprocessor is first fit separately so that feature names can be
        resolved, and a constraint vector is injected into the model before
        the final fit.

        Args:
            estimator: An unfitted sklearn Pipeline.
            X_train: Training features.
            y_train: Training labels.
            sample_weight: Optional per-sample weights.

        Returns:
            Fitted Pipeline.
        """
        constraints = self.params.get("monotonic_constraints", {})

        if not constraints:
            return estimator.fit(
                X_train, y_train, model__sample_weight=sample_weight
            )

        # With constraints: fit preprocessor first to get feature names
        preprocessor = estimator[:-1]
        model = estimator[-1]
        preprocessor = preprocessor.fit(X_train, y_train)
        feature_names = preprocessor[-1].columns

        constraint_vector = [constraints.get(col, 0) for col in feature_names]
        model = model.set_params(monotone_constraints=constraint_vector)

        final_pipeline = Pipeline(
            [*preprocessor.steps, ("model", model)], memory=None
        )
        return final_pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predictions.
        """
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def build_model_pipeline(self, params: tp.Dict[str, tp.Any]) -> Pipeline:
        """Instantiate an sklearn Pipeline from the ``pipeline`` config section.

        Each key in ``params["pipeline"]`` becomes a named step. The class is
        loaded dynamically from the dotted path in ``class``, and ``kwargs``
        are passed to the constructor.

        Args:
            params: Hyperparameter dict (may differ from ``self.params`` during
                cross-validation with injected trial values).

        Returns:
            Unfitted sklearn Pipeline.
        """
        return Pipeline(
            steps=[
                (step_name, load_object(params["pipeline"][step_name]))
                for step_name in self.pipeline_order
            ],
            memory=None,
        )

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def hypertune_on_validation_framework(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> dict:
        """Run Optuna hyperparameter search with cross-validation.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            sample_weight: Optional per-sample weights.

        Returns:
            Dict with keys ``study``, ``best_trial_params``, and
            ``cross_validation_metrics``.
        """
        kwargs_study = self.params["optuna"]["kwargs_study"]
        kwargs_optimize = self.params["optuna"]["kwargs_optimize"]
        sampler = load_object(self.params["optuna"]["sampler"])
        pruner = load_object(self.params["optuna"]["pruner"])

        objective_kwargs = {
            "X": deepcopy(X),
            "y": deepcopy(y),
            "params": deepcopy(self.params),
            "sample_weight": sample_weight,
        }

        objective = lambda trial: self.hypertune_on_validation_framework_objective_function(  # noqa: E731
            trial, **deepcopy(objective_kwargs)
        )

        study = optuna.create_study(sampler=sampler, pruner=pruner, **kwargs_study)
        study.optimize(objective, **kwargs_optimize)

        best_trial = study.best_trial
        best_estimator_params = study.trials[best_trial.number].user_attrs[
            "estimator_params"
        ]

        # Final backtest with best params to get stable CV scores
        final_estimator = self.build_model_pipeline(best_estimator_params)
        final_score = self.cross_validate_estimator(
            final_estimator, X, y, deepcopy(best_estimator_params),
            sample_weight=sample_weight,
        )

        scoring_metric = best_estimator_params.get("cv_score", {}).get("metric_to_optimize", "")
        logger.info(f"Best trial {scoring_metric}: {final_score:.4f}")

        cross_validation_metrics = self.format_metrics_dict(self.scores)
        return {
            "study": study,
            "best_trial_params": best_estimator_params,
            "cross_validation_metrics": cross_validation_metrics,
        }

    def hypertune_on_validation_framework_objective_function(
        self,
        trial: optuna.Trial,
        **kwargs,
    ) -> float:
        """Optuna objective: inject trial params → build pipeline → cross-validate.

        Args:
            trial: Current Optuna trial.
            **kwargs: Must include ``X``, ``y``, ``params``, ``sample_weight``.

        Returns:
            Cross-validation score for this trial.
        """
        params = kwargs.get("params", {})
        X = kwargs.get("X")
        y = kwargs.get("y")
        sample_weight = kwargs.get("sample_weight")

        new_params = self.inject_trial_parameter(deepcopy(params), trial)
        trial.set_user_attr(key="estimator_params", value=new_params)

        estimator = self.build_model_pipeline(new_params)
        early_stopping = new_params.get("early_stopping", {})

        if early_stopping.get("enabled", False):
            score = self.cross_validate_estimator_with_pruning(
                estimator, X, y, deepcopy(new_params),
                trial=trial, sample_weight=sample_weight,
            )
        else:
            score = self.cross_validate_estimator(
                estimator, X, y, deepcopy(new_params), sample_weight=sample_weight
            )

        return score

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate_estimator(
        self,
        estimator: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """K-fold cross-validate ``estimator`` and store per-fold metrics.

        Args:
            estimator: Unfitted sklearn Pipeline.
            X: Feature DataFrame with integer RangeIndex.
            y: Target Series with integer RangeIndex.
            params: Parameter dict for this trial.
            sample_weight: Optional per-sample weights.

        Returns:
            Cross-validation score on the optimisation metric.
        """
        cv_strategy = load_object(params["cv_strategy"])
        scoring_metric = params["cv_score"]["metric_to_optimize"]
        target_col = params["target"]
        probability_threshold = params.get("probability_threshold", 0.5)
        is_multiclass = isinstance(self, MultiClassClassifierSklearnPipeline)

        list_scores = []
        results = []

        for _, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y)):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            train_sw, test_sw = self._split_sample_weights(
                sample_weight, train_idx, test_idx
            )

            estimator = self.partial_fit(estimator, X_train, y_train, train_sw)
            y_pred, y_score = self._predict_with_threshold(
                estimator, X_test, is_multiclass, probability_threshold
            )

            scores = self.evaluate(
                y_pred=y_pred, y_true=y_test,
                y_score=y_score, sample_weight=test_sw,
            )
            list_scores.append(pd.DataFrame.from_dict(scores, orient="index").T)
            results.append(scores[scoring_metric])

        self.df_scores = pd.concat(list_scores, axis=0).reset_index(drop=True)
        self.y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=["prediction"])
        self.y_test = pd.DataFrame(y_test, index=y_test.index, columns=[target_col])
        self._store_score_if_available(y_score, y_test)
        self.X_test = X_test
        self.scores = self.df_scores.mean().to_dict()

        return scores[scoring_metric]

    def cross_validate_estimator_with_pruning(
        self,
        estimator: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict,
        trial: optuna.Trial,
        sample_weight: tp.Optional[np.ndarray] = None,
    ) -> float:
        """Cross-validate with Optuna intermediate reporting for early stopping.

        Prunes unpromising trials after ``min_folds_before_pruning`` folds to
        save compute time.

        Args:
            estimator: Unfitted sklearn Pipeline.
            X: Feature DataFrame.
            y: Target Series.
            params: Parameter dict for this trial.
            trial: Current Optuna trial (used for reporting and pruning).
            sample_weight: Optional per-sample weights.

        Returns:
            Mean cross-validation score across all completed folds.
        """
        cv_strategy = load_object(params["cv_strategy"])
        scoring_metric = params["cv_score"]["metric_to_optimize"]
        target_col = params["target"]
        probability_threshold = params.get("probability_threshold", 0.5)
        is_multiclass = isinstance(self, MultiClassClassifierSklearnPipeline)
        min_folds = params.get("early_stopping", {}).get("min_folds_before_pruning", 1)

        trial.set_user_attr("fold_scores", [])

        list_scores = []
        fold_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y)):
            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            train_sw, test_sw = self._split_sample_weights(
                sample_weight, train_idx, test_idx
            )

            estimator = self.partial_fit(estimator, X_train, y_train, train_sw)
            y_pred, y_score = self._predict_with_threshold(
                estimator, X_test, is_multiclass, probability_threshold
            )

            scores = self.evaluate(
                y_pred=y_pred, y_true=y_test,
                y_score=y_score, sample_weight=test_sw,
            )
            list_scores.append(pd.DataFrame.from_dict(scores, orient="index").T)

            fold_score = scores[scoring_metric]
            fold_scores.append(fold_score)
            self._update_trial_attributes(trial, fold_score, scores)
            self._check_and_prune_trial(trial, fold_idx, fold_scores, min_folds)

        self.df_scores = pd.concat(list_scores, axis=0).reset_index(drop=True)
        self.y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=["prediction"])
        self.y_test = pd.DataFrame(y_test, index=y_test.index, columns=[target_col])
        self._store_score_if_available(y_score, y_test)
        self.X_test = X_test
        self.scores = self.df_scores.mean().to_dict()

        final_score = float(np.mean(fold_scores))
        logger.debug(f"Trial {trial.number} mean CV score: {final_score:.4f}")
        return final_score

    # ------------------------------------------------------------------
    # Optuna utilities
    # ------------------------------------------------------------------

    def inject_trial_parameter(
        self, d: tp.Dict[str, tp.Any], trial: optuna.Trial
    ) -> tp.Dict[str, tp.Any]:
        """Replace ``trial.*`` strings with actual Optuna suggestions.

        Traverses ``d`` recursively and evaluates any string value that
        contains ``"trial."`` using the current trial object.

        Args:
            d: Parameter dict to modify in-place.
            trial: Current Optuna trial.

        Returns:
            Updated dict with trial suggestions resolved.

        Example:
            YAML value ``"trial.suggest_int('n_estimators', 50, 500)"``
            becomes the integer sampled by the trial.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                self.inject_trial_parameter(v, trial)
            elif isinstance(v, str) and "trial." in v:
                d[k] = eval(v, {"trial": trial})  # noqa: S307
        return d

    def save_best_model(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Optuna callback to persist best trial params into the study.

        Args:
            study: Current Optuna study.
            trial: Current trial being evaluated.
        """
        if study.best_trial.number == trial.number:
            study.set_user_attr(
                key="estimator_params", value=trial.user_attrs["estimator_params"]
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_sample_weights(
        self,
        sample_weight: tp.Optional[np.ndarray],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple:
        if sample_weight is None:
            return None, None
        return sample_weight[train_idx], sample_weight[test_idx]

    def _predict_with_threshold(
        self,
        estimator: Pipeline,
        X_test: pd.DataFrame,
        is_multiclass: bool,
        probability_threshold: float,
    ) -> tuple:
        if is_multiclass:
            y_pred = estimator.predict(X_test)
            y_score = (
                estimator.predict_proba(X_test)
                if hasattr(estimator, "predict_proba")
                else None
            )
            return y_pred, y_score

        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(X_test)
            y_probs = y_score[:, 1]
            y_pred = np.where(y_probs > probability_threshold, 1, 0)
            return y_pred, y_score

        return estimator.predict(X_test), None

    def _update_trial_attributes(
        self, trial: optuna.Trial, fold_score: float, scores: dict
    ) -> None:
        current = trial.user_attrs.get("fold_scores", [])
        current.append(fold_score)
        trial.set_user_attr("fold_scores", current)

    def _check_and_prune_trial(
        self,
        trial: optuna.Trial,
        fold_idx: int,
        fold_scores: list,
        min_folds_before_pruning: int,
    ) -> None:
        if fold_idx < min_folds_before_pruning:
            return
        intermediate_score = float(np.mean(fold_scores))
        trial.report(intermediate_score, fold_idx)
        if trial.should_prune():
            logger.info(
                f"Trial {trial.number} pruned at fold {fold_idx} "
                f"(score={intermediate_score:.4f})"
            )
            raise optuna.TrialPruned()

    def _store_score_if_available(self, y_score, y_test) -> None:
        if y_score is not None:
            self.y_score = pd.DataFrame(y_score, index=y_test.index)

    def _validate_pandas_dataframe(self, array: tp.Any) -> pd.DataFrame:
        if not isinstance(array, pd.DataFrame):
            return pd.DataFrame(array, columns=[self.target]).reset_index(drop=True)
        return deepcopy(array).reset_index(drop=True)


# ======================================================================
# Concrete subclasses
# ======================================================================


class RegressionSklearnPipeline(BaseSklearnCompatibleModel, RegressorMixin):
    """Regression pipeline with Optuna hyperparameter tuning.

    Uses ``compute_regression_metrics`` for evaluation.
    """

    def __init__(self, params: dict) -> "RegressionSklearnPipeline":
        super().__init__(params)

    def evaluate(
        self,
        y_true,
        y_pred,
        y_score=None,
        sample_weight=None,
    ) -> dict:
        return compute_regression_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )


class BinaryClassifierSklearnPipeline(BaseSklearnCompatibleModel, ClassifierMixin):
    """Binary classification pipeline with Optuna hyperparameter tuning.

    Uses ``compute_binary_classification_metrics`` for evaluation.
    Supports probability thresholding via ``params["probability_threshold"]``.
    """

    def __init__(self, params: dict) -> "BinaryClassifierSklearnPipeline":
        super().__init__(params)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities from the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape ``(n_samples, 2)`` with class probabilities.

        Raises:
            AttributeError: If the model has neither ``predict_proba`` nor ``predict``.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError(
            "The fitted model has neither a 'predict_proba' nor a 'predict' method."
        )

    def evaluate(
        self,
        y_true,
        y_pred,
        y_score=None,
        sample_weight=None,
    ) -> dict:
        if y_score is not None:
            y_score = y_score[:, 1]
            if np.isnan(y_score).any():
                y_score = np.nan_to_num(y_score, nan=0.0)
                logger.warning("NaN probabilities detected — replaced with 0.")
        return compute_binary_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            sample_weight=sample_weight,
        )


class MultiClassClassifierSklearnPipeline(BaseSklearnCompatibleModel, ClassifierMixin):
    """Multi-class classification pipeline with Optuna hyperparameter tuning.

    Uses ``compute_multiclass_classification_metrics`` for evaluation.
    """

    def __init__(self, params: dict) -> "MultiClassClassifierSklearnPipeline":
        super().__init__(params)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities from the fitted pipeline.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of class probabilities.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        raise AttributeError(
            "The fitted model has neither a 'predict_proba' nor a 'predict' method."
        )

    def evaluate(
        self,
        y_true,
        y_pred,
        y_score=None,
        sample_weight=None,
    ) -> dict:
        return compute_multiclass_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )
