import copy
import logging
import os
import typing as tp

import mlflow
import mlflow.sklearn
from kedro.io.core import AbstractDataset
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_INFERENCE_ONLY_ATTRS = {
    "X_train", "y_train", "sample_weight",
    "X_test", "y_test", "y_pred", "y_score",
    "df_scores", "hypertune_results",
}


class MLflowModelStageManager(AbstractDataset):
    """Kedro dataset that logs a sklearn model to the MLflow Model Registry.

    On ``save``: logs the model via ``mlflow.sklearn.log_model``, registers it
    under ``model_name``, and transitions it to ``save_stage``. Training-time
    attributes (X_train, y_train, etc.) are stripped before logging so the
    MLflow artifact stays small.

    On ``load``: loads the model from the registry at ``load_stage``, falling
    back to ``save_stage`` if the primary stage is unavailable.

    In non-production environments the model name is prefixed with ``dev_``
    to prevent accidental promotion to production.

    Args:
        model_name: Name to register the model under in the MLflow registry.
        load_stage: Registry stage to load from (default ``"Staging"``).
        save_stage: Registry stage to transition to after logging (default ``"Staging"``).
        pyfunc_predict_fn: The predict function to expose via the pyfunc flavour.
        kedro_env: Current Kedro environment. Non-production envs prefix the
            model name with ``dev_``.
        production_envs: List of environment names considered production.
    """

    def __init__(
        self,
        model_name: str,
        load_stage: str = "Staging",
        save_stage: str = "Staging",
        pyfunc_predict_fn: str = "predict",
        kedro_env: str = "base",
        production_envs: tp.List[str] = ["prd", "prod"],
    ) -> None:
        self.model_name = model_name
        self.load_stage = load_stage
        self.save_stage = save_stage
        self.pyfunc_predict_fn = pyfunc_predict_fn
        self.kedro_env = kedro_env

        if self.kedro_env not in production_envs:
            logger.warning(
                f"Environment '{self.kedro_env}' is not a production environment. "
                f"Model '{self.model_name}' will be registered with a 'dev_' prefix."
            )
            self.model_name = f"dev_{self.model_name}"

    def _load(self) -> tp.Any:
        load_uri = f"models:/{self.model_name}/{self.load_stage}"
        save_uri = f"models:/{self.model_name}/{self.save_stage}"

        os.environ["TQDM_DISABLE"] = "1"
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
        try:
            try:
                return mlflow.pyfunc.load_model(load_uri)
            except Exception:
                logger.warning(
                    f"Could not load from {load_uri}, falling back to {save_uri}."
                )
                return mlflow.pyfunc.load_model(save_uri)
        finally:
            os.environ.pop("TQDM_DISABLE", None)
            os.environ.pop("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", None)

    def _save(self, model: tp.Any) -> None:
        model_copy = copy.copy(model)
        for attr in _INFERENCE_ONLY_ATTRS:
            model_copy.__dict__.pop(attr, None)

        metadata = {"best_params": getattr(model, "best_params", {})}

        os.environ["TQDM_DISABLE"] = "1"
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
        try:
            mlflow.sklearn.log_model(
                name=self.model_name,
                sk_model=model_copy,
                registered_model_name=self.model_name,
                pyfunc_predict_fn=self.pyfunc_predict_fn,
                metadata=metadata,
            )
        finally:
            os.environ.pop("TQDM_DISABLE", None)
            os.environ.pop("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", None)

        client = MlflowClient()
        try:
            versions = client.get_latest_versions(self.model_name, stages=["None"])
            latest_version = int(versions[0].version)
            client.transition_model_version_stage(
                name=self.model_name,
                version=latest_version,
                stage=self.save_stage,
            )
            logger.info(
                f"Model '{self.model_name}' v{latest_version} transitioned to '{self.save_stage}'."
            )
        except Exception as e:
            logger.warning(
                f"Model was logged but stage transition failed for '{self.model_name}': {e}"
            )

    def _describe(self) -> tp.Dict[str, tp.Any]:
        return {
            "model_name": self.model_name,
            "load_stage": self.load_stage,
            "save_stage": self.save_stage,
            "pyfunc_predict_fn": self.pyfunc_predict_fn,
        }
