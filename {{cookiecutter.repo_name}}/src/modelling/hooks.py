import logging
from typing import Any

import mlflow
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)

_METRICS_SUFFIXES = ("_metrics", "_scores", "_score")


class MLflowHook:
    """Kedro hook that logs pipeline runs to MLflow automatically.

    - One MLflow run per ``kedro run`` invocation.
    - Experiment name = pipeline name (or "default" when running all pipelines).
    - Any node that returns a ``dict[str, float]`` whose name ends with
      ``_metrics``, ``_scores``, or ``_score`` has its values logged as metrics.
    - The flat ``params:*`` inputs of every node are logged as run parameters
      (first node that carries them wins; duplicates are silently skipped).
    """

    @hook_impl
    def before_pipeline_run(self, run_params: dict, pipeline, catalog) -> None:
        pipeline_name = run_params.get("pipeline_name") or "default"
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(pipeline_name)
        mlflow.start_run(run_name=run_params.get("run_id"))
        mlflow.set_tags(
            {
                "kedro_pipeline": pipeline_name,
                "kedro_env": run_params.get("env", "base"),
            }
        )
        logger.info(f"MLflow run started — experiment: '{pipeline_name}'")

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        is_async: bool,
        run_id: str,
    ) -> None:
        # Log flat numeric params (skip already-logged keys silently)
        for name, value in inputs.items():
            if name.startswith("params:") and isinstance(value, dict):
                flat = _flatten_dict(value)
                for k, v in flat.items():
                    if isinstance(v, (int, float, str, bool)):
                        try:
                            mlflow.log_param(k, v)
                        except mlflow.exceptions.MlflowException:
                            pass  # duplicate key — already logged

        # Log metrics from nodes whose output name ends with a metrics suffix
        for name, value in outputs.items():
            if any(name.endswith(s) for s in _METRICS_SUFFIXES) and isinstance(
                value, dict
            ):
                numeric = {
                    k: float(v)
                    for k, v in value.items()
                    if isinstance(v, (int, float))
                }
                if numeric:
                    mlflow.log_metrics(numeric)
                    logger.info(f"MLflow logged metrics from '{name}': {numeric}")

    @hook_impl
    def after_pipeline_run(self, run_params: dict, pipeline, catalog) -> None:
        mlflow.end_run()
        logger.info("MLflow run ended.")

    @hook_impl
    def on_pipeline_error(self, error: Exception, run_params: dict, pipeline, catalog) -> None:
        mlflow.end_run(status="FAILED")
        logger.warning(f"MLflow run ended with FAILED status due to: {error}")


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
