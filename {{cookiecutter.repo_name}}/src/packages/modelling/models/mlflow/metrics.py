import logging
import typing as tp

logger = logging.getLogger(__name__)


class MlflowTransformations:
    """Mixin that converts a flat metrics dict into MLflow-compatible format.

    MLflow's ``log_metrics`` expects a dict of ``{name: value}`` pairs. For
    step-based metrics (lists) it expects ``{name: [{"value": v, "step": i}]}``.
    This mixin provides ``format_metrics_dict`` to perform that transformation
    so subclasses can pass cross-validation results directly to MLflow.
    """

    def format_metrics_dict(
        self, output_dict: tp.Dict[str, tp.Any]
    ) -> tp.Dict[str, tp.Union[float, tp.List[float]]]:
        """Transform a metrics dict into MLflow-compatible format.

        Args:
            output_dict: Dict whose values are either a scalar or a list of scalars.

        Returns:
            Dict where each value is ``{"value": v, "step": i}`` (scalar) or a
            list of such dicts (list).

        Example:
            >>> format_metrics_dict({"acc": 0.9, "f1": [0.8, 0.85]})
            {
                "acc": {"value": 0.9, "step": 1},
                "f1": [{"value": 0.8, "step": 1}, {"value": 0.85, "step": 2}],
            }
        """
        transformed_dict = {}
        for key, value in output_dict.items():
            if isinstance(value, list):
                transformed_value = [
                    {"value": v, "step": i + 1} for i, v in enumerate(value)
                ]
            else:
                transformed_value = {"value": value, "step": 1}
            transformed_dict[key] = transformed_value
        return transformed_dict
