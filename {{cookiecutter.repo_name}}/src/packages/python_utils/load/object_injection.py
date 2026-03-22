import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

OBJECT_KW = "object"
INSTANTIATE_KW = "instantiate"


def _load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_object(object_params: dict) -> Any:
    """Load and instantiate an object from a class path and kwargs.

    Loads a Python object from the class path given as a parameter.

    Args:
        object_params: Dictionary with keys:
            - ``class``: Dotted path to the class (e.g. ``sklearn.model_selection.KFold``).
            - ``kwargs``: Keyword arguments passed to the constructor.

    Returns:
        Instantiated Python object.
    """
    model_class = object_params["class"]
    model_kwargs = object_params.get("kwargs", {})
    python_object = _load_obj(model_class)(**model_kwargs)
    return python_object


def load_estimator(object_params: dict) -> Any:
    """Load an sklearn-compatible estimator and validate its interface.

    Args:
        object_params: Dictionary with ``class`` and ``kwargs`` keys.

    Raises:
        AttributeError: If the returned object lacks ``.fit`` or ``.predict``.

    Returns:
        sklearn-compatible estimator.
    """
    model_class = object_params["class"]
    model_kwargs = object_params.get("kwargs", {})
    estimator = _load_obj(model_class)(**model_kwargs)
    if getattr(estimator, "fit", None) is None:
        raise AttributeError("Model object must have a .fit method")
    if getattr(estimator, "predict", None) is None:
        raise AttributeError("Model object must have a .predict method.")
    return estimator
