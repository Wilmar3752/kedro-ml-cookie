import importlib
import logging

import pandas as pd
from feature_engine.encoding import OneHotEncoder, OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)

from packages.modelling.transformers.columns_selector import ColumnsSelector

logger = logging.getLogger(__name__)


class AutoConfigFeaturePreprocessors:
    """Automatically inject feature variable lists into pipeline preprocessing steps.

    When building an sklearn Pipeline from a YAML config, many transformers
    need explicit ``variables`` lists (e.g. ``MeanMedianImputer(variables=[...])``)
    that depend on the actual data schema. This mixin inspects the input
    DataFrame and fills in those lists automatically, so the YAML config can
    leave ``variables:`` empty and have them resolved at fit-time.

    Supported transformers
    ----------------------
    - ``ColumnsSelector``         → injected with ``params["features"]["columns"]``
    - ``CategoricalImputer``      → injected with categorical columns that have NaNs
    - ``MeanMedianImputer``       → injected with numerical columns that have NaNs
    - ``AddMissingIndicator``     → injected with categorical columns that have NaNs
    - ``RareLabelEncoder``        → injected with all categorical columns
    - ``OrdinalEncoder``          → injected with all categorical columns
    - ``OneHotEncoder``           → injected with all categorical columns
    """

    DEFAULT_PREPROCESSORS = [
        CategoricalImputer,
        RareLabelEncoder,
        OrdinalEncoder,
        OneHotEncoder,
        AddMissingIndicator,
        MeanMedianImputer,
    ]

    def configure_feature_processing_params(
        self,
        X: pd.DataFrame,
        params: dict,
    ) -> dict:
        """Detect feature types in ``X`` and inject them into known pipeline steps.

        Args:
            X: Input dataset used to infer feature types.
            params: Pipeline configuration dictionary (modified in-place).

        Returns:
            Updated ``params`` dictionary with ``variables`` injected for each step.
        """
        features = params["features"]["columns"]
        variable_lists = self._identify_columns(X[features])

        for step_key in params.get("pipeline", {}):
            self._configure_single_pipeline_step(step_key, params, variable_lists)

        return params

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _configure_single_pipeline_step(
        self, step_key: str, params: dict, variable_lists: dict
    ) -> None:
        """Detect transformer class and assign appropriate variable lists."""
        cls = self._load_transformer_class(step_key, params)
        if cls is None:
            return
        self._ensure_kwargs_exist(step_key, params)
        self._assign_variables_to_transformer(step_key, params, variable_lists, cls)

    def _load_transformer_class(self, step_key: str, params: dict):
        """Dynamically import the transformer class from its dotted path."""
        step = params["pipeline"][step_key]
        class_path = step.get("class")
        if not class_path:
            logger.error(f"No class found for step '{step_key}'")
            return None
        try:
            module_path = ".".join(class_path.split(".")[:-1])
            class_name = class_path.split(".")[-1]
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load class {class_path} for step '{step_key}': {e}")
            return None

    def _ensure_kwargs_exist(self, step_key: str, params: dict) -> None:
        if "kwargs" not in params["pipeline"][step_key]:
            params["pipeline"][step_key]["kwargs"] = {}

    def _assign_variables_to_transformer(
        self, step_key: str, params: dict, variable_lists: dict, cls
    ) -> None:
        """Route each transformer to the correct variable list."""
        if issubclass(cls, ColumnsSelector):
            self._configure_columns_selector(step_key, params)
        elif issubclass(cls, AddMissingIndicator):
            self._configure_missing_indicator(step_key, params, variable_lists)
        elif issubclass(cls, MeanMedianImputer):
            self._configure_mean_median_imputer(step_key, params, variable_lists)
        elif issubclass(cls, CategoricalImputer):
            self._configure_categorical_imputer(step_key, params, variable_lists)
        elif issubclass(cls, (RareLabelEncoder, OrdinalEncoder, OneHotEncoder)):
            self._configure_encoders(step_key, params, variable_lists)
        else:
            logger.debug(f"Transformer {cls} for step '{step_key}' not auto-configured.")

    def _configure_columns_selector(self, step_key: str, params: dict) -> None:
        params["pipeline"][step_key]["kwargs"]["columns"] = params["features"]["columns"]

    def _configure_missing_indicator(
        self, step_key: str, params: dict, variable_lists: dict
    ) -> None:
        categorical_with_na = variable_lists["categorical_vars_with_na"]
        if categorical_with_na:
            params["pipeline"][step_key]["kwargs"]["variables"] = categorical_with_na
            return
        # Fallback: use first categorical variable when none have NaNs
        categorical_vars = variable_lists["categorical_vars"]
        if categorical_vars:
            logger.info(
                f"No categorical NaNs for '{step_key}'. Using '{categorical_vars[0]}' as fallback."
            )
            params["pipeline"][step_key]["kwargs"]["variables"] = [categorical_vars[0]]
        else:
            logger.warning(
                f"No categorical variables found for '{step_key}'. Using first feature as fallback."
            )
            params["pipeline"][step_key]["kwargs"]["variables"] = [
                params["features"]["columns"][0]
            ]

    def _configure_mean_median_imputer(
        self, step_key: str, params: dict, variable_lists: dict
    ) -> None:
        params["pipeline"][step_key]["kwargs"]["variables"] = variable_lists[
            "numeric_vars_with_na"
        ]

    def _configure_categorical_imputer(
        self, step_key: str, params: dict, variable_lists: dict
    ) -> None:
        categorical_to_impute = variable_lists["categorical_vars_with_na"]
        if not categorical_to_impute:
            categorical_to_impute = variable_lists["categorical_vars"]
        params["pipeline"][step_key]["kwargs"]["variables"] = categorical_to_impute

    def _configure_encoders(
        self, step_key: str, params: dict, variable_lists: dict
    ) -> None:
        params["pipeline"][step_key]["kwargs"]["variables"] = variable_lists[
            "categorical_vars"
        ]

    def _identify_columns(self, data: pd.DataFrame) -> dict:
        """Classify columns by dtype and missing-value status.

        Args:
            data: Subset of the DataFrame restricted to the feature columns.

        Returns:
            Dict with keys:
            - ``vars_with_na``: all columns with at least one NaN.
            - ``numeric_vars``: all numeric columns.
            - ``numeric_vars_with_na``: numeric columns with NaNs.
            - ``categorical_vars``: object-dtype columns.
            - ``categorical_vars_with_na``: object-dtype columns with NaNs.
        """
        vars_with_na = [col for col in data.columns if data[col].isnull().any()]

        numeric_vars = [
            col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])
        ]
        numeric_vars_na = [col for col in numeric_vars if col in vars_with_na]

        categorical_vars = [
            col for col in data.columns if data[col].dtype == object
        ]
        categorical_vars_na = [col for col in categorical_vars if col in vars_with_na]

        return {
            "vars_with_na": vars_with_na,
            "numeric_vars": numeric_vars,
            "numeric_vars_with_na": numeric_vars_na,
            "categorical_vars": categorical_vars,
            "categorical_vars_with_na": categorical_vars_na,
        }
