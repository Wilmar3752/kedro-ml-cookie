import shap
from sklearn.pipeline import Pipeline

from packages.python_utils.typing.tensors import Tensor


class BaseShapComputation:
    """Mixin that adds SHAP value computation to an sklearn-compatible model.

    Splits the fitted pipeline into a preprocessor and a tree-based estimator,
    then uses ``shap.TreeExplainer`` to compute SHAP values on a sample of the
    training data for computational efficiency.
    """

    def compute_shap_values(
        self, model_pipeline: Pipeline, df: Tensor, sample_size: int = 500
    ) -> dict:
        """Compute SHAP values for a fitted sklearn Pipeline.

        Args:
            model_pipeline: A fitted sklearn ``Pipeline`` whose last step is a
                tree-based estimator (LightGBM, XGBoost, RandomForest …).
            df: Input data used to compute SHAP values (typically the training set).
            sample_size: Number of rows to sample for SHAP computation. Smaller
                values speed up computation; larger values improve accuracy.

        Returns:
            Dict with keys:
            - ``shap_values``: ``shap.Explanation`` object.
            - ``estimator``: The final estimator extracted from the pipeline.
            - ``preprocessor``: All pipeline steps except the last one.
        """
        # Decompose pipeline into preprocessor + final estimator
        preprocessor = model_pipeline[:-1]
        estimator = model_pipeline[-1]

        # Transform training data through the preprocessor
        X_transformed = preprocessor.transform(df)

        # Sample for computational efficiency
        X_sample = X_transformed.sample(min(sample_size, len(X_transformed)))

        # Compute SHAP values using TreeExplainer (fast for tree-based models)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(X_sample)

        return {
            "shap_values": shap_values,
            "estimator": estimator,
            "preprocessor": preprocessor,
        }
