from kedro.pipeline import Pipeline

from modelling.california.pipeline_registry import register_pipelines as california_pipelines
from modelling.titanic.pipeline_registry import register_pipelines as titanic_pipelines


def register_pipelines() -> dict[str, Pipeline]:
    """Register all project pipelines.

    Delegates to each model's own registry and merges the results.

    Models
    ------
    titanic    — binary classification (BinaryClassifierSklearnPipeline).
    california — regression (RegressionSklearnPipeline).

    Returns:
        Mapping from pipeline name to ``Pipeline`` object.
    """
    titanic = titanic_pipelines()
    california = california_pipelines()

    all_pipelines = {**titanic, **california}

    # __default__ runs every model end-to-end
    all_pipelines["__default__"] = titanic["e2e_titanic"] + california["e2e_california"]

    return all_pipelines
