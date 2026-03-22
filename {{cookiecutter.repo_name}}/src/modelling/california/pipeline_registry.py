from kedro.pipeline import Pipeline

from modelling.california.hypertune.pipeline import create_pipeline as hypertune
from modelling.california.preprocessing.pipeline import create_pipeline as preprocessing
from modelling.california.reporting.pipeline import create_pipeline as reporting


def register_pipelines() -> dict[str, Pipeline]:
    """Register California Housing (regression) pipelines.

    Returns:
        california_preprocessing, california_hypertune, california_reporting, e2e_california.
    """
    preprocessing_pipe = preprocessing()
    hypertune_pipe = hypertune()
    reporting_pipe = reporting()
    e2e = preprocessing_pipe + hypertune_pipe + reporting_pipe

    return {
        "california_preprocessing": preprocessing_pipe,
        "california_hypertune": hypertune_pipe,
        "california_reporting": reporting_pipe,
        "e2e_california": e2e,
    }
