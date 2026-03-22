from kedro.pipeline import Pipeline

from modelling.titanic.hypertune.pipeline import create_pipeline as hypertune
from modelling.titanic.preprocessing.pipeline import create_pipeline as preprocessing
from modelling.titanic.reporting.pipeline import create_pipeline as reporting


def register_pipelines() -> dict[str, Pipeline]:
    """Register Titanic (binary classification) pipelines.

    Returns:
        titanic_preprocessing, titanic_hypertune, titanic_reporting, e2e_titanic.
    """
    preprocessing_pipe = preprocessing()
    hypertune_pipe = hypertune()
    reporting_pipe = reporting()
    e2e = preprocessing_pipe + hypertune_pipe + reporting_pipe

    return {
        "titanic_preprocessing": preprocessing_pipe,
        "titanic_hypertune": hypertune_pipe,
        "titanic_reporting": reporting_pipe,
        "e2e_titanic": e2e,
    }
