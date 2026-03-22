from kedro.pipeline import Pipeline, node

from modelling.titanic.preprocessing.nodes import preprocess_titanic, split_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the preprocessing pipeline.

    Nodes
    -----
    1. ``preprocess_titanic`` — selects relevant columns and cleans dtypes.
    2. ``split_data``         — stratified train / test split.

    The raw dataset is read directly from the catalog (``titanic_raw``), which
    points to the committed CSV at ``data/01_raw/titanic.csv``.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=preprocess_titanic,
                inputs=["titanic_raw", "params:titanic.data_processing"],
                outputs="titanic_preprocessed",
                name="preprocess_titanic_node",
                tags=["preprocessing"],
            ),
            node(
                func=split_data,
                inputs=["titanic_preprocessed", "params:titanic.data_processing"],
                outputs=["titanic_train", "titanic_test"],
                name="split_data_node",
                tags=["preprocessing"],
            ),
        ]
    )
