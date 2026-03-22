from kedro.pipeline import Pipeline, node

from modelling.titanic.preprocessing.nodes import preprocess_titanic, split_data


def create_pipeline(**kwargs) -> Pipeline:
    """Create the California Housing preprocessing pipeline.

    Nodes
    -----
    1. ``preprocess_california`` — selects feature columns and resets index.
    2. ``split_data``            — random train / test split (KFold-compatible).

    The raw dataset is read directly from the catalog (``california_raw``),
    committed at ``data/01_raw/california_housing.csv``.

    Returns:
        Kedro ``Pipeline`` object.
    """
    return Pipeline(
        [
            node(
                func=preprocess_titanic,
                inputs=["california_raw", "params:california.data_processing"],
                outputs="california_preprocessed",
                name="preprocess_california_node",
                tags=["california", "preprocessing"],
            ),
            node(
                func=split_data,
                inputs=["california_preprocessed", "params:california.data_processing"],
                outputs=["california_train", "california_test"],
                name="california_split_data_node",
                tags=["california", "preprocessing"],
            ),
        ]
    )
