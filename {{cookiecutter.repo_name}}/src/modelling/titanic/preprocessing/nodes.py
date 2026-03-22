import logging
import typing as tp

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_titanic(
    df: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> pd.DataFrame:
    """Select and lightly clean the Titanic dataset.

    Steps:
    - Retain only the modelling-relevant columns (from params).
    - Cast ``survived`` to ``int``.
    - Reset the index.

    Missing values in ``age`` and ``embarked`` are intentionally left for the
    sklearn preprocessing pipeline (``MeanMedianImputer`` + ``CategoricalImputer``)
    to handle at fit-time. This keeps the feature engineering logic in a single,
    reproducible place.

    Args:
        df: Raw Titanic DataFrame from the catalog (``titanic_raw``).
        params: Data processing parameters. Expected keys:
            - ``target``: Name of the target column.
            - ``feature_columns``: List of feature column names to keep.

    Returns:
        Preprocessed DataFrame with target and feature columns only.
    """
    target = params["target"]
    feature_columns = params["feature_columns"]

    logger.info(f"Preprocessing Titanic data: input shape {df.shape}")

    cols = [target] + feature_columns
    df = df[cols].copy()
    df[target] = df[target].astype(int)
    df = df.reset_index(drop=True)

    logger.info(f"Preprocessed shape: {df.shape}")
    logger.info(f"Missing values per column:\n{df.isnull().sum().to_string()}")
    logger.info(f"Class distribution:\n{df[target].value_counts().to_string()}")

    return df


def split_data(
    df: pd.DataFrame,
    params: tp.Dict[str, tp.Any],
) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the preprocessed dataset into stratified train and test sets.

    Args:
        df: Preprocessed Titanic DataFrame.
        params: Data processing parameters. Expected keys:
            - ``target``: Name of the target column.
            - ``test_size``: Fraction of rows assigned to the test set.
            - ``random_state``: Random seed for reproducibility.

    Returns:
        Tuple of (train DataFrame, test DataFrame), both with a reset integer
        RangeIndex required by the downstream cross-validation logic.
    """
    target = params["target"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    df_train = X_train.copy()
    df_train[target] = y_train
    df_train = df_train.reset_index(drop=True)

    df_test = X_test.copy()
    df_test[target] = y_test
    df_test = df_test.reset_index(drop=True)

    logger.info(
        f"Split: train={df_train.shape}, test={df_test.shape} "
        f"(test_size={test_size}, seed={random_state})"
    )
    return df_train, df_test
