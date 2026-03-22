import warnings

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Select a fixed set of columns from a DataFrame.

    This transformer selects columns from a pandas DataFrame and is useful
    as the first step in a scikit-learn pipeline to restrict input features
    before any further transformations.

    Attributes:
        columns: List of column names to retain.

    Examples:
        >>> import pandas as pd
        >>> selector = ColumnsSelector(columns=["age", "fare"])
        >>> selector.fit_transform(df)
    """

    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnsSelector":
        """No-op fit — required by the sklearn transformer interface.

        Args:
            X: Input DataFrame.
            y: Ignored.

        Returns:
            self
        """
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing only ``self.columns``.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame restricted to the selected columns.

        Raises:
            ValueError: If ``X`` is not a pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame to preserve column and index information."
            )
        return X[self.columns]
