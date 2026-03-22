import typing as tp

import pandas as pd

from ._base import ReportElementIdentifier

_TColumnName = tp.Union[str, int]
_TSortingConfig = tp.Optional[tp.List[tp.Tuple[_TColumnName, str]]]


class Table(ReportElementIdentifier):
    def __init__(
        self,
        table: tp.Union[pd.Series, pd.DataFrame],
        columns: tp.Optional[tp.List[str]] = None,
        precision: tp.Optional[int] = 4,
        title: tp.Optional[str] = None,
        width: float = 80,
        show_index: bool = True,
    ) -> None:
        super().__init__(report_element=table)
        self.table = table
        self.columns = columns
        self.precision = precision
        self.title = title
        self.width = width
        self.show_index = show_index

    def _repr_html_(self) -> str:
        return self.table._repr_html_()
