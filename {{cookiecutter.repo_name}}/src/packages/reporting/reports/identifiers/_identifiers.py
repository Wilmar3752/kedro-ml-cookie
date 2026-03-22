import typing as tp

import pandas as pd
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure

from ._base import ReportElementIdentifier
from ._code import Code
from ._section_header import SectionHeader
from ._table import Table

SUPPORTED_COMMON_TYPES = (MatplotlibFigure, PlotlyFigure, pd.DataFrame)
SUPPORTED_IDENTIFIERS = (Code, Table, ReportElementIdentifier)
SUPPORTED_RENDERING_AGNOSTIC_TYPES = SUPPORTED_COMMON_TYPES + SUPPORTED_IDENTIFIERS


TRenderingAgnosticContent = tp.Union[
    MatplotlibFigure,
    PlotlyFigure,
    pd.DataFrame,
    Code,
    Table,
]
TRenderingAgnosticDictKey = tp.Union[str, int]
TRenderingAgnosticDictValue = tp.Union[
    TRenderingAgnosticDictKey,
    TRenderingAgnosticContent,
    tp.List[TRenderingAgnosticContent],
]
TRenderingAgnosticDict = tp.Dict[
    TRenderingAgnosticDictKey,
    tp.Union[
        TRenderingAgnosticDictValue,
        "TRenderingAgnosticDict",
    ],
]

__all__ = [
    "Code",
    "Table",
    "SectionHeader",
    "ReportElementIdentifier",
    "MatplotlibFigure",
    "PlotlyFigure",
    "SUPPORTED_COMMON_TYPES",
    "SUPPORTED_IDENTIFIERS",
    "SUPPORTED_RENDERING_AGNOSTIC_TYPES",
    "TRenderingAgnosticContent",
    "TRenderingAgnosticDict",
    "TRenderingAgnosticDictKey",
]
